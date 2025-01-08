from torchvision import transforms
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.cuda.amp import GradScaler
from robust_facecloak.attacks.worker.differential_color_functions import rgb2lab_diff, ciede2000_diff
import random
class PANAttacker():
    def __init__(self, lambda_D=0.1, lambda_S=10, omiga=0.5, step_size=1, k=2, radius=11, args=None, x_range=[-1,1], steps=1, mode = "S", use_val = "last", trans = None):
        self.lambda_D = lambda_D
        self.lambda_S = torch.tensor(lambda_S, requires_grad=True)
        self.lambda_S_rate = lambda_S * 1e-1
        self.use_gau = True if args.gau_kernel_size > 0 else False
        self.omiga = torch.tensor(omiga, requires_grad=True)
        self.omiga_rate = omiga * 1e-1
        self.hpara_update_interval = args.hpara_update_interval
        self.k = k
        self.alpha = (step_size)/(0.5*255)
        self.radius = (radius-0.25)/(0.5*255)
        self.radius_d = (args.radius_d-0.25)/(0.5*255)
        self.dynamic_mode = args.dynamic_mode
        assert self.dynamic_mode in ['L_only',"multi","L+m",""]
        print(f"lambda_D: {lambda_D}, lambda_S: {lambda_S}, omiga: {omiga}")
        print(f'use Ltype:{args.Ltype}')
        print(f'r:{radius},rd:{args.radius_d}')
        print(f'max_L:{args.max_L},min_L:{args.min_L}')
        print(f'dynamic_mode:{self.dynamic_mode}')
        print(f'hpara_update_interval:{self.hpara_update_interval}')
        self.random_start = args.attack_pgd_random_start
        self.weight_dtype = torch.bfloat16  # 默认类型
        self.left = x_range[0]
        self.right = x_range[1]
        self.norm_type = 'l-infty'
        self.steps = steps
        self.mode = mode
        self.use_val = use_val
        self.noattack = radius == 0. or steps == 0 or step_size == 0.
        self.args = args
        self.time_select = 1
        self.pertubation_D = None
        self.pertubation_S = None
        self.trans = trans
        self.normalizer = transforms.Normalize([0.5*255]*3,[0.5*255]*3)
        self.inv_normalizer = transforms.Normalize(
            mean=[-1]*3,
            std=[1/(0.5*255)]*3
        )
        
        # self.ciede_max = 2500
        self.seed = random.randint(0, 2**32 - 1)
        self.gau_filter = transforms.GaussianBlur(kernel_size=args.gau_kernel_size,)
        self.step_cnt = 0
        self.device = torch.device("cuda")
        if args.mixed_precision in ["fp32",'no']:
            self.weight_dtype = torch.float32
        elif args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
    def update_seed(self):
        self.seed = random.randint(0, 2**32 - 1)

    def attack(self, f, ori_image, vae, tokenizer, noise_scheduler):
        if self.noattack:
            print("attacker no need to defend")
            return 0, 0
            

        device=torch.device("cuda")
        ori_image = self.normalizer(ori_image)
        ori_image = ori_image.detach().clone().to(device, dtype=self.weight_dtype)

        if self.pertubation_S is None:
            perturbed_data_in = ori_image.clone()
        else:
            perturbed_data_in = (self.pertubation_S + ori_image)
        # print(f'diff:{ori_image - perturbed_data_in}')

        f = [f[0].to(device, dtype=self.weight_dtype), f[1].to(device, dtype=self.weight_dtype)]
        vae.to(device, dtype=self.weight_dtype)
        perturbed_data = perturbed_data_in.detach().clone().to(device, dtype=self.weight_dtype)
        # ori_image = deepcopy(perturbed_data).to(device)
        # print(f'ori_image[0]:{ori_image[0]}')
        
        # batch_size = ori_image.size(0)
        # random start部分操作逻辑未设计
        if self.random_start:
            raise Exception("random start not implemented yet")
            r=self.radius
            initial_pertubations = torch.zeros_like(ori_image).uniform_(-r, r).to(device)
            adv_image = perturbed_data+initial_pertubations
            perturbed_data = adv_image - self._clip_(adv_image, ori_image)
        else:
            initial_pertubations = torch.zeros_like(ori_image).to(device,dtype=self.weight_dtype)

        if self.pertubation_S is None:
        # if True:
            pertubation_data_D = perturbed_data.clone()
            pertubation_data_D.require_grad = True
            pertubation_data_S = perturbed_data.clone()
            pertubation_data_S.require_grad = True
            if self.random_start:
                r=self.radius
                initial_pertubations = torch.zeros_like(ori_image).uniform_(-r, r).to(device,dtype=self.weight_dtype)
                adv_image = perturbed_data+initial_pertubations
                perturbed_data = adv_image - self._clip_(adv_image, ori_image)
            else:
                initial_pertubations = torch.zeros_like(ori_image).to(device,dtype=self.weight_dtype)
            self.pertubation_S = initial_pertubations.clone()
            self.pertubation_D = initial_pertubations.clone()

        pertubation_data_D = self.pertubation_D + ori_image
        pertubation_data_S = self.pertubation_S + ori_image

        # perturbed_data_in_traned = self.trans(perturbed_data_in).to(device, dtype=self.weight_dtype)
        
        loss_D = "nopudate"
        loss_S = "nopudate"
        # if self.use_gau == True:
        #     pertubation_data_D = self.gau_filter(pertubation_data_D)
        #     pertubation_data_S = self.gau_filter(pertubation_data_S)

        pertubation_data_D_in = pertubation_data_D.clone()
        pertubation_data_S_in = pertubation_data_S.clone()

        D_grad_list = []
        S_grad_list = []
        tmp_noise_D = torch.zeros_like(ori_image).to(device,dtype=self.weight_dtype)
        tmp_noise_S = torch.zeros_like(ori_image).to(device,dtype=self.weight_dtype)
        for i in range(self.steps):
            self.step_cnt+=1
            # if self.use_gau == True:
            #     # print(f'before trans:{pertubation_data_S_in[0]}')
            #     use_pertubation_data_D = self.trans(pertubation_data_D_in).to(device, dtype=self.weight_dtype) + tmp_noise_D
            #     use_pertubation_data_S = self.trans(pertubation_data_S_in).to(device, dtype=self.weight_dtype) + tmp_noise_S
            #     ori_image_tran = self.trans(ori_image).to(device, dtype=self.weight_dtype)
            #     # print(f'after trans:{use_pertubation_data_S[0]}')
            # else:
            #     use_pertubation_data_D = (pertubation_data_D_in + tmp_noise_D).clone()
            #     use_pertubation_data_S = (pertubation_data_S_in + tmp_noise_S).clone()
            #     ori_image_tran = ori_image
            use_pertubation_data_D = (pertubation_data_D_in + tmp_noise_D).clone()
            use_pertubation_data_S = (pertubation_data_S_in + tmp_noise_S).clone()
            # 提前看一步
            # use_pertubation_data_D, _, _ = self.update_pertubation_data_D(f, use_pertubation_data_D, ori_image, vae, tokenizer, noise_scheduler)
            # use_pertubation_data_S, _, _ = self.update_pertubation_S(f, use_pertubation_data_S, use_pertubation_data_D, ori_image, vae, tokenizer, noise_scheduler)  
            # print(f'use_pertubation_data_D - ori_image{self.get_Linfty_norm(use_pertubation_data_D - ori_image)}')
            self.update_seed()
            use_pertubation_data_D, loss_D, D_grad = self.update_pertubation_data_D(f, use_pertubation_data_D, ori_image, vae, tokenizer, noise_scheduler)
            use_pertubation_data_S, loss_S, S_grad = self.update_pertubation_S(f, use_pertubation_data_S, use_pertubation_data_D, ori_image, vae, tokenizer, noise_scheduler)
            D_grad_list.append(D_grad.clone())
            S_grad_list.append(S_grad.clone())
            # print(S_grad)
            # tmp_noise_D = use_pertubation_data_D - pertubation_data_D_in
            # tmp_noise_S = use_pertubation_data_S - pertubation_data_S_in
            # tmp_noise_D = use_pertubation_data_D - self.trans(pertubation_data_D_in).to(device, dtype=self.weight_dtype)
            # tmp_noise_S = use_pertubation_data_S - self.trans(pertubation_data_S_in).to(device, dtype=self.weight_dtype)
            tmp_noise_D = tmp_noise_D - D_grad * self.alpha
            tmp_noise_S = tmp_noise_S - S_grad * self.alpha
            # print(f"S max change: {self.get_Linfty_norm(pertubation_data_S - ori_image_tran)}")
            # print(f'pertubation_data_S: {pertubation_data_S[0]}')
            print(f'tmp_noise_S: {self.get_Linfty_norm(tmp_noise_S)}')
            print(f'epoch: {i}, loss_S: {loss_S:.4f}, loss_D: {loss_D: .4f}')
        D_grad_sum = torch.sum(torch.stack(D_grad_list), dim=0)
        S_grad_sum = torch.sum(torch.stack(S_grad_list), dim=0)
        # D_grad_sum[torch.abs(D_grad_sum) < self.steps/3] = 0
        # S_grad_sum[torch.abs(S_grad_sum) < self.steps/3] = 0
        # D_grad = torch.clamp(D_grad_sum, min=-1, max=1)
        # S_grad = torch.clamp(S_grad_sum, min=-1, max=1)
        D_grad = D_grad_sum
        S_grad = S_grad_sum

        # pertubation_data_D = pertubation_data_D - self.alpha * D_grad
        # pertubation_data_S = pertubation_data_S - self.alpha * S_grad
        pertubation_data_D = pertubation_data_D_in + tmp_noise_D
        pertubation_data_S = pertubation_data_S_in + tmp_noise_S
        pertubation_data_D = self._clip_(pertubation_data_D, ori_image, mode="D")
        pertubation_data_S = self._clip_(pertubation_data_S, ori_image)

        assert self.mode in ["S", "D"]
        assert self.use_val in ["best", "last"]
        # print(f'pertubation_data_S_in-ori_image:{self.get_Linfty_norm(pertubation_data_S_in - ori_image)}')
        if self.mode == "S":
            use_pertubation_data = pertubation_data_S 
            used_pertubation_data_in = pertubation_data_S_in
            loss = loss_S 
        elif self.mode == "D":
            use_pertubation_data = pertubation_data_D
            used_pertubation_data_in = pertubation_data_D_in
            loss = loss_D 
        self.pertubation_S.add_(pertubation_data_S-pertubation_data_S_in)
        self.pertubation_D.add_(pertubation_data_D-pertubation_data_D_in)
        use_pertubations = use_pertubation_data - ori_image
        # adding_noise = use_pertubation_data - ori_image

        # print(f"find a better pertubation_{self.mode} , max val is {self.get_Linfty_norm(self.inv_normalizer(use_pertubations))-127.5}")
        # print(f"use_per is :{use_pertubations[2]}")
        # use_pertubation_data = use_pertubation_data.detech_()
        append_noise = use_pertubation_data - used_pertubation_data_in
        result = use_pertubation_data.detach_()
        result_D = pertubation_data_D.detach_()
        # normalizer = transforms.Normalize([0.5*255]*3,[0.5*255]*3) 逆变换)
        # print(f'before inv:{result[0]}')
        result_unnor = self.inv_normalizer(result)
        result_D_unnor = self.inv_normalizer(result_D)
        # print(f'after inv:{result_unnor[0]}')
        print(f'append_noise:{self.get_Linfty_norm(append_noise)}')
        return result_unnor, result_D_unnor, loss

    def certi(self, models, adv_x, vae, noise_scheduler, input_ids, weight_dtype=None, target_tensor=None,timesteps = None):
        unet, text_encoder = models
        unet.zero_grad()
        text_encoder.zero_grad()
        device = torch.device("cuda")
        # print(f'input adv_x:{adv_x[0]}')
        adv_latens = vae.encode(adv_x.to(device, dtype=weight_dtype)).latent_dist.sample()
        adv_latens = adv_latens * vae.config.scaling_factor
        noise = torch.randn_like(adv_latens)
        bsz = adv_latens.shape[0]
        if timesteps is None:
            torch.manual_seed(self.seed)
            timesteps = torch.randint(0, int(noise_scheduler.config.num_train_timesteps*self.time_select), (bsz,), device=adv_latens.device)
            timesteps = timesteps.long()
        # print(f'noise:{noise[0]}')
        # print(f'adv_latens:{adv_latens[0]}')
        noisy_latents = noise_scheduler.add_noise(adv_latens, noise, timesteps)
        encoder_hidden_states = text_encoder(input_ids.to(device))[0]
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # print(f'model_pred:{model_pred[0]}')
        # print("Model pred grad_fn:", model_pred.grad_fn) 
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(adv_latens, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        # print(f'target_tensor:{target[0]}')
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        # print(f'loss fun return loss:{loss}')
        if target_tensor is not None:
            timesteps = timesteps.to(device)
            noisy_latents = noisy_latents.to(device)
            xtm1_pred = torch.cat(
                [
                    noise_scheduler.step(
                        model_pred[idx: idx + 1],
                        timesteps[idx: idx + 1],
                        noisy_latents[idx: idx + 1],
                    ).prev_sample
                    for idx in range(len(model_pred))
                ]
            )
            xtm1_target = noise_scheduler.add_noise(target_tensor, noise.to(device), (timesteps - 1).to(device))
            loss = loss - F.mse_loss(xtm1_pred, xtm1_target)
        del unet, text_encoder
        torch.cuda.empty_cache()
        return loss

    def get_loss_D(self, f, adv_image_tran, adv_image, ori_image, vae, tokenizer, noise_scheduler):
        input_ids = tokenizer(
            self.args.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(len(adv_image_tran), 1)

        loss_P = self.certi(f, adv_image_tran, vae, noise_scheduler, input_ids, weight_dtype=self.weight_dtype)
        pertubation_linf = self.get_pertubation_linf(adv_image,ori_image)
        print(f"loss_d linf: {pertubation_linf}")
        loss = - loss_P + (self.lambda_D * torch.abs(pertubation_linf))
        return loss

    def update_pertubation_data_D(self, f, Ped_data_D, ori_image, vae, tokenizer, noise_scheduler):
        # adv_image.requires_grad = True
        # adv_image = Ped_data_D.clone().detach()
        # adv_image.requires_grad = True
        # loss = self.get_loss_D(f, adv_image, ori_image, vae, tokenizer, noise_scheduler)
        # loss.backward()
        # grad_ml_alpha = self.alpha * adv_image.grad.sign()
        # # print(f"D:adv_image.grad.sign(): {adv_image.grad.sign()}")
        # adv_image_new = adv_image - grad_ml_alpha
        # adv_image_new = self._clip_(adv_image_new, ori_image, mode = "D")
        # # adv_image_new = adv_image_new.detach()
        # # del f
        # torch.cuda.empty_cache()
        # return adv_image_new, loss.item()
        if self.lambda_D == 0 and self.omiga == 0:
            print(f'pan_lambda_D is 0, no need to update pertubation_data_D')
            return Ped_data_D,0,torch.zeros_like(Ped_data_D)
        scaler = GradScaler()
        adv_image = Ped_data_D.clone().detach() 
        adv_image.requires_grad = True
        adv_image_tran = self.trans(adv_image) if self.use_gau else adv_image

        # 计算损失
        loss = self.get_loss_D(f, adv_image_tran, adv_image, ori_image, vae, tokenizer, noise_scheduler)
        
        # 缩放损失并反向传播
        # scaler.scale(loss).backward()
        loss.backward()
        # print(f'adv_image.grad.sign(): {adv_image.grad.sign()}')
        # 取消缩放梯度
        # scaler.unscale_(optimizer=None)  # 由于没有使用优化器，传 None
        # torch.nn.utils.clip_grad_norm_(adv_image, max_norm=1.0)  # 可选的：进行梯度裁剪
        
        # 继续更新
        grad_ml_alpha = self.alpha * adv_image.grad.sign()
        adv_image_new = adv_image - grad_ml_alpha
        adv_image_new = self._clip_(adv_image_new, ori_image, mode="D")
        out_grad = adv_image.grad.sign().clone()
        torch.cuda.empty_cache()
        return adv_image_new, loss.item(), out_grad

    def update_pertubation_S(self, f, pertubation_data_S, pertubation_data_D, ori_image, vae, tokenizer, noise_scheduler):
        # print(f'old pertubation_S: {pertubation_S[2]}')
        # pertubation_data_S.requires_grad = True
        scaler = GradScaler()
        adv_image_S = pertubation_data_S.clone().detach()
        adv_image_D = pertubation_data_D.clone().detach()
        adv_image_S.requires_grad = True
        adv_image_D.requires_grad = False
        adv_image_S_tran = self.trans(adv_image_S) if self.use_gau else adv_image_S
        adv_image_D_tran = self.trans(adv_image_D) if self.use_gau else adv_image_D
        input_ids = tokenizer(
            self.args.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(len(adv_image_S), 1)

        loss_P_S = self.certi(f, adv_image_S_tran, vae, noise_scheduler, input_ids, weight_dtype=self.weight_dtype)
        loss_P_D = self.certi(f, adv_image_D_tran, vae, noise_scheduler, input_ids, weight_dtype=self.weight_dtype)
        # if loss_P_D > loss_P_S and self.omiga > 0:
        #     self.lambda_S = 1e-3/(loss_P_D.item()-loss_P_S.item())
        #     print(f'update lambda_S: {self.lambda_S}')
        # 约束由Linfty更改为L3
        # pertubation_linf_S = torch.max(self.get_Linfty_norm(adv_image_S-ori_image))
        # print(f'adv_image_S: {adv_image_S}')
        pertubation_linf_S = self.get_pertubation_linf(adv_image_S,ori_image)
        # if pertubation_linf_S > 21160000:
        #     self.lambda_S = self.lambda_S * 1.1
        #     print(f'update lambda_S: {self.lambda_S}')
        # if pertubation_linf_S < 14440000:
        #     self.lambda_S = self.lambda_S * 0.9
        #     print(f'update lambda_S: {self.lambda_S}')
        # if self.step_cnt%5==0 and (self.args.max_L != 0 or self.args.min_L != 0):
        #     if pertubation_linf_S > self.args.max_L**self.k:
        #         self.lambda_S = self.lambda_S * 2
        #         print(f'update lambda_S: {self.lambda_S}')
        #         # self.omiga = self.omiga * 0.5
        #         # print(f'update omiga: {self.omiga}')
        #     if pertubation_linf_S < self.args.min_L**self.k:
        #         self.lambda_S = self.lambda_S * 0.8
        #         print(f'update lambda_S: {self.lambda_S}')
                # self.omiga = self.omiga * 1.2
                # print(f'update omiga: {self.omiga}')
        # print(f'lossPS: {loss_P_S}')
        # print(f'lossPD: {loss_P_D}')
        # print(f'abs(lossPS - lossPD): {torch.abs(loss_P_S - loss_P_D)}')
        # print(f'pertubation_linf_S: {pertubation_linf_S}')
        loss_con = torch.abs(pertubation_linf_S)
        loss_diff = torch.abs(loss_P_S - loss_P_D)**self.k
        # loss = - loss_P_S + self.lambda_S * loss_con + self.omiga * loss_diff
        if self.args.dynamic_mode in ['L+m','multi']:
            loss = - loss_P_S + self.lambda_S * loss_con + self.omiga * loss_diff - 0.5 * torch.log(self.lambda_S) - 0.5 * self.args.omiga_strength * torch.log(self.omiga)
        else:
            loss = - loss_P_S + self.lambda_S * loss_con + self.omiga * loss_diff
        print(f'loss_s compose: - loss_P_S:{- loss_P_S},lambda_LP:{self.lambda_S * loss_con}, omiga_diff:{(self.omiga * loss_diff):.8f}')

        loss.backward()
        self.update_lambda_S(pertubation_linf_S)
        self.update_omiga()
        
        # print("Gradient of adv_image_S:", adv_image_S.grad)
        # print(f'grad:{self.alpha * pertubation_S.grad.sign()[0]}')
        # print(f'now pertubation_S: {pertubation_S[0]}')
        grad_ml_alpha = self.alpha * adv_image_S.grad.sign()
        # print(f'old pertubation_S: {pertubation_S[2]}')
        # print(f' adv_image_S.grad.sign(): {adv_image_S.grad.sign()}')
        # print(f'grad_ml_alpha:{self.get_Linfty_norm(grad_ml_alpha)}')
        # print(f"old per max val is {self.get_Linfty_norm(adv_image_S - ori_image)}")
        adv_image_S_new = adv_image_S.clone()
        # print(f'adv_image_S_new: {self.get_Linfty_norm(adv_image_S_new)}')
        # print(f'before add: {self.get_Linfty_norm(adv_image_S_new - ori_image)}')
        # print(f'before add, adv_image_S_new: {adv_image_S_new[0][0][0][:10]}')
        adv_image_S_new.add_(- grad_ml_alpha)
        # print(f'after add, adv_image_S_new: {adv_image_S_new[0][0][0][:10]}')
        # tensor 原地操作

        # print(f'inner:{self.get_Linfty_norm(adv_image_S - grad_ml_alpha-ori_image)}')
        # 裁剪到0～255之间,并确保扰动没有超出范围
        # print(f'before clip: {self.get_Linfty_norm(adv_image_S_new - ori_image)}')
        adv_image_S_new = self._clip_(adv_image_S_new, ori_image)
        # print(f'new pertubation_S: {pertubation_S_new[2]}')
        # adv_image_S_new = adv_image_S_new.detach()
        # print(f'new pertubation_S: {pertubation_S[2]}')
        out_grad = adv_image_S.grad.sign().clone()
        # 清空loss的梯度
        
        torch.cuda.empty_cache()
        # print(f"new per max val is {self.get_Linfty_norm(adv_image_S_new - ori_image)}")
        return adv_image_S_new, loss.item(),out_grad

    def get_pertubation_linf(self, adv_image,ori_image,mode = None):
        # if mode == "S":
            # pertubation_linf = torch.max(self.get_Linfty_norm(adv_image-ori_image))
        result = torch.tensor(0.0, device=self.device)
        per_data = adv_image-ori_image
        # print(f'per_data: {per_data[0][0]}')
        pix_num = per_data.shape[2] * per_data.shape[3]
        per_data_inr = torch.clamp(per_data - self.radius, min=0)
        # pertubation_linf = torch.max(self.get_Linfty_norm(per_data))
        if 'L2' in self.args.Ltype:
            L2_n = torch.mean(self.get_L2_norm(per_data_inr)**self.k)
            result += L2_n
        if 'L1' in self.args.Ltype:
            L1_n = torch.mean(self.get_L1_norm(per_data)**self.k)
            result += L1_n
        if 'L0' in self.args.Ltype:
            L0_rho = torch.mean(self.get_rho_norm(per_data).to(dtype=self.weight_dtype)**self.k)
            result += L0_rho
        # L2_n = torch.mean(self.get_L2_norm(per_data_inr))
        # L1_n = torch.mean(self.get_L1_norm(per_data))
        # L0_rho = torch.mean(self.get_rho_norm(per_data).to(dtype=self.weight_dtype))
        if 'ciede2000' in self.args.Ltype:
            ciede2000_diff = self.get_ciede2000_diff(adv_image,ori_image)
            result += torch.mean(ciede2000_diff**self.k)
        # result += pertubation_linf
        # result += L2_n
        # result += L1_n

        # print(f'result: {result}')
        # result += L0_rho
        return result

    def get_Linfty_norm(self, images):
        abs_images = torch.abs(images)
        max_pixels_per_image, _ = torch.max(abs_images, dim=3)
        max_pixels_per_image, _ = torch.max(max_pixels_per_image, dim=2)
        Linfty_norm, _ = torch.max(max_pixels_per_image, dim=1)
        print(f'return Linfty_norm: {Linfty_norm}')
        return Linfty_norm
    
    def get_L1_norm(self, images):
        # inv_images = self.inv_normalizer(images).to(dtype=torch.int32)
        abs_images = torch.abs(images)
        L1_norm = torch.mean(abs_images, dim=[1, 2, 3])
        # print(abs_images[0][0])
        # print(f'L1_norm: {L1_norm}')
        return L1_norm
    
    def get_L2_norm(self, images):
        abs_images = torch.abs(images)
        L2_norm = torch.sqrt(torch.sum(abs_images ** 2, dim=[1, 2, 3]))
        return L2_norm
    
    def get_L3_norm(self, images):
        abs_images = torch.abs(images)
        L3_norm = torch.cbrt(torch.sum(abs_images ** 3, dim=[1, 2, 3]))
        return L3_norm

    def get_ciede2000_diff(self,advimgs,ori_imgs):
        device = torch.device('cuda')
        ori_imgs_0_1 = (ori_imgs+1)/2
        advimgs_0_1 = (advimgs+1)/2
        advimgs_0_1.clamp_(0,1)
        # print(f'ori_imgs_0_1.min:{ori_imgs_0_1.min()}, ori_imgs_0_1.max:{ori_imgs_0_1.max()}')
        # print(f'advimgs_0_1.min:{advimgs_0_1.min()}, advimgs_0_1.max:{advimgs_0_1.max()}')
        X_ori_LAB = rgb2lab_diff(ori_imgs_0_1,device)
        advimgs_LAB = rgb2lab_diff(advimgs_0_1,device)
        # print(f'advimgs: {advimgs}')
        # print(f'ori_imgs: {ori_imgs}')
        color_distance_map=ciede2000_diff(X_ori_LAB,advimgs_LAB,device)
        # print(color_distance_map)
        scores = torch.norm(color_distance_map.view(ori_imgs.shape[0],-1),dim=1)
        print(f'scores: {scores}')
        # mean_scores = torch.mean(scores)
        # 100
        return scores
    

    def update_lambda_S(self,pertubation_linf_S):
        if self.dynamic_mode == "L_only" or self.dynamic_mode == "L+m":
            if self.step_cnt%self.hpara_update_interval==0 and (self.args.max_L != 0 or self.args.min_L != 0):
                if pertubation_linf_S > self.args.max_L**self.k:
                    self.lambda_S = self.lambda_S * 2
                    print(f'update lambda_S: {self.lambda_S}')
                    # self.omiga = self.omiga * 0.5
                    # print(f'update omiga: {self.omiga}')
                if pertubation_linf_S < self.args.min_L**self.k:
                    self.lambda_S = self.lambda_S * 0.8
                    print(f'update lambda_S: {self.lambda_S}')
                self.lambda_S = self.lambda_S.detach()
                self.lambda_S.requires_grad=True
        if self.dynamic_mode == "multi":
            grad_ml_lambda = self.lambda_S_rate * self.lambda_S.grad.sign()
            self.lambda_S = self.lambda_S - grad_ml_lambda
            self.lambda_S = self.lambda_S.detach()
            self.lambda_S.requires_grad=True
            print(f'update lambda_S: {self.lambda_S}')
            self.lambda_S_rate = self.lambda_S * 0.1

    def update_omiga(self):
        if self.dynamic_mode == "multi" or self.dynamic_mode == "L+m":
            # if self.step_cnt%self.hpara_update_interval==0:
            grad_ml_omiga = self.omiga_rate * self.omiga.grad.sign()
            self.omiga = self.omiga - grad_ml_omiga
            self.omiga = self.omiga.detach()
            self.omiga.requires_grad=True
            print(f'update omiga: {self.omiga}')
            self.omiga_rate = self.omiga * 0.1

    def _clip_(self, adv_x, x, mode = None):
        # print(f"clip to {x[0]}")
        adv_x.to(dtype=torch.float32)
        x.to(dtype=torch.float32)
        adv_x = adv_x - x
        # if self.norm_type == 'l-infty':
        #     if mode == 'S':
        #         adv_x.clamp_(-self.radius, self.radius)
        # else:
        #     raise NotImplementedError
        if mode == "D":
            adv_x.clamp_(-(self.radius_d), (self.radius_d))
        else:
            adv_x.clamp_(-self.radius, self.radius)
        adv_x = adv_x + x
        
        # if mode == 'S':
        adv_x.clamp_(self.left, self.right)
        adv_x = adv_x.to(dtype=self.weight_dtype)
        # adv_x.clamp_(self.left, self.right)
        return adv_x