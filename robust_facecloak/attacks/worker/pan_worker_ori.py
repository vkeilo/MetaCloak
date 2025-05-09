import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.cuda.amp import GradScaler

class PANAttacker():
    def __init__(self, lambda_D=0.1, lambda_S=10, omiga=0.5, step_size=1, k=2, radius=11, args=None, x_range=[0,255], steps=1, mode = "D", use_val = "last"):
        self.lambda_D = lambda_D
        self.lambda_S = lambda_S
        self.omiga = omiga
        self.alpha = step_size
        self.k = k
        self.radius = radius
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
        if args.mixed_precision == "fp32":
            self.weight_dtype = torch.float32
        elif args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        
    def attack(self, f, perturbed_data_in, ori_image, vae, tokenizer, noise_scheduler):
        if self.noattack:
            print("attacker no need to defend")
            return perturbed_data_in, 0

        device=torch.device("cuda")
        f = [f[0].to(device, dtype=self.weight_dtype), f[1].to(device, dtype=self.weight_dtype)]
        vae.to(device, dtype=self.weight_dtype)
        perturbed_data = perturbed_data_in.detach().clone().to(device, dtype=self.weight_dtype)
        # ori_image = deepcopy(perturbed_data).to(device)
        # print(f'ori_image[0]:{ori_image[0]}')
        ori_image = ori_image.detach().clone().to(device, dtype=self.weight_dtype)
        # batch_size = ori_image.size(0)
        # random start部分操作逻辑未设计
        if self.random_start:
            r=self.radius
            initial_pertubations = torch.zeros_like(ori_image).uniform_(-r, r).to(device)
            adv_image = perturbed_data+initial_pertubations
            perturbed_data = adv_image - self._clip_(adv_image, ori_image, mode="D")
        else:
            initial_pertubations = torch.zeros_like(ori_image).to(device)
        # 此轮攻击的初始扰动都是0
        pertubation_data_D = perturbed_data.clone()
        pertubation_data_D.require_grad = True
        pertubation_data_S = perturbed_data.clone()
        pertubation_data_S.require_grad = True
        best_loss_S = float('inf')
        best_loss_D = float('inf')
        best_pertubation_data_S = perturbed_data.clone()
        best_pertubation_data_D = perturbed_data.clone()
        best_pertubation_data_S.require_grad = True
        best_pertubation_data_D.require_grad = True
        loss_D = "nopudate"
        loss_S = "nopudate"

        for i in range(self.steps):
            # print(f'step {i} :pertubation_data_S is {pertubation_data_S[0]}')
            # 更新扰动D
            pertubation_data_D, loss_D = self.update_pertubation_data_D(f, pertubation_data_D, ori_image, vae, tokenizer, noise_scheduler)
            if loss_D < best_loss_D:
                best_loss_D = loss_D
                # if mode == "D":
                    # print(f'pertubation_D, max val is {self.get_Linfty_norm(pertubations_D)}')
                    # print(f"find a better pertubation , max val is {self.get_Linfty_norm( pertubations_D.to('cpu') + perturbed_data.to('cpu') - ori_image.to('cpu') )}")
                best_pertubation_data_D = pertubation_data_D.clone()
            # 更新扰动S
            pertubation_data_S, loss_S = self.update_pertubation_S(f, pertubation_data_S, pertubation_data_D, ori_image, vae, tokenizer, noise_scheduler)
            # print(f"S max change: {self.get_Linfty_norm(pertubation_data_S.to('cpu') - ori_image.to('cpu'))}")
            print(f'epoch: {i}, loss_S: {loss_S:.4f}, loss_D: {loss_D: .4f}')
            if loss_S < best_loss_S:
                best_loss_S = loss_S
                # if self.mode == "S":
                    # print(f"find a better pertubation , max val is {self.get_Linfty_norm(pertubation_data_S.to('cpu') - ori_image.to('cpu'))}")
                best_pertubation_data_S = pertubation_data_S.clone()
            # 更新扰动S
        
        assert self.mode in ["S", "D"]
        assert self.use_val in ["best", "last"]

        if self.mode == "S":
            use_pertubation_data = pertubation_data_S if self.use_val == "last" else best_pertubation_data_S
            loss = loss_S if self.use_val == "last" else best_loss_S
        elif self.mode == "D":
            use_pertubation_data = pertubation_data_D if self.use_val == "last" else best_pertubation_data_D
            loss = loss_D if self.use_val == "last" else best_loss_D
        adding_noise = use_pertubation_data.to('cpu') - perturbed_data_in.to('cpu')
        # print(f"find a better pertubation_{self.mode} , max val is {self.get_Linfty_norm(use_pertubations)}")
        # print(f"use_per is :{use_pertubations[2]}")
        # use_pertubation_data = use_pertubation_data.detech_()
        return perturbed_data_in.to('cpu') + adding_noise.detach_().to('cpu'), loss

    def certi(self, models, adv_x, vae, noise_scheduler, input_ids, weight_dtype=None, target_tensor=None):
        unet, text_encoder = models
        unet.zero_grad()
        text_encoder.zero_grad()
        device = torch.device("cuda")

        adv_latens = vae.encode(adv_x.to(device, dtype=weight_dtype)).latent_dist.sample()
        adv_latens = adv_latens * vae.config.scaling_factor
        noise = torch.randn_like(adv_latens)
        bsz = adv_latens.shape[0]
        timesteps = torch.randint(0, int(noise_scheduler.config.num_train_timesteps*self.time_select), (bsz,), device=adv_latens.device)
        timesteps = timesteps.long()

        noisy_latents = noise_scheduler.add_noise(adv_latens, noise, timesteps)
        encoder_hidden_states = text_encoder(input_ids.to(device))[0]
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # print("Model pred grad_fn:", model_pred.grad_fn) 
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(adv_latens, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

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

    def get_loss_D(self, f, adv_image, ori_image, vae, tokenizer, noise_scheduler):
        input_ids = tokenizer(
            self.args.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(len(adv_image), 1)

        loss_P = self.certi(f, adv_image, vae, noise_scheduler, input_ids, weight_dtype=self.weight_dtype)
        # 取最大是否合适
        pertubation_linf = torch.mean(self.get_Linfty_norm(adv_image-ori_image))
        loss = - loss_P + (self.lambda_D * torch.abs(pertubation_linf)**self.k)
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
        scaler = GradScaler()
        adv_image = Ped_data_D.clone().detach()
        adv_image.requires_grad = True

        # 计算损失
        loss = self.get_loss_D(f, adv_image, ori_image, vae, tokenizer, noise_scheduler)
        
        # 缩放损失并反向传播
        scaler.scale(loss).backward()
        # print(f'adv_image.grad.sign(): {adv_image.grad.sign()}')
        # 取消缩放梯度
        # scaler.unscale_(optimizer=None)  # 由于没有使用优化器，传 None
        # torch.nn.utils.clip_grad_norm_(adv_image, max_norm=1.0)  # 可选的：进行梯度裁剪
        
        # 继续更新
        grad_ml_alpha = self.alpha * adv_image.grad.sign()
        adv_image_new = adv_image - grad_ml_alpha
        adv_image_new = self._clip_(adv_image_new, ori_image, mode="D")
        
        torch.cuda.empty_cache()
        return adv_image_new, loss.item()

    def update_pertubation_S(self, f, pertubation_data_S, pertubation_data_D, ori_image, vae, tokenizer, noise_scheduler):
        # print(f'old pertubation_S: {pertubation_S[2]}')
        # pertubation_data_S.requires_grad = True
        scaler = GradScaler()
        adv_image_S = pertubation_data_S.clone().detach()
        adv_image_D = pertubation_data_D.clone().detach()
        adv_image_S.requires_grad = True
        adv_image_D.requires_grad = False

        input_ids = tokenizer(
            self.args.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(len(adv_image_S), 1)

        loss_P_S = self.certi(f, adv_image_S, vae, noise_scheduler, input_ids, weight_dtype=self.weight_dtype)
        loss_P_D = self.certi(f, adv_image_D, vae, noise_scheduler, input_ids, weight_dtype=self.weight_dtype)

        # 
        pertubation_linf_S = torch.mean(self.get_Linfty_norm(adv_image_S-ori_image))
        loss = - loss_P_S + self.lambda_S * (torch.abs(pertubation_linf_S)**self.k) + self.omiga * (torch.abs(loss_P_S - loss_P_D)**self.k)
        # loss.backward()
        scaler.scale(loss).backward()
        # print("Gradient of adv_image_S:", adv_image_S.grad)
        # print(f'grad:{self.alpha * pertubation_S.grad.sign()[0]}')
        print(f'pertubation_linf_S: {pertubation_linf_S}')
        grad_ml_alpha = self.alpha * adv_image_S.grad.sign()
        # print(f'old pertubation_S: {pertubation_S[2]}')
        # print(f' adv_image_S.grad.sign(): {adv_image_S.grad.sign()}')
        adv_image_S_new = adv_image_S - grad_ml_alpha
        # print(f'inner:{self.get_Linfty_norm(adv_image_S - grad_ml_alpha-ori_image)}')
        # 裁剪到0～255之间,并确保扰动没有超出范围
        adv_image_S_new = self._clip_(adv_image_S_new, ori_image, mode='S')
        # print(f'new pertubation_S: {pertubation_S_new[2]}')
        # adv_image_S_new = adv_image_S_new.detach()
        # print(f'new pertubation_S: {pertubation_S[2]}')
        torch.cuda.empty_cache()
        # print(f"new per max val is {self.get_Linfty_norm(adv_image_S_new.to('cpu') - ori_image.to('cpu'))}")
        return adv_image_S_new, loss.item()

    def get_Linfty_norm(self, images):
        abs_images = torch.abs(images)
        max_pixels_per_image, _ = torch.max(abs_images, dim=3)
        max_pixels_per_image, _ = torch.max(max_pixels_per_image, dim=2)
        Linfty_norm, _ = torch.max(max_pixels_per_image, dim=1)
        return Linfty_norm

    def _clip_(self, adv_x, x, mode):
        # print(f"clip to {x[0]}")
        adv_x = adv_x - x
        if self.norm_type == 'l-infty':
            if mode == 'S':
                adv_x.clamp_(-self.radius, self.radius)
        else:
            raise NotImplementedError
        adv_x = adv_x + x
        adv_x.clamp_(self.left, self.right)
        return adv_x