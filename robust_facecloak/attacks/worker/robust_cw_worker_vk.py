import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

import random
# import lpips

class FieldLoss(torch.nn.Module):
    def __init__(self):
        super(FieldLoss, self).__init__()

    def forward(self, class2target_v_a,model_pred_latents,model_target_latents):
        bsz = len(model_pred_latents)//len(class2target_v_a)
        model_pred_latents_rsp = model_pred_latents.reshape([bsz,-1])
        model_target_latents_rsp = model_target_latents.reshape([bsz,-1])
        avg_impro_in_feild = 0
        for batch in range(bsz):
            pred_more = model_pred_latents_rsp[batch] - model_target_latents_rsp[batch]
            impro_in_feild = torch.dot(pred_more,class2target_v_a)
            avg_impro_in_feild += impro_in_feild/bsz
        # add -  is classv  no - is -classv
        return  - avg_impro_in_feild*avg_impro_in_feild

def low_pass_filter(tensor, cutoff=1):
    """
    对 bx4x64x64 的 tensor 进行低通滤波，滤去高频部分
    :param tensor: 输入张量 (bx4x64x64)
    :param cutoff: 低通滤波器的截止频率（越小，保留的低频成分越少）
    :return: 低通滤波后的张量
    """
    b, c, h, w = tensor.shape
    assert h==w
    cutoff = int(h*cutoff)
    # 进行 2D FFT 变换到频域
    fft_tensor = torch.fft.fft2(tensor, dim=(-2, -1))
    fft_tensor = torch.fft.fftshift(fft_tensor, dim=(-2, -1))  # 把 DC 成分移到中心
    
    # 构造低通滤波掩码
    mask = torch.zeros((h, w), device=tensor.device)
    center_h, center_w = h // 2, w // 2
    for i in range(h):
        for j in range(w):
            if (i - center_h) ** 2 + (j - center_w) ** 2 <= cutoff ** 2:
                mask[i, j] = 1  # 低频部分保留

    # 应用掩码
    mask = mask[None, None, :, :]  # 扩展维度以匹配 (b, c, h, w)
    fft_filtered = fft_tensor * mask

    # 逆变换回时域
    fft_filtered = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))  # 还原 DC 位置
    filtered_tensor = torch.fft.ifft2(fft_filtered, dim=(-2, -1)).real  # 只取实部

    return filtered_tensor

    
class RobustCWAttacker():
    def __init__(self, radius, steps, step_size, random_start, trans, sample_num, attacker, norm_type='l-infty', ascending=True, args=None, x_range=[0, 255], target_weight=1.0):        
        self.noattack = radius == 0. or steps == 0 or step_size == 0.
        self.radius = radius
        self.step_size = step_size
        self.steps = steps # how many step to conduct pgd
        self.random_start = random_start
        self.norm_type = norm_type # which norm of your noise
        self.ascending = ascending # perform gradient ascending, i.e, to maximum the loss
        self.args=args
        self.transform = trans
        self.sample_num = sample_num
        self.attacker = attacker
        self.left, self.right = x_range
        self.pattern = np.random.randint( 0, 72, size=(16, 16, 3), dtype=np.uint8)
        self.time_window_pos = random.uniform(float(args.time_window_start), float(args.time_window_end))
        self.time_window_len = float(args.time_window_len)
        print(
            "summary of the attacker: \n"
            f"radius: {radius}\n"
            f"steps: {self.steps}\n"
            f"step_size: {self.step_size}\n"
            f"random_start: {self.random_start}\n"
            f"norm_type: {self.norm_type}\n"
            f"ascending: {self.ascending}\n"
            f"sample_num: {self.sample_num}\n"
            # f"attacker: {self.attacker}\n", 
            f"x_range: {self.left} ~ {self.right}\n"
            f"loss_mode: {self.args.loss_mode}\n"
            f"low_f_filter:{self.args.low_f_filter}\n"
        )
        self.target_weight = target_weight
    
    # vkeilo add tokenizer
    def certi(self, models, adv_x,vae,  noise_scheduler, input_ids, device=torch.device("cuda"), weight_dtype=torch.float32, target_tensor=None,loss_type=None,ori_x=None,tokenizer=None):
        # args=self.args
        unet, text_encoder = models
        adv_latens = vae.encode(adv_x.to(device, dtype=weight_dtype)).latent_dist.sample()
        adv_latens = adv_latens * vae.config.scaling_factor
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(adv_latens)
        bsz = adv_latens.shape[0]
        # Sample a random timestep for each image
        timesteps = None
        max_timestep = int(noise_scheduler.config.num_train_timesteps)
        time_select_mode = 1
        if self.args.diff_time_diff_loss == '1' or self.args.diff_time_diff_loss == '2':
            if self.args.diff_time_diff_loss == '1':
                time_select_mode = random.randint(0,1)
            normal_time_range = torch.cat([
                torch.arange(0, int(max_timestep * self.time_window_pos)),
                torch.arange(int(max_timestep * (self.time_window_pos + self.time_window_len)), max_timestep)
            ], dim=0)
            attacked_time_range = torch.arange(int(max_timestep*self.time_window_pos), int(max_timestep*(self.time_window_pos+self.time_window_len)))
            if time_select_mode==0:
                # print("time no attack")
                timesteps = normal_time_range[torch.randint(0, normal_time_range.shape[0], (bsz,))].to(device=adv_latens.device)
            else:
                # print("time attack")
                timesteps = attacked_time_range[torch.randint(0, attacked_time_range.shape[0], (bsz,))].to(device=adv_latens.device)
            timesteps = timesteps.long()
        else:
            timesteps = torch.randint(0, int(max_timestep * self.args.time_select), (bsz,), device=adv_latens.device)
            timesteps = timesteps.long()
        # timesteps_classv = torch.randint(0, int(noise_scheduler.config.num_train_timesteps * self.args.time_select * 0.1), (bsz,), device=adv_latens.device)
        # timesteps_classv = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(adv_latens, noise, timesteps)
        # vkeilo add it
        # noise_antar = torch.randn_like(adv_latens)
        # if self.args.loss_mode == "classv":
        #     noisy_latents_antar = noise_scheduler.add_noise(self.args.another_target_img_latents, noise_antar, timesteps)
        # args=self.args
        # vkeilo add it
        from copy import deepcopy
        text_encoder_noised = deepcopy(text_encoder)
        encoder_hidden_states = None
        noised_unet = deepcopy(unet)
        encoder_hidden_states = text_encoder(input_ids.to(device))[0]

        if "robust_instance_conditioning_vector" in vars(self.args).keys() and self.args.robust_instance_conditioning_vector:
            condition_vector = self.args.robust_instance_conditioning_vector_data
            # print('this is your condition vector')
            # print(condition_vector.shape)
            encoder_hidden_states[0,:7,:] = condition_vector.to(device, dtype=weight_dtype)

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
            # vkeilo change it 
            # print("epsilon")
            if self.args.loss_mode == "classv":
                input_ids_antar = tokenizer(
                    self.args.classv_prompt,
                    # "New York Architecture Complex",
                    # "Van Gogh's Starry Sky Paintings",
                    # "a post-impressionist car",
                    # "Surrealist landscape with melting clocks",
                    # "A person without face",
                    # "A person whose eyes are replaced by burning galaxies",
                    # "a ph oto of s ksper son",
                    truncation=True,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.repeat(len(adv_x), 1)
                encoder_hidden_states_antar = text_encoder(input_ids_antar.to(device))[0]
                model_pred_antar = unet(noisy_latents, timesteps, encoder_hidden_states_antar).sample
                # sp_num = 2
                # model_pred.div_(sp_num)
                # model_pred.add_(-unet(noisy_latents, timesteps, encoder_hidden_states_antar).sample.div_(sp_num))
                # for _ in range(sp_num-1):
                #     model_pred.add_(unet(noisy_latents, timesteps, encoder_hidden_states).sample.div_(sp_num))
                #     model_pred.add_(-unet(noisy_latents, timesteps, encoder_hidden_states_antar).sample.div_(sp_num))
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(adv_latens, noise, timesteps)

        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


        loss = 0.0
        # vkeilo change it
        mse_loss = None
        if self.args.loss_mode == "mse":
            mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        elif self.args.loss_mode == "classv":
            # loss_fn = FieldLoss()
            # mse_loss = loss_fn(self.args.class2target_v_a.to(device, dtype=weight_dtype),model_pred.flatten().to(device, dtype=weight_dtype), target.flatten().to(device, dtype=weight_dtype))
            # mse_loss = -F.mse_loss(model_pred,-model_pred)
            mse_loss = -F.mse_loss(target.float(),model_pred_antar.float(), reduction="mean")
            # mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # mse_loss = -F.kl_div(model_pred_antar_list, model_pred_list, reduction='batchmean')
            # mse_loss = -F.mse_loss(model_pred.float(), torch.randn_like(adv_latens).float(), reduction="mean")
        elif self.args.loss_mode == "-noise":
            mse_loss = -F.mse_loss(model_pred.float(), -target.float(), reduction="mean")
        elif self.args.loss_mode == "dot0":
            pred_flat = model_pred.float().flatten()
            target_flat = target.float().flatten()
            mse_loss = -(torch.dot(pred_flat/torch.norm(pred_flat), target_flat/torch.norm(target_flat))**2)
        elif self.args.loss_mode == "randnoise":
            mse_loss = -F.mse_loss(model_pred.float(), torch.randn_like(adv_latens).float(), reduction="mean")
        else:
            exit(f'mse_loss:{mse_loss} not support')
        target_loss=0.0
        if target_tensor is not None:
            timesteps = timesteps.to('cpu')
            noisy_latents = noisy_latents.to('cpu')
            xtm1_pred = torch.cat(
                [
                    noise_scheduler.step(
                        model_pred[idx : idx + 1].to('cpu'),
                        timesteps[idx : idx + 1].to('cpu'),
                        noisy_latents[idx : idx + 1].to('cpu'),
                    ).prev_sample
                    for idx in range(len(model_pred))
                ]
            )
            xtm1_target = noise_scheduler.add_noise(target_tensor, noise, (timesteps - 1))
            target_loss = self.target_weight*F.mse_loss(xtm1_pred.to(device), xtm1_target.to(device))
        
        loss = mse_loss - target_loss
    
        return loss

    def perturb_over_branchs_of_model(self, list_of_model, x, ori_x, vae, tokenizer, noise_scheduler, target_tensor=None,):
        args=self.args
        if self.noattack:
            return ori_x
        # for each PGD step, we load back a list of model and craft perturbation using gradients from this esemble of models
        device = torch.device("cuda")
        weight_dtype = torch.bfloat16
        if args.mixed_precision == "fp32":
            weight_dtype = torch.float32
        elif args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        vae.to(device, dtype=weight_dtype)
    
        ori_x = ori_x.detach().clone().to(device, dtype=weight_dtype)
        x = x.detach().clone().to(device, dtype=weight_dtype)
        
        
        if self.random_start:
            r=self.radius
            r_noise = torch.zeros_like(x).uniform_(-r, r)
            r_x=x+r_noise
            x=self._clip_(r_x, x)

        input_ids = tokenizer(
            args.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(len(x), 1)
        
        x.requires_grad_(True)
        
        
        for _step in range(self.steps):
            gradd = 0.0
            print('pgd step: ', _step)
            for model_i in list_of_model:
                print('index', list_of_model.index(model_i))
                x.requires_grad = True 
                def_x_trans = self.transform(x).to(device, dtype=weight_dtype)
                unet, text_encoder = model_i
                two_model = [unet, text_encoder]
                text_encoder.to(device, dtype=weight_dtype)
                unet.to(device, dtype=weight_dtype)
                text_encoder.eval()
                unet.eval()
                for mi in [text_encoder, unet, vae]:
                    for pp in mi.parameters():
                        pp.requires_grad = False
                adv_x = self.attacker.perturb(
                    two_model, def_x_trans, self.transform(ori_x), vae, tokenizer, noise_scheduler, 
                )
                loss = self.certi(two_model, adv_x, vae, noise_scheduler, input_ids, device, weight_dtype, target_tensor)
                loss.backward()
                gradd += x.grad.data
                # put the model to cpu first 
                text_encoder.to('cpu')
                unet.to('cpu')
                del text_encoder; del unet
                del two_model
                del model_i
                import gc 
                gc.collect()
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                # grad = x.grad.data
                grad = gradd
                if not self.ascending:
                    grad.mul_(-1)
                if self.norm_type == 'l-infty':
                    x.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    raise NotImplementedError
                x = self._clip_(x, ori_x, ).detach_()
        return x.cpu()
        
    def perturb(self, models, x, ori_x, vae, tokenizer, noise_scheduler, target_tensor=None, adaptive_target=False, loss_type=None, device = torch.device("cuda")):
        args=self.args
        unet, text_encoder = models
        weight_dtype = torch.bfloat16
        if args.mixed_precision == "fp32":
            weight_dtype = torch.float32
        elif args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        device = torch.device("cuda")
        vae.to(device, dtype=weight_dtype)
        text_encoder.to(device, dtype=weight_dtype)
        unet.to(device, dtype=weight_dtype)
        
        ori_x = ori_x.detach().clone().to(device, dtype=weight_dtype)
        x = x.detach().clone().to(device, dtype=weight_dtype)

        # vkeilo add for cw
        def artanh(x):
            return 0.5 * torch.log((1 + x) / (1 - x))
        ori_adv_part = (x - ori_x)/self.radius
        ori_adv_part = torch.clamp(ori_adv_part, -0.999, 0.999)
        adv_part_map = artanh((x - ori_x)/self.radius)
        # adv_part  = self.radius * torch.tanh(adv_part_map)
        
        if self.noattack:
            print("defender no need to defend")
            return x
        
        # ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        # vae.eval()
        text_encoder.eval()
        unet.eval()
        for mi in [text_encoder, unet]:
           for pp in mi.parameters():
               pp.requires_grad = False
               
        # 初始化随机扰动
        if self.random_start:
            r=self.radius
            r_noise = torch.zeros_like(x).uniform_(-r, r)
            r_x=x+r_noise
            x=self._clip_(r_x, x)
            
        input_ids = tokenizer(
            args.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(len(x), 1)
        
        # x.requires_grad_(True)
        adv_part_map.requires_grad_(True)
        # ori_adv_part.requires_grad_(True)
        # adv_part.requires_grad_(True)
        # 多次采样
        loss_list = []
        print(f'defender start {self.steps} steps perturb')
        for _step in range(self.steps):
            print(f'\tdefender {_step}/{self.steps} step perturb')
            # x.requires_grad = True
            adv_part_map.requires_grad_(True)
            # ori_adv_part.requires_grad_(True)
            # adv_part.requires_grad_(True)
            print(f'\tdefender start {self.sample_num} samples perturb')
            # 默认一次更新就对抗扰动一次
            for _sample in range(self.sample_num):
                print(f'\t\tdefender{_sample}/{self.sample_num} sample perturb')
                # 模拟微调训练方对图像进行一定变换以缓解毒性
                # print(f'before trans x is:{x} ')
                adv_part  = self.radius * torch.tanh(adv_part_map)
                def_x_trans = self.transform(ori_x+adv_part).to(device, dtype=weight_dtype)
                # 获得被进一步对抗扰动后的样本adv_x
                adv_x = def_x_trans
                # 在额外扰动的adv_x上评估模型鲁棒性
                loss = self.certi(models, adv_x, vae, noise_scheduler, input_ids, device, weight_dtype, target_tensor,loss_type, ori_x,tokenizer = tokenizer)
                loss_list.append(loss.item())
                # 多次采样积累能够降低模型鲁棒性的梯度
                loss.backward()
            # 根据当前梯度信息更新x
            with torch.no_grad():
                # vkeilo change it for cw
                grad = adv_part_map.grad.data
                # print(f"grad is :{grad}")
                # print(torch.sign(grad))
                if not self.ascending: 
                    grad.mul_(-1)
                    
                if self.norm_type == 'l-infty':
                    # adv_part_map.add_(grad)
                    adv_part_map.add_(grad, alpha=self.step_size)
                else:
                    raise NotImplementedError
                # x = self._clip_(x, ori_x, ).detach_()
                adv_part_map = adv_part_map.detach_()
            # vkeilo add it for CW  480step ~ x13430
            # self.step_size = self.step_size
        # final_adv_part = self.radius * torch.tanh(adv_part_map)

        x = ori_x + self.radius * torch.tanh(adv_part_map)
        x.clamp_(self.left, self.right)
        ''' reopen autograd of model after pgd '''
        for mi in [text_encoder, unet]:
            for pp in mi.parameters():
                pp.requires_grad = True
        mean_loss = np.mean(loss_list)
        return x.cpu(),mean_loss
    def _clip_(self, adv_x, x, ):
        adv_x = adv_x - x
        if self.norm_type == 'l-infty':
            adv_x.clamp_(-self.radius, self.radius)
        else:
            raise NotImplementedError
        adv_x = adv_x + x
        adv_x.clamp_(self.left, self.right)
        return adv_x
    