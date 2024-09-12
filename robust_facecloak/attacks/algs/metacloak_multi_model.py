# -----------------------------------------------------------------------
# Copyright (c) 2023 Yixin Liu Lehigh University
# All rights reserved.
#
# This file is part of the MetaCloak project. Please cite our paper if our codebase contribute to your project. 
# -----------------------------------------------------------------------

import random
import wandb
import argparse
import copy
import hashlib
import itertools
import logging
import os
from pathlib import Path
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from robust_facecloak.model.db_train import  DreamBoothDatasetFromTensor
from robust_facecloak.model.db_train import import_model_class_from_model_name_or_path
from robust_facecloak.generic.data_utils import PromptDataset, load_data
from robust_facecloak.generic.share_args import share_parse_args


logger = get_logger(__name__)

import torch
import torch.nn.functional as F

# 针对模型的unet和文本编码器进行的训练
def train_few_step(
    args,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    data_tensor: torch.Tensor,
    num_steps=20,
    step_wise_save=False,
    save_step=100, 
    retain_graph=False,
    task_loss_name = None,
):
    # Load the tokenizer

    unet, text_encoder = copy.deepcopy(models[0]), copy.deepcopy(models[1])
    # 绑定unet和文本编码器的参数，共同优化
    params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())

    # 设置优化器，优化目标为unet参数和文本编码器参数
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    train_dataset = DreamBoothDatasetFromTensor(
        data_tensor,
        # A photo of sks person
        args.instance_prompt,
        tokenizer,
        args.class_data_dir,
        args.class_prompt,
        args.resolution,
        args.center_crop,
    )

    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    # 将关键模型移动到对应设备
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    
    step2modelstate={}
        
    pbar = tqdm(total=num_steps, desc="training")
    for step in range(num_steps):
        # 根据设置选择是否保存训练中间过程参数
        if step_wise_save and ((step+1) % save_step == 0 or step == 0):
            # make sure the model state dict is put to cpu
            step2modelstate[step] = {
                "unet": copy.deepcopy(unet.cpu().state_dict()),
                "text_encoder": copy.deepcopy(text_encoder.cpu().state_dict()),
            }
            # move the model back to gpu
            unet.to(device, dtype=weight_dtype); text_encoder.to(device, dtype=weight_dtype)
            
        pbar.update(1)
        # 训练模式
        unet.train()
        text_encoder.train()
        # 循环从训练数据集中取一个样本
        step_data = train_dataset[step % len(train_dataset)]
        # 将样本中的类别图片和实例图片整合并移动到设备上
        pixel_values = torch.stack([step_data["instance_images"], step_data["class_images"]]).to(
            device, dtype=weight_dtype
        )
        # 将样本中的类别提示词和实例提示词整合并移动到设备上
        input_ids = torch.cat([step_data["instance_prompt_ids"], step_data["class_prompt_ids"]], dim=0).to(device)
        # 使用VAE对图像进行编码，并对潜在表示进行后处理
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        # 向图片编码向量（潜在空间向量表示）添加随机噪声
        noise = torch.randn_like(latents)
        # batch_size
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        # 为每个图片生成一个随机step
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        # 前向过程，得到前向扩散特定时间步后的图片的潜在空间向量
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        # 文本编码向量作为条件信息
        encoder_hidden_states = text_encoder(input_ids)[0]
        
        # Predict the noise residual
        # 模型基于当前的噪声潜在表示（noisy_latents）、时间步（timesteps）和文本条件（encoder_hidden_states），预测噪声残差
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        # 预测的可以是噪声，也可以是变化速度
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # with prior preservation loss
        # 可选是否使用先验保留损失
        if args.with_prior_preservation:
            # 再次分为一半一半，对应之前的stack操作
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Compute prior loss  确保在原来类别上的生成能力不丢失
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = instance_loss + args.prior_loss_weight * prior_loss

        else:
            # 不使用先验保留损失
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        if task_loss_name is not None:
            wandb.log({f"{task_loss_name}": loss.item()})
        # 反向传播
        loss.backward(retain_graph=retain_graph)
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0, error_if_nonfinite=True)
        # 参数优化
        optimizer.step()
        optimizer.zero_grad()

    pbar.close()
    # 返回训练的参数数据
    if step_wise_save:
        return [unet, text_encoder], step2modelstate
    else:     
        return [unet, text_encoder]

# 主要模型的加载
def load_model(args, model_path):
    print(model_path)
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(model_path, args.revision)

    # Load scheduler and models
    # 文本编码器加载
    text_encoder = text_encoder_cls.from_pretrained(
        model_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    # unet加载
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=args.revision)
    # tokenizer加载
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    # 使用DDPM同款调度器
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    # 加载预训练的vae，vae不需要更新参数
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=args.revision)

    vae.requires_grad_(False)

    # 甚至可以不更新文本编码器的参数
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        print("You selected to used efficient xformers")
        print("Make sure to install the following packages before continue")
        print("pip install triton==2.0.0.dev20221031")
        print("pip install pip install xformers==0.0.17.dev461")

        unet.enable_xformers_memory_efficient_attention()
    # 返回5个关键模型
    return text_encoder, unet, tokenizer, noise_scheduler, vae


# 解析参数
def parse_args(): 
    
    parser = share_parse_args()
    # 是否在数据增强时使用水平翻转
    parser.add_argument(
        "--transform_hflip",
        action="store_true",
        help="Whether to use horizontal flip for transform.",
    )
    
    parser.add_argument(
        "--instance_data_dir_for_train",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )

    parser.add_argument(
        "--instance_data_dir_for_adversarial",
        type=str,
        default=None,
        required=True,
        help="A folder containing the images to add adversarial noise",
    )
    # 控制生成对抗样本时增加扰动的程度扰动是否递增
    parser.add_argument(
        "--defense_pgd_ascending",
        action="store_true",
        help="Whether to use ascending order for pgd.",
    )
    
    # 防御 PGD 的半径（扰动的最大程度）
    parser.add_argument(
        "--defense_pgd_radius",
        type=float,
        default=8,
        help="The radius for defense pgd.",
    )
    # 防御 PGD 的步长（每一步的扰动大小）
    parser.add_argument(
        "--defense_pgd_step_size",
        type=float,
        default=2,
        help="The step size for defense pgd.",
    )
    # 防御 PGD 的步数
    parser.add_argument(
        "--defense_pgd_step_num",
        type=int,
        default=8,
        help="The number of steps for defense pgd.",
    )
    # 是否在防御 PGD 中使用随机开始点
    parser.add_argument(
        "--defense_pgd_random_start",
        action="store_true",
        help="Whether to use random start for pgd.",
    )
    # 攻击 PGD 的半径（扰动的最大程度）
    parser.add_argument(
        "--attack_pgd_radius",
        type=float,
        default=4,
        help="The radius for attack pgd.",
    )
    # 攻击 PGD 的步长（每一步的扰动大小）
    parser.add_argument(
        "--attack_pgd_step_size",
        type=int,
        default=2,
        help="The step size for attack pgd.",
    )
    # 攻击 PGD 的步数
    parser.add_argument(
        "--attack_pgd_step_num",
        type=int,
        default=4,
        help="The number of steps for attack pgd.",
    )
    # 是否在攻击 PGD 中使用递增顺序
    parser.add_argument(
        "--attack_pgd_ascending",
        action="store_true",
        help="Whether to use ascending order for pgd.",
    )
    # 是否在攻击 PGD 中使用随机开始点
    parser.add_argument(
        "--attack_pgd_random_start",
        action="store_true",
        help="Whether to use random start for pgd.",
    )
    # 目标图像路径
    parser.add_argument(
        "--target_image_path",
        default=None,
        help="target image for attacking",
    )
    
    # 高斯滤波器的核大小（在数据增强中使用）
    parser.add_argument(
        "--gau_kernel_size",
        type=int,
        default=5,
        help="The kernel size for gaussian filter.",
    )
    # 用于防御的样本数量
    parser.add_argument(
        "--defense_sample_num",
        type=int,
        default=1,
        help="The number of samples for defense.",
    )
    # 旋转的角度
    parser.add_argument(
        "--rot_degree",
        type=int,
        default=5,
        help="The degree for rotation.",
    )
    # 是否在数据增强中使用旋转
    parser.add_argument(
        "--transform_rot", 
        action="store_true",
        help="Whether to use rotation for transform.",
        
    )
    # 是否在数据增强中使用高斯滤波
    parser.add_argument(
        "--transform_gau",
        action="store_true",
        help="Whether to use gaussian filter for transform.",
    )
    # 是否在对抗样本准备和标注中使用默认的原始流程
    parser.add_argument(
        "--original_flow", 
        action="store_true",
        help="Whether to use original flow in ASPL for transform.",
    )
    # 总实验次数
    parser.add_argument(
        "--total_trail_num",
        type=int,
        default=60,
    )
    # 记录间隔，展开数（提前训练步）——实验设置为1，其实就是没有使用
    parser.add_argument(
        "--unroll_steps",
        type=int,
        default=2,
    )
    # 训练总步长
    parser.add_argument(
        "--interval",
        type=int,
        default=40,
    )
    
    parser.add_argument(
        "--total_train_steps",
        type=int,
        default=1000, 
    )

    # vkeilo add it 
    # 采样次数
    parser.add_argument(
        "--sampling_times_theta",
        type=int,
        default=10,
    )

    # vkeilo add it
    # 滑动平均参数
    parser.add_argument(
        "--beta_s",
        type=float,
        default=0.3,
    )

    
    args = parser.parse_args()
    return args

# 核心处理流程
def main(args):
    # 指定日志目录
    logging_dir = Path(args.output_dir, args.logging_dir)
    # Hugging Face加速器，指定混合精度训练模式和记录方式，默认为wandb
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # logging_dir=logging_dir,
    )

    # 初始化日志记录器，指定格式和日志级别
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # 记录加速器信息
    logger.info(accelerator.state, main_process_only=False)
    # 只在主进程上尽可能详细地记录日志
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    # 如果启用了先验保留，则先生成类图像，保存在变量cur_class_images中
    if args.with_prior_preservation:
        # 检查并创建一个用于存储类别图像的目录，并统计当前目录中已经存在的类别图像数量
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))
        # 如果当前类别图像数量小于所需数量，则生成新的类别图像
        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.mixed_precision == "fp32":
                torch_dtype = torch.float32
            elif args.mixed_precision == "fp16":
                torch_dtype = torch.float16
            elif args.mixed_precision == "bf16":
                torch_dtype = torch.bfloat16
            # 此处的pipline是用来根据类别提示生成类别图像的模型
            pipeline = DiffusionPipeline.from_pretrained(
                list(args.pretrained_model_name_or_path.split(","))[-1], 
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            # 生成类别图像时，不显示进度条
            pipeline.set_progress_bar_config(disable=True)
            # 计算还需要生成的类别图像数量，num_class_images默认为200
            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")
            # 类别生成的提示词数据（其实都是 a photo of a person，因为在隐私保护中，扩散模型的的先验能力就是在人像类别上的生成能力）
            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            # sample_batch_size默认为4
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)
            # 将数据集使用accelerator进行预处理（就是设置为使用混合精度和wandb记录）
            sample_dataloader = accelerator.prepare(sample_dataloader)
            # 将模型移动到accelerator.device上（accelerator） 会自动选择
            pipeline.to(accelerator.device)

            # 每批4个提示词，生成4个类别图像（一样的提示词）
            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    # 使用哈希值为图像文件名，并保存
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)
            # 删除模型，释放显存
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 判断是否启用tf32加速
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 加载干净数据
    clean_data = load_data(
        args.instance_data_dir_for_train,
        # size=args.resolution,
        # center_crop=args.center_crop,
    )
    
    # 加载原始扰动数据
    perturbed_data = load_data(
        args.instance_data_dir_for_adversarial,
        # size=args.resolution,
        # center_crop=args.center_crop,
    )
    
    # original_data当前为原始扰动数据
    original_data= copy.deepcopy(perturbed_data)
        
    import torchvision
    # 定义训练和测试时的数据增强（图像处理）
    train_aug = [
            # 双线性插值到512x512
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # 裁剪
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    ]
    # 图像随机旋转
    rotater = torchvision.transforms.RandomRotation(degrees=(0, args.rot_degree))
    # 高斯模糊
    gau_filter = transforms.GaussianBlur(kernel_size=args.gau_kernel_size,)
    # 制定防御对抗样本变换策略
    defense_transform = [
    ]
    if args.transform_hflip:
        defense_transform = defense_transform + [transforms.RandomHorizontalFlip(p=0.5)]
    if args.transform_rot:
        defense_transform = defense_transform + [rotater]
    if args.transform_gau:
        defense_transform = [gau_filter] + defense_transform
    
    # 标准化均值和标准差
    tensorize_and_normalize = [
        transforms.Normalize([0.5*255]*3,[0.5*255]*3),
    ]
    
    # 将所有变换组合起来，整合为一个Compose对象
    all_trans = train_aug + defense_transform + tensorize_and_normalize
    all_trans = transforms.Compose(all_trans)
    
    
    from robust_facecloak.attacks.worker.robust_pgd_worker import RobustPGDAttacker
    from MetaCloak.robust_facecloak.attacks.worker.pgd_worker import PGDAttacker
    # 构建攻击者和防御者，攻击者使用PGD算法
    attacker = PGDAttacker(
        radius=args.attack_pgd_radius, 
        steps=args.attack_pgd_step_num, 
        step_size=args.attack_pgd_step_size,
        random_start=args.attack_pgd_random_start,
        ascending=args.attack_pgd_ascending,
        args=args, 
        x_range=[-1, 1],
    )
    defender = RobustPGDAttacker(
        radius=args.defense_pgd_radius,
        steps=args.defense_pgd_step_num,
        step_size=args.defense_pgd_step_size,
        random_start=args.defense_pgd_random_start,
        ascending=args.defense_pgd_ascending,
        args=args,
        attacker=attacker, 
        trans=all_trans,
        sample_num=args.defense_sample_num,
        x_range=[0, 255],
    )

    # 模型加载，本次实验只有一个
    model_paths = list(args.pretrained_model_name_or_path.split(","))
    num_models = len(model_paths)

    MODEL_BANKS = [load_model(args, path) for path in model_paths]

    # 提取模型的文本编码器和UNet的状态字典
    MODEL_STATEDICTS = [
        {
            "text_encoder": MODEL_BANKS[i][0].state_dict(),
            "unet": MODEL_BANKS[i][1].state_dict(),
        }
        for i in range(num_models)
    ]
    # 此函数将保存经过扰动处理的图像数据到noise-ckpt/{id_stamp}目录中，id_stamp在此次实验中为迭代次数
    def save_image(perturbed_data, id_stamp):
        save_folder = f"{args.output_dir}/noise-ckpt/{id_stamp}"
        os.makedirs(save_folder, exist_ok=True)
        noised_imgs = perturbed_data.detach()
        img_names = [
            str(instance_path).split("/")[-1]
            for instance_path in list(Path(args.instance_data_dir_for_adversarial).iterdir())
        ]
        for img_pixel, img_name in zip(noised_imgs, img_names):
            save_path = os.path.join(save_folder, f"noisy_{img_name}")
            Image.fromarray(
                img_pixel.float().detach().cpu().permute(1, 2, 0).numpy().squeeze().astype(np.uint8)
            ).save(save_path)

    
    init_model_state_pool = {}
    pbar = tqdm(total=num_models, desc="initializing models")
    # split sub-models
    # 对于每一个模型，都进行一次训练
    for j in range(num_models):
        init_model_state_pool[j] = {}
        # 提取关键模块
        text_encoder, unet, tokenizer, noise_scheduler, vae = MODEL_BANKS[j]
        
        # 加载unet和text_encoder的模型参数
        unet.load_state_dict(MODEL_STATEDICTS[j]["unet"])
        text_encoder.load_state_dict(MODEL_STATEDICTS[j]["text_encoder"])
        # 打包unet和text_encoder
        f_ori = [unet, text_encoder]
        # 得到训练total_train_steps步之后的unet, text_encoder参数以及中间状态参数
        f_ori, step2state_dict = train_few_step(
                args,
                f_ori,
                tokenizer,
                noise_scheduler,
                vae,
                perturbed_data.float(),
                args.total_train_steps,
                step_wise_save=True,
                save_step=args.interval,
                task_loss_name="ori_model_train_loss",
        )  
        # init_model_state_pool就来保存训练中间状态参数
        init_model_state_pool[j] = step2state_dict

        # 释放占用的资源
        del f_ori, unet, text_encoder, tokenizer, noise_scheduler, vae
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        pbar.update(1)
    pbar.close()
    
    # 提取保存的中间状态的step数据（1000，2000，3000.....）        
    steps_list = list(init_model_state_pool[0].keys())
    # 进度条，总train_few_step调用的次数
    pbar = tqdm(total=args.total_trail_num * num_models * (args.interval // args.advance_steps) * len(steps_list), desc="meta poison with model ensemble")
    cnt=0
    # learning perturbation over the ensemble of models
    # 在多个模型集合上进行扰动优化
    # 多次实验
    for _ in range(args.total_trail_num):
        # 针对每一个模型
        for model_i in range(num_models):
            # 确定关键组件
            text_encoder, unet, tokenizer, noise_scheduler, vae = MODEL_BANKS[model_i]
            # 对于每一个中间状态step
            for split_step in steps_list: 
                # 加载unet和文本编码器的中间状态参数
                unet.load_state_dict(init_model_state_pool[model_i][split_step]["unet"])
                text_encoder.load_state_dict(init_model_state_pool[model_i][split_step]["text_encoder"])
                f = [unet, text_encoder]
                # 每advance_steps步进行一次防御优化
                for j in range(args.interval // args.advance_steps):
                    perturbed_data,rubust_loss = defender.perturb(f, perturbed_data, original_data, vae, tokenizer, noise_scheduler,)
                    wandb.log({"defender_rubust_loss_without_MAT": rubust_loss})
                    cnt+=1
                    f = train_few_step(
                        args,
                        f,
                        tokenizer,
                        noise_scheduler,
                        vae,
                        perturbed_data.float(),
                        args.advance_steps,
                        task_loss_name = "model_theta_loss_without_MAT"
                    )
                    pbar.update(1)
                    # 每1000次扰动优化，保存一次扰动示例图像
                    if cnt % 1000 == 0:
                        save_image(perturbed_data, f"{cnt}")
                
                # frequently release the memory due to limited GPU memory, 
                # env with more gpu might consider to remove the following lines for boosting speed
                # 释放资源
                del f 
                torch.cuda.empty_cache()
                
            del unet, text_encoder, tokenizer, noise_scheduler, vae

            if torch.cuda.is_available():
                torch.cuda.empty_cache() 

        import gc
        gc.collect()
        torch.cuda.empty_cache()      
    pbar.close()
    # 保存最后的结果
    save_image(perturbed_data, "final")


if __name__ == "__main__":
    # 获取脚本传参
    args = parse_args()
    wandb.init(project="metacloak", entity=args.wandb_entity_name)
    wandb.config.update(args)
    wandb.log({'status': 'gen'})
    # 核心代码
    main(args)
