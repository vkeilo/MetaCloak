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
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from robust_facecloak.model.db_train import  DreamBoothDatasetFromTensor
from robust_facecloak.model.db_train import import_model_class_from_model_name_or_path
from robust_facecloak.generic.data_utils import PromptDataset, load_data
from robust_facecloak.generic.share_args import share_parse_args
# vkeilo add it
import utils
import GPUtil
import time
import pickle
from robust_facecloak.attacks.worker.differential_color_functions import rgb2lab_diff, ciede2000_diff
import torchvision.transforms as transforms
from robust_facecloak.attacks.worker.get_v import get_identify_feature_latents


logger = get_logger(__name__)

# vkeilo add it
def get_class2target_v_a(args,vae,trans,device = torch.device("cuda")):
    weight_dtype = torch.float32
    target_imgs = load_data(args.instance_data_dir_for_adversarial)
    class_imgs = load_data(args.class_data_dir)
    target_imgs_trans = trans(target_imgs).to(device, dtype=weight_dtype)
    class_imgs_trans = trans(class_imgs).to(device, dtype=weight_dtype)
    batch_size = 20
    vae = vae.to(device, dtype = weight_dtype)
    # print(f"vae:{vae.device},data:{target_imgs_trans.device}")
    target_imgs_latens = vae.encode(target_imgs_trans).latent_dist.sample()
    # class_imgs_latens = vae.encode(class_imgs_trans).latent_dist.sample()
    class_imgs_latens = torch.tensor([]).to(device, dtype=weight_dtype)
    for i in range(0, len(class_imgs), batch_size):
        tmp_class_imgs_latens = vae.encode(class_imgs_trans[i:i+batch_size]).latent_dist.sample()
        class_imgs_latens = torch.cat((class_imgs_latens, tmp_class_imgs_latens), dim=0)
    random_imgs_latens = torch.randn([10000,4,64,64]).to(device)
    mean_target_imgs_latens = torch.mean(target_imgs_latens, dim=0, keepdim=True).squeeze()
    mean_class_imgs_latens = torch.mean(class_imgs_latens, dim=0, keepdim=True).squeeze()
    mean_random_imgs_latens = torch.mean(random_imgs_latens, dim=0, keepdim=True).squeeze()
    del random_imgs_latens

    def get_orthog(x,y):
        y_a = y / torch.norm(y)
        x2y_pjt = (torch.dot(y,x)/torch.norm(y))*y_a
        x2y_orthor = x - x2y_pjt
        return x2y_orthor

    rand2class = mean_class_imgs_latens - mean_random_imgs_latens
    rand2target = mean_target_imgs_latens - mean_random_imgs_latens
    class2target = mean_target_imgs_latens - mean_class_imgs_latens
    # rand2class = torch.tensor([2,2]).to(dtype=args.weight_dtype)
    # rand2target = torch.tensor([1,2]).to(dtype=args.weight_dtype)
    # class2target = torch.tensor([-1,0]).to(dtype=args.weight_dtype)


    rand2class_flat = rand2class.flatten()
    rand2target_flat = rand2target.flatten()
    class2target_flat = class2target.flatten()
    mean_random_imgs_latens_flat = mean_random_imgs_latens.flatten()

    rand2target_to_rand2class_orthog = get_orthog(rand2target_flat,rand2class_flat)
    rand2target_to_rand2class_orthog_a = rand2target_to_rand2class_orthog/torch.norm(rand2target_to_rand2class_orthog)
    # vkeilo change it 
    # rand2target_to_rand2class_orthog_a = torch.randn_like(rand2target_to_rand2class_orthog_a)
    # rand2target_to_rand2class_orthog_a = rand2target_to_rand2class_orthog_a/torch.norm(rand2target_to_rand2class_orthog_a)
    for idx, feature in rand2target_to_rand2class_orthog_a:
        if torch.dot(feature,class2target) < 0:
            rand2target_to_rand2class_orthog_a[idx] = -rand2target_to_rand2class_orthog_a[idx]
    return rand2target_to_rand2class_orthog_a


def get_mixed_features_v_a(args,vae,trans,device = torch.device("cuda")):
    weight_dtype = torch.float32
    target_imgs = load_data(args.instance_data_dir_for_adversarial)
    class_imgs = load_data(args.class_data_dir)
    target_imgs_trans = trans(target_imgs).to(device, dtype=weight_dtype)
    class_imgs_trans = trans(class_imgs).to(device, dtype=weight_dtype)
    batch_size = 20
    vae = vae.to(device, dtype = weight_dtype)
    # print(f"vae:{vae.device},data:{target_imgs_trans.device}")
    target_imgs_latens = vae.encode(target_imgs_trans).latent_dist.sample()
    # class_imgs_latens = vae.encode(class_imgs_trans).latent_dist.sample()
    class_imgs_latens = torch.tensor([]).to(device, dtype=weight_dtype)
    for i in range(0, len(class_imgs), batch_size):
        tmp_class_imgs_latens = vae.encode(class_imgs_trans[i:i+batch_size]).latent_dist.sample()
        class_imgs_latens = torch.cat((class_imgs_latens, tmp_class_imgs_latens), dim=0)
    random_imgs_latens = torch.randn([10000,4,64,64]).to(device)
    mean_target_imgs_latens = torch.mean(target_imgs_latens, dim=0, keepdim=True).squeeze()
    mean_class_imgs_latens = torch.mean(class_imgs_latens, dim=0, keepdim=True).squeeze()
    mean_random_imgs_latens = torch.mean(random_imgs_latens, dim=0, keepdim=True).squeeze()
    del random_imgs_latens

    def get_some_principal_component(X,num):
        """
        输入:
            X : Tensor of shape (200, 16000)
            200个样本，每个样本16000维特征
        
        输出:
            pc1 : Tensor of shape (1, 16000)
            第一个主成分向量
        """
        # 数据标准化：中心化
        X_centered = X - X.mean(dim=0, keepdim=True)  # 按列求均值
        
        # 奇异值分解 (SVD)
        _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        
        # 第一主成分是右奇异矩阵的第一行
        pc1 = Vh[:num, :]  # 保持输出形状为(1, 16000)
        return pc1

    rand2class = mean_class_imgs_latens - mean_random_imgs_latens
    rand2target = mean_target_imgs_latens - mean_random_imgs_latens
    class2target = mean_target_imgs_latens - mean_class_imgs_latens
    # rand2class = torch.tensor([2,2]).to(dtype=args.weight_dtype)
    # rand2target = torch.tensor([1,2]).to(dtype=args.weight_dtype)
    # class2target = torch.tensor([-1,0]).to(dtype=args.weight_dtype)


    rand2class_flat = rand2class.flatten()
    rand2target_flat = rand2target.flatten()
    class2target_flat = class2target.flatten()
    mean_random_imgs_latens_flat = mean_random_imgs_latens.flatten()
    class_imgs_latens_flat =  class_imgs_latens.flatten(1)
    class_principal_component = get_some_principal_component(class_imgs_latens_flat,50)
    feature_possi_list = [0] * len(class_principal_component)
    feature_strength_list = [0] * len(class_principal_component)
    for index,feature in enumerate(class_principal_component):
        feature_strength = torch.dot(feature,class2target_flat)
        if feature_strength>0:
            feature_possi_list[index] += 1
            feature_strength_list[index] += feature_strength
        else:
            feature_possi_list[index] -= 1
            feature_strength_list[index] -= feature_strength

    class_principal_component_all_positive_flag = [1] * len(class_principal_component)
    class_principal_component_all_positive_idx = [i for i in range(len(class_principal_component))]

    feature_strength_list_idx_sorted = sorted(range(len(feature_strength_list)), key=lambda i: feature_strength_list[i], reverse=True)
    max_strength_num = 10
    start = 0
    final_feature_id = []
    for id in feature_strength_list_idx_sorted[start:max_strength_num]:
        if id in class_principal_component_all_positive_idx:
            final_feature_id.append(id)

    final_feature = []
    for id in final_feature_id:
        final_feature.append(class_principal_component[id]*class_principal_component_all_positive_flag[id])



    mixed_feature = torch.zeros_like(final_feature[0])
    for feature_id,feature_latents in zip(final_feature_id,final_feature):
        mixed_feature += feature_latents * feature_strength_list[feature_id]
    mixed_feature = mixed_feature/torch.norm(mixed_feature)

    return mixed_feature


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
    dpcopy = True,
    task_loss_name = None,
    loss_return = False,
):
    # Load the tokenizer
    # vkeilo remove deepcopy
    if dpcopy:
        unet, text_encoder = copy.deepcopy(models[0]), copy.deepcopy(models[1])
    else:
        unet, text_encoder = models[0], models[1]
    # unet, text_encoder = models[0], models[1]
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
        # print(f'latents shape: {latents.shape}')
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
        # print(f'noisy_latents shape: {noisy_latents.shape}')
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # print('model_pred shape', model_pred.shape)
        # Get the target for loss depending on the prediction type
        # 预测的可以是噪声，也可以是变化速度
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

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
        if not task_loss_name is None:
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
    if not loss_return:
        if step_wise_save:
            return [unet, text_encoder], step2modelstate
        else:     
            return [unet, text_encoder]
    else:
        if step_wise_save:
            return [unet, text_encoder], step2modelstate, loss.item()
        else:     
            return [unet, text_encoder], loss.item()

# 主要模型的加载
def load_model(args, model_path):
    print(f'out {model_path}')
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

    # vkeilo add it
    # text_encoder = text_encoder.bfloat16()
    # unet = unet.bfloat16()

    # tokenizer加载
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    # 使用DDPM同款调度器
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", prediction_type=args.prediction_type)
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
        type=float,
        default=1,
        help="The step size for attack pgd.",
    )
    # 攻击 PGD 的步数,就是一次扰动更新的采样次数
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
    # 记录间隔
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
    # 采样内部滑动平均参数
    parser.add_argument(
        "--beta_s",
        type=float,
        default=0.3,
    )

    # vkeilo add it
    # 参数更新的滑动平均参数
    parser.add_argument(
        "--beta_p",
        type=float,
        default=0.3,
    )

    # vkeilo add it
    # 模型参数采样（优化）次数
    parser.add_argument(
        "--sampling_times_theta",
        type=int,
        default=10,
    )

    # vkeilo add it
    # 扰动采样（优化）次数
    parser.add_argument(
        "--sampling_times_delta",
        type=int,
        default=10,
    )

    # vkeilo add it
    # 模型参数采样（优化）间隔
    parser.add_argument(
        "--sampling_step_theta",
        type=float,
        default=1e-5,
    )

    # vkeilo add it
    # 扰动采样（优化）步长,SGLD
    parser.add_argument(
        "--sampling_step_delta",
        type=float,
        default=1e-3,
    )

    # vkeilo add it
    # 采样时加入噪声的比例
    parser.add_argument(
        "--sampling_noise_ratio",
        type=float,
        default=0.05,
    )

    # vkeilo add it
    parser.add_argument(
        "--mat_lambda_s",
        type=float,
        default=3,
    )

    # vkeilo add it
    # 攻击模式选择
    parser.add_argument(
        "--attack_mode",
        type=str,
        default='pdg',
    )

    # vkeilo add it
    # PAN攻击的判别器lambda
    parser.add_argument(
        "--pan_lambda_D",
        type=float,
        default=0.0001,
    )

    # vkeilo add it
    # PAN攻击的求解器lambda
    parser.add_argument(
        "--pan_lambda_S",
        type=float,
        default=0.05,
    )
    
    # vkeilo add it
    # PAN攻击的omiga参数
    parser.add_argument(
        "--pan_omiga",
        type=float,
        default=0.5,
    )

    # vkeilo add it
    # PAN攻击的k值
    parser.add_argument(
        "--pan_k",
        type=int,
        default=2,
    )

    # vkeilo add it
    # PAN攻击解的来源（判别器或求解器）
    parser.add_argument(
        "--pan_mode",
        type=str,
        default='S',
    )

    # vkeilo add it
    # 预训练过的扩散模型路径（init_model_state_pool）
    parser.add_argument(
        "--init_model_state_pool_pth_path",
        type=str,
        default=None,
    )    
    
    # vkeilo add it
    # PAN攻击轮次使用的最终结果（loss最小值还是最后值）
    parser.add_argument(
        "--pan_use_val",
        type=str,
        default='last',
    )

    # vkeilo add it
    # PAN攻击轮次使用的最终结果（loss最小值还是最后值）
    parser.add_argument(
        "--model_select_mode",
        type=str,
        default='order',
    )

    # vkeilo add it
    # 如果采用动态模型选择，训练迭代的次数
    parser.add_argument(
        "--total_gan_step",
        type=int,
        default=0,
    )

    # vkeilo add it
    # save间隔
    parser.add_argument(
        "--img_save_interval",
        type=int,
        default=1000,
    )



    # # vkeilo add it
    # # wandb run name
    # parser.add_argument(
    #     "--wandb_run_name",
    #     type=str,
    #     default="test_run_name",
    # )

    # # vkeilo add it
    # # wandb run name
    # parser.add_argument(
    #     "--wandb_project_name",
    #     type=str,
    #     default="metacloak_test",
    # )

    # vkeilo add it
    # SGLD应用在哪里
    parser.add_argument(
        "--SGLD_method",
        type=str,
        default="allSGLD",
    )

    # vkeilo add it
    # 正则化方法
    parser.add_argument(
        "--Ltype",
        type=str,
        default="",
    )
    # vkeilo add it
    # rd
    parser.add_argument(
        "--radius_d",
        type=int,
        default=11,
    )
    # vkeilo add it
    # 
    parser.add_argument(
        "--max_L",
        type=float,
        default=None,
    )
    # vkeilo add it
    # 
    parser.add_argument(
        "--min_L",
        type=float,
        default=None,
    )
    # vkeilo add it
    # 
    parser.add_argument(
        "--hpara_update_interval",
        type=int,
        default=5,
    )
    # vkeilo add it
    # 
    parser.add_argument(
        "--dynamic_mode",
        type=str,
        default="",
    )
    # vkeilo add it
    # 
    parser.add_argument(
        "--omiga_strength",
        type=float,
        default="1",
    )
    # vkeilo add it
    parser.add_argument(
        "--time_select",
        type=float,
        default="1",
    )
    # vkeilo add it
    parser.add_argument(
        "--use_edge_filter",
        type=int,
        default=0,
    )
    
    # vkeilo add it
    parser.add_argument(
        "--use_unet_noise",
        type=int,
        default=0,
    )

    # vkeilo add it
    parser.add_argument(
        "--use_text_noise",
        type=int,
        default=0,
    )

    # vkeilo add it
    parser.add_argument(
        "--unet_noise_r",
        type=float,
        default=0,
    )

    # vkeilo add it
    parser.add_argument(
        "--text_noise_r",
        type=float,
        default=0,
    )

    # vkeilo add it
    parser.add_argument(
        "--loss_mode",
        type=str,
        default='mse',
    )

    # vkeilo add it
    parser.add_argument(
        "--prediction_type",
        type=str,
        default='epsilon',
    )

    # vkeilo add it
    parser.add_argument(
        "--classv_prompt",
        type=str,
        default='a pohto of sks person',
    )
    # vkeilo add it
    parser.add_argument(
        "--low_f_filter",
        type=float,
        default=-1,
    )
    args = parser.parse_args()
    return args

# 核心处理流程
def main(args):
    # 确保SGLD方法参数合法
    assert args.SGLD_method in ["allSGLD", "thetaSGLD", "deltaSGLD" ,"noSGLD","select_com"]
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
    # save array: perturbed_data to file
    
    # original_data当前为原始扰动数据
    original_data= copy.deepcopy(perturbed_data)
    # print(original_data[0])
    import torchvision
# vkeilo add it
    class EdgeDetectionTransform:
        def __init__(self, mode='sobel'):

            assert mode in ['sobel'], f"Unsupported mode: {mode}"
            self.mode = mode
            
            # Sobel 滤波器
            self.sobel_x = torch.tensor([[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.sobel_y = torch.tensor([[-1, -2, -1],
                                        [ 0,  0,  0],
                                        [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        def __call__(self, image):
            """
            执行边缘检测转换。
            参数:
            - image (Tensor): 输入图像张量，形状为 (C, H, W)
            
            返回:
            - Tensor: 每个通道独立边缘检测后的三通道图像。
            """
            self.sobel_x = self.sobel_x.to(dtype=image.dtype, device=image.device)
            self.sobel_y = self.sobel_y.to(dtype=image.dtype, device=image.device)
            if self.mode == 'sobel':
                edges = []
                
                # 对每个通道分别进行边缘检测
                for c in range(image.shape[1]):
                    channel = image[:,c:c+1, :, :]  # 获取单个通道
                    print(channel.shape)
                    edge_x = F.conv2d(channel, self.sobel_x, padding=1)
                    edge_y = F.conv2d(channel, self.sobel_y, padding=1)
                    
                    # 计算梯度幅值
                    magnitude = torch.sqrt(edge_x**2 + edge_y**2)
                    
                    # 如果需要将输出裁剪为[0, 1]范围
                    magnitude = torch.clamp(magnitude, 0, 1)
                    
                    edges.append(magnitude.squeeze(0))  # 积累每个通道的边缘结果
                
                # 将三个通道的边缘图合并
                edges = torch.stack(edges,2)
            print(f"filtered image shape:{edges.shape}")
            return edges.squeeze(1)

    # 定义训练和测试时的数据增强（图像处理）
    train_aug = [
            # 双线性插值到512x512
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # 裁剪
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    ]
    if args.use_edge_filter == 1:
        print("Using edge filter")
        train_aug = train_aug + [EdgeDetectionTransform()]
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
    two_trans = train_aug + defense_transform
    all_trans = transforms.Compose(all_trans)
    two_trans = transforms.Compose(two_trans)
    
    
    from robust_facecloak.attacks.worker.robust_pgd_worker_vk import RobustPGDAttacker
    from robust_facecloak.attacks.worker.robust_pan_worker_vk import RobustPANAttacker
    from MetaCloak.robust_facecloak.attacks.worker.pgd_worker import PGDAttacker
    from MetaCloak.robust_facecloak.attacks.worker.pan_worker import PANAttacker
    # 构建攻击者和防御者，攻击者使用PGD算法
    # attacker = PGDAttacker(
    #     radius=args.attack_pgd_radius, 
    #     steps=args.attack_pgd_step_num, 
    #     step_size=args.attack_pgd_step_size,
    #     random_start=args.attack_pgd_random_start,
    #     ascending=args.attack_pgd_ascending,
    #     args=args, 
    #     x_range=[-1, 1],
    # )
    # defender = RobustPGDAttacker(
    #     radius=args.defense_pgd_radius,
    #     steps=args.defense_pgd_step_num, # 6
    #     step_size=args.defense_pgd_step_size,
    #     random_start=args.defense_pgd_random_start,
    #     ascending=args.defense_pgd_ascending,
    #     args=args,
    #     attacker=attacker, 
    #     trans=all_trans,
    #     sample_num=args.defense_sample_num,
    #     x_range=[0, 255],
    # )
    if args.attack_mode == 'pgd':
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
            steps=args.defense_pgd_step_num, # 6
            step_size=args.defense_pgd_step_size,
            random_start=args.defense_pgd_random_start,
            ascending=args.defense_pgd_ascending,
            args=args,
            attacker=attacker, 
            trans=all_trans,
            sample_num=args.defense_sample_num,
            x_range=[0, 255],
            # vkeilo add it
            # step_sample_num=args.sampling_times_delta
        )
    elif args.attack_mode in ['pan']:
        attacker = PANAttacker(
            radius=args.defense_pgd_radius,
            steps=args.defense_pgd_step_num,
            step_size=args.defense_pgd_step_size,
            # ascending=args.defense_pgd_ascending,
            args=args,
            # trans=all_trans,
            # sample_num=args.defense_sample_num,
            x_range=[-1, 1],
            lambda_D = args.pan_lambda_D,
            lambda_S = args.pan_lambda_S,
            omiga = args.pan_omiga,
            k = args.pan_k,
            mode = args.pan_mode,
            use_val = args.pan_use_val,
            trans=two_trans,
            # attack = all_trans,
        )
    elif args.attack_mode in ['EOTpan','panrobust']:
        attacker = PANAttacker(
            radius=args.attack_pgd_radius,
            steps=args.attack_pgd_step_num,
            step_size=args.attack_pgd_step_size,
            # ascending=args.defense_pgd_ascending,
            args=args,
            # trans=all_trans,
            # sample_num=args.defense_sample_num,
            x_range=[0, 255],
            lambda_D = args.pan_lambda_D,
            lambda_S = args.pan_lambda_S,
            k = args.pan_k,
            mode = args.pan_mode,
            use_val = args.pan_use_val,
        )
        defender = RobustPANAttacker(
            radius=args.defense_pgd_radius,
            steps=args.defense_pgd_step_num, # 6
            step_size=args.defense_pgd_step_size,
            random_start=args.defense_pgd_random_start,
            ascending=args.defense_pgd_ascending,
            args=args,
            attacker=attacker, 
            trans=two_trans,
            sample_num=args.defense_sample_num,
            x_range=[0, 255],
            # vkeilo add it
            # step_sample_num=args.sampling_times_delta
        )

    # 模型加载，本次实验只有一个
    print(f'args model path:{args.pretrained_model_name_or_path}')
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
        if perturbed_data is None:
            return
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
    print(perturbed_data)
    save_image(perturbed_data, "load")
    
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
        # print('start train few 702')
        if args.init_model_state_pool_pth_path in [None,'None']:
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
        else:
            pre_trained_pth_path = args.init_model_state_pool_pth_path.split(',')[j]
            print(f'model {j} use trained pth:{pre_trained_pth_path}')
            with open(pre_trained_pth_path, 'rb') as f:
                model_pth_dict = pickle.load(f)
                init_model_state_pool[j] = model_pth_dict[0]
        # 释放占用的资源
        del f_ori, unet, text_encoder, tokenizer, noise_scheduler, vae
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        pbar.update(1)
    pbar.close()
    # 提取保存的中间状态的step数据（1000，2000，3000.....）        
    steps_list = list(init_model_state_pool[0].keys())
    print(steps_list)
    # 进度条，总train_few_step调用的次数*模型数量1*

    if args.total_gan_step == 0:
        total_gan_step = args.total_trail_num * num_models * (args.interval // args.advance_steps) * len(steps_list)
        pbar = tqdm(total=args.total_trail_num * num_models * (args.interval // args.advance_steps) * len(steps_list), desc="meta poison with model ensemble")
    else:
        total_gan_step = args.total_gan_step
        pbar = tqdm(total=total_gan_step, desc="meta poison with model ensemble")
    cnt=0
    # vkeilo add it  确定噪声强度
    theta_noise_epsion = args.sampling_step_theta * args.sampling_noise_ratio
    delta_noise_epsion = args.sampling_step_delta * args.sampling_noise_ratio

    def select_target_model(loss_log):
        # 选择loss最小的模型
        loss_mean_list = []
        for model_loss in loss_log:
            for steps,loss_list in model_loss.items():
                loss_mean_list.append(np.mean(loss_list))
        taget_index = np.argmin(loss_mean_list)
        model_i = taget_index // len(steps_list)
        split_step = steps_list[taget_index % len(steps_list)]
        return model_i, split_step
    # learning perturbation over the ensemble of models
    # 在多个模型集合上进行扰动优化
    # 多次实验
    # total_iterations = args.epochs * len(train_dataloader)
    # model_state_num = steps_list*len(num_models)
    # vkeilo add it
    _, _, _, _, vae = MODEL_BANKS[j]
    # args.class2target_v_a = get_class2target_v_a(args,vae,all_trans)
    args.weight_dtype = perturbed_data.dtype
    args.device = torch.device("cuda")
    # args.class2target_v_a = get_identify_feature_latents(args,vae,all_trans)
    args.class2target_v_a = get_mixed_features_v_a(args,vae,all_trans)
    del vae
    torch.cuda.empty_cache()
    assert args.model_select_mode in ['order','min_loss']
    print(f"avalaible model num: {num_models},available steps: {str(steps_list)},total train step :{str(args.total_train_steps)}")
    # 如果是平均顺序选择模型池中的模型
    perturbed_data_D = None
    # def get_mask(rate = 0.02):
    #     size = (3, 512, 512)

    #     # 计算需要填充1的元素数量
    #     num_elements = torch.prod(torch.tensor(size))
    #     num_ones = int(num_elements * rate)

    #     # 生成全零张量
    #     mask = torch.zeros(size)

    #     # 随机选择 num_ones 个索引，并设置为1
    #     indices = torch.randperm(num_elements)[:num_ones]
    #     mask.view(-1)[indices] = 1
    #     return mask
    # masks = []
    # for i in range(len(perturbed_data)):
    #     masks.append(get_mask(rate = 0.05))
    # masks = torch.stack(masks)
    if args.model_select_mode == 'order':
        for _ in range(args.total_trail_num):          
            # 针对每一个模型
            for model_i in range(num_models):
                print(f'using model {model_i}')
                # 确定关键组件
                start_time = time.time()
                text_encoder, unet, tokenizer, noise_scheduler, vae = MODEL_BANKS[model_i]
                # 对于每一个中间状态step
                for split_step in steps_list: 
                    # 加载unet和文本编码器的中间状态参数
                    unet.load_state_dict(init_model_state_pool[model_i][split_step]["unet"])
                    text_encoder.load_state_dict(init_model_state_pool[model_i][split_step]["text_encoder"])
                    f = [unet, text_encoder]
                    # f = [unet.to(device_1), text_encoder.to(device_1)]
                    
                    # 每advance_steps步进行一次防御优化/对于每一组模型参数，进行200/2=100次对抗训练
                    this_modle_step = total_gan_step // args.total_trail_num//num_models//(len(steps_list))
                    print(f'start {this_modle_step} times of defense optimization in step-{split_step} model')
                    for j in range(this_modle_step):
                        # 更新一次扰动，使得扰动更加强大,后续需要在此处引入随机性（多轮采样优化），并以扰动的平均值作为后续的扰动
                        # vkeilo add it
                        mean_delta = perturbed_data.clone().detach()
                        print(f'start {args.sampling_times_delta} times of delta sampling ')
                        for k in range(args.sampling_times_delta):
                            print(f'sample delta {k}/{args.sampling_times_delta} times')
                            # vkeilo add it
                            # adv_x_ori = perturbed_data.clone()
                            if args.attack_mode in ["pgd","EOTpan","panrobust"]:
                                perturbed_data,rubust_loss = defender.perturb(f, perturbed_data, original_data, vae, tokenizer, noise_scheduler,)
                            elif args.attack_mode in ["pan"]:
                                perturbed_data,perturbed_data_D,rubust_loss = attacker.attack(f, original_data, vae, tokenizer, noise_scheduler,)
                            # perturbed_data = adv_x_ori+(perturbed_data-adv_x_ori)*masks
                            wandb.log({"perturbedloss": rubust_loss})
                            # 此处引入随机梯度朗之万动力学
                            if args.SGLD_method == 'allSGLD' or args.SGLD_method == 'deltaSGLD':
                                perturbed_data = utils.SGLD(perturbed_data, args.sampling_step_delta, delta_noise_epsion).detach()
                            mean_delta = args.beta_s * mean_delta + (1 - args.beta_s) * perturbed_data.to('cpu')
                        mean_delta.detach()
                        perturbed_data = mean_delta
                        print(f"max pixel change:{find_max_pixel_change(perturbed_data, original_data)}")
                        # f[0] = f[0].to(device_0)
                        # f[1] = f[1].to(device_0)
                        # perturbed_data = defender.perturb(f, perturbed_data, original_data, vae, tokenizer, noise_scheduler)
                        
                        # 扰动优化次数更新 +1
                        cnt+=1
                        # 在新的扰动数据下，训练advance_steps步，后续需要在此处引入随机性（多轮采样优化参数），并以参数的平均值作为模型的参数
                        back_parameters_list = [f[0].state_dict(),
                                                f[1].state_dict()]

                        mean_theta_list = [f[0].state_dict(),
                                        f[1].state_dict()]
                        
                        # print(f'start {args.sampling_times_theta} times of theta sampling')
                        for k in range(args.sampling_times_theta):
                            print(f'sample theta {k}/{args.sampling_times_theta} times')
                            f = train_few_step(
                                args,
                                f,
                                tokenizer,
                                noise_scheduler,
                                vae,
                                perturbed_data.float(),
                                args.advance_steps,
                                # device = device_1
                                dpcopy = False,
                                task_loss_name='model_theta_loss',
                            )
                            torch.cuda.empty_cache()
                            for model_index, model in enumerate(f):
                                # print(f"\nbefore culcu, GPU: {gpu.name}, Free Memory: {gpu.memoryFree / 1024:.2f} GB")
                                for name, p in model.named_parameters():
                                    # 先尝试固定学习率的（因为迭代次数暂未确定）
                                    # lr_now = lr_scheduler.get_last_lr()[0]
                                    # 参数采样,引入随机性
                                    if args.SGLD_method == 'allSGLD' or args.SGLD_method == 'thetaSGLD':
                                        p.data = utils.SGLD(p.data, args.sampling_step_theta, theta_noise_epsion)
                                    # 模型参数也使用指数平均
                                    # mean_theta_list[model_index][name] = args.beta_s * mean_theta_list[model_index][name] + (1 - args.beta_s) * p.data.to('cpu')
                                    # mean_theta_list[model_index][name] = args.beta_s * mean_theta_list[model_index][name] + (1 - args.beta_s) * p.data
                                    mean_theta_list[model_index][name].mul_(args.beta_s).add_((1 - args.beta_s) * p.data)
                                # print(f"\nafter calcu params, GPU: {gpu.name}, Free Memory: {gpu.memoryFree / 1024:.2f} GB")
                                # torch.cuda.empty_cache()
                        # lr_scheduler.step()
                        # 对于模型的unet和文本编码器，分别更新参数
                        for back_parameters, mean_theta in zip(back_parameters_list,mean_theta_list):
                            for name in back_parameters:
                                back_parameters[name] = args.beta_p * back_parameters[name] + (1 - args.beta_p) * mean_theta[name]
                                # back_parameters[name] = back_parameters[name].float()
                                # back_parameters[name].mul_(args.beta_p).add_((1 - args.beta_p) * mean_theta[name])
                        for index, model in enumerate(f):
                            # model.load_state_dict({k: v.to(device_g) for k, v in back_parameters_list[index].items()})
                            model.load_state_dict(back_parameters_list[index])
                            pass
                        del back_parameters_list
                        del mean_theta_list
                        gc.collect()
                        torch.cuda.empty_cache()
                        # f = train_few_step(
                        #     args,
                        #     f,
                        #     tokenizer,
                        #     noise_scheduler,
                        #     vae,
                        #     perturbed_data.float(),
                        #     args.advance_steps,
                        # )
                        pbar.update(1)
                        # 每1000次扰动优化，保存一次扰动示例图像
                        if cnt % args.img_save_interval == 0:
                            save_image(perturbed_data, f"{cnt}")
                            save_image(perturbed_data_D,f"{cnt}_D")
                    # frequently release the memory due to limited GPU memory, 
                    # env with more gpu might consider to remove the following lines for boosting speed
                    # 释放资源
                    del f 
                    torch.cuda.empty_cache()
                end_time = time.time()
                logger.info(f"model {model_i} adversarial training Time cost: {(end_time - start_time) / 60} min")
                wandb.log({f"Time cost of model {model_i} adversarial training": (end_time - start_time) / 60})
                del unet, text_encoder, tokenizer, noise_scheduler, vae

                if torch.cuda.is_available():
                    torch.cuda.empty_cache() 

            import gc
            gc.collect()
            torch.cuda.empty_cache()      
        end_time = time.time()
        import gc
        gc.collect()
        torch.cuda.empty_cache()   
    pbar.close()
    # 保存最后的结果
    save_image(perturbed_data, "final")

    print(f"max_noise_r {find_max_pixel_change(perturbed_data, original_data)}")
    noise_L0 = get_L0(perturbed_data, original_data)
    noise_L1 = get_L1(perturbed_data, original_data)
    noise_p = get_change_p(perturbed_data, original_data)
    ciede2000_score = get_ciede2000_diff(original_data, perturbed_data)
    print(f"noise_L0 {noise_L0:.2f}")
    print(f"pix_change_mean {noise_L1/(512*512)/2:.2f}")
    print(f"change_area_mean {noise_p*100:.2f}")
    print(f"ciede2000_score {ciede2000_score:.2f}")

def find_max_pixel_change(original_img, noisy_img):
    diff = torch.abs(original_img - noisy_img)
    
    # Find the maximum pixel difference
    max_change = torch.max(diff)
    
    return max_change.item()

def get_L0(original_img, noisy_img):
    diff = torch.abs(original_img - noisy_img)
    diff_L0 = torch.sum(diff > 0, dim=(1, 2, 3))
    # Find the maximum pixel difference
    mean_L0 = torch.mean(diff_L0.float())
    return mean_L0.item()

def get_L1(original_img, noisy_img):
    diff = torch.abs(original_img - noisy_img)
    diff_L1 = torch.sum(diff, dim=(1, 2, 3))
    mean_L1 = torch.mean(diff_L1.float())
    return mean_L1.item()

def get_change_p(original_img, noisy_img):
    diff = torch.abs(original_img - noisy_img)
    diff_L0_all = torch.sum(diff > 0, dim=(0, 1, 2, 3))
    pix_num_all = original_img.shape[0] * original_img.shape[1] * original_img.shape[2] * original_img.shape[3]
    change_p = diff_L0_all / pix_num_all
    return change_p.item()

def get_ciede2000_diff(ori_imgs,advimgs):
    device = torch.device('cuda')
    ori_imgs_0_1 = ori_imgs/255
    advimgs_0_1 = advimgs/255
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
    return torch.mean(scores)

if __name__ == "__main__":
    # 获取脚本传参
    args = parse_args()
    print(args)
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity_name, name=args.wandb_run_name)
    wandb.config.update(args)
    wandb.log({'status': 'gen'})
    # 核心代码
    main(args)
