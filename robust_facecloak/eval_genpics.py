
import os
import sys
import torch
sys.path.append("/data/home/yekai/github/mypro/MetaCloak")

from eval_score import get_score
import numpy as np
from robust_facecloak.attacks.worker.differential_color_functions import rgb2lab_diff, ciede2000_diff
from robust_facecloak.generic.data_utils import PromptDataset, load_data

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



exp_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori"

ori_pics_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori/gen_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7/dataset-VGGFace2-clean-r-11-model-SD21base-gen_prompt-sks/0/image_clean_ref"
noisy_pics_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori/gen_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7/dataset-VGGFace2-clean-r-11-model-SD21base-gen_prompt-sks/0/noise-ckpt/final"
gen_pics_dir = "/data/home/yekai/github/mypro/MetaCloak/robust_facecloak/a_photo_of_sks_person"

score_dict = get_score(gen_pics_dir,ori_pics_dir)
print(score_dict)
k_list = list(score_dict.keys())
for k in k_list:
    means = []
    means.append(score_dict[k])
    print(f"{k}_mean {np.mean(means)}")
    stds = []
    stds.append(score_dict[k])
    print(f"{k}_std {np.std(stds)}")



original_data = load_data(ori_pics_dir)
perturbed_data = load_data(noisy_pics_dir)

max_noise_r = find_max_pixel_change(perturbed_data, original_data)
noise_L0 = get_L0(perturbed_data, original_data)
noise_L1 = get_L1(perturbed_data, original_data)
noise_p = get_change_p(perturbed_data, original_data)
ciede2000_score = get_ciede2000_diff(original_data, perturbed_data)
score_dict['max_noise_r'] = max_noise_r
score_dict['noise_L0'] = noise_L0
score_dict['pix_change_mean'] = noise_L1
score_dict['change_area_mean'] = noise_p*100
score_dict['ciede2000_score'] = ciede2000_score
print(f"max_noise_r {max_noise_r:.2f}")
print(f"noise_L0 {noise_L0:.2f}")
print(f"pix_change_mean {noise_L1:.2f}")
print(f"change_area_mean {noise_p*100:.2f}")
print(f"ciede2000_score {ciede2000_score:.2f}")

# print(score_dict)