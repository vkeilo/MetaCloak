
import os
import sys

sys.path.append("/data/home/yekai/github/mypro/MetaCloak")

from eval_score import get_score
import numpy as np

exp_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori"

ori_pics_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori/gen_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7/dataset-VGGFace2-clean-r-11-model-SD21base-gen_prompt-sks/0/image_clean_ref"
gen_pics_dir = "/data/home/yekai/github/mypro/MetaCloak/exp_data-ori/train_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7-gau-gau-eval/gen-release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7-dataset-VGGFace2-clean-r-11-model-SD21base-gen_prompt-sks-eval-gau-rate-/0_DREAMBOOTH/checkpoint-1000/dreambooth/a_dslr_photo_of_sks_person"

score_dict = get_score(gen_pics_dir,ori_pics_dir)
print(score_dict)
k_list = list(score_dict.keys())
for k in k_list:
    means = []
    means.append(score_dict[k])
    print(f"{k}_mean: {np.mean(means)}")
    stds = []
    stds.append(score_dict[k])
    print(f"{k}_std: {np.std(stds)}")

# print(score_dict)