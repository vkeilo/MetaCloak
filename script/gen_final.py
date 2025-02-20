import os
import argparse
import numpy as np
from PIL import Image

def process_images(path_a, path_b, path_c, path_d):
    # 创建路径D下的final目录
    final_path = os.path.join(path_d, "final")
    os.makedirs(final_path, exist_ok=True)
    
    # 获取C目录下的图片列表
    images = os.listdir(path_b)
    images = [img for img in images if img.endswith(('.png', '.jpg', '.jpeg'))]  # 过滤图片文件
    
    for image_name in images:
        print(f'processing {image_name}')
        # 构建每个路径下对应图片的完整路径
        path_img_a = os.path.join(path_a, image_name)
        path_img_b = os.path.join(path_b, image_name)
        path_img_c = os.path.join(path_c, "noisy_"+image_name)
        path_img_final = os.path.join(final_path, "noisy_"+image_name)
        
        # 确保图片在A、B、C中都存在
        if not (os.path.exists(path_img_a) and os.path.exists(path_img_b) and os.path.exists(path_img_c)):
            print(f"Warning: {image_name} does not exist in all required directories.")
            continue
        
        # 读取图片
        img_a = np.array(Image.open(path_img_a).convert("RGB"), dtype=np.float32)
        img_b = np.array(Image.open(path_img_b).convert("RGB"), dtype=np.float32)
        img_c = np.array(Image.open(path_img_c).convert("RGB"), dtype=np.float32)
        
        # 计算扰动图片
        perturbation = img_c - img_b
        
        # 加入扰动到A中的图片
        modified_img = img_a + perturbation
        modified_img = np.clip(modified_img, 0, 255).astype(np.uint8)  # 规范到0～255范围
        
        # 保存处理后的图片到D/final
        Image.fromarray(modified_img).save(path_img_final)
        print(f"save to {path_img_final}")
        # 保存直接覆盖A中的图片
        # Image.fromarray(modified_img).save(path_img_a)

    print(f"Processing complete. Modified images saved to {final_path} and updated in {path_a}.")

def main():
    parser = argparse.ArgumentParser(description="Process images by adding perturbations.")
    parser.add_argument("--path_a", required=True, help="Path to directory A (destination images)")
    parser.add_argument("--path_b", required=True, help="Path to directory B (original images)")
    parser.add_argument("--path_c", required=True, help="Path to directory C (perturbation images)")
    parser.add_argument("--path_d", required=True, help="Path to directory D (output directory)")

    args = parser.parse_args()
    
    process_images(args.path_a, args.path_b, args.path_c, args.path_d)

if __name__ == "__main__":
    main()


# path_a: /data/home/yekai/github/MetaCloak/tmp/set_B
# path_b: /data/home/yekai/github/MetaCloak/tmp/PAN_VGGFace2_r6rd8_eval0_600to800_trail4_omiga2e-5_idx28_total120_lambdasp15e-1_202501011_test/exp_data_MAT-PAN-VGGFace2-id28-pan-totle120-200-1-1-x6x0s1-radius6-noSGLD-minL600-Linterval200-robust7-order-4e-07-0-0.13894954943731375-k=2-useS-last-omigas2e-5-1736582151/gen_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7/dataset-VGGFace2-clean-r-6-model-SD21base-gen_prompt-sks/28/image_before_addding_noise
# path_c: /data/home/yekai/github/MetaCloak/tmp/PAN_VGGFace2_r6rd8_eval0_600to800_trail4_omiga2e-5_idx28_total120_lambdasp15e-1_202501011_test/exp_data_MAT-PAN-VGGFace2-id28-pan-totle120-200-1-1-x6x0s1-radius6-noSGLD-minL600-Linterval200-robust7-order-4e-07-0-0.13894954943731375-k=2-useS-last-omigas2e-5-1736582151/gen_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7/dataset-VGGFace2-clean-r-6-model-SD21base-gen_prompt-sks/28/noise-ckpt/final_ori
# path_d: /data/home/yekai/github/MetaCloak/tmp/PAN_VGGFace2_r6rd8_eval0_600to800_trail4_omiga2e-5_idx28_total120_lambdasp15e-1_202501011_test/exp_data_MAT-PAN-VGGFace2-id28-pan-totle120-200-1-1-x6x0s1-radius6-noSGLD-minL600-Linterval200-robust7-order-4e-07-0-0.13894954943731375-k=2-useS-last-omigas2e-5-1736582151/gen_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD21base-robust-gauK-7/dataset-VGGFace2-clean-r-6-model-SD21base-gen_prompt-sks/28/noise-ckpt