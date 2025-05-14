import itertools
import json
import copy
import os
import numpy as np

def generate_combinations(base_params, repeat_times=1):
    """
    Generate all possible parameter combinations from a dictionary of parameter lists and repeat each combination.
    
    :param base_params: A dictionary where the values are lists of possible values for each parameter.
    :param repeat_times: Number of times each experiment configuration should be repeated.
    :return: A list of dictionaries, each representing one combination of parameters.
    """
    # Extract parameter keys and value lists
    keys = base_params.keys()
    value_lists = base_params.values()
    
    # Generate all combinations of parameter values
    combinations = list(itertools.product(*value_lists))
    
    # Convert combinations into a list of dictionaries and repeat them
    experiments = []
    for combination in combinations:
        experiment_params = dict(zip(keys, combination))
        for _ in range(repeat_times):  # Repeat each combination 'repeat_times' times
            experiments.append(copy.deepcopy(experiment_params))
    
    return experiments

def generate_log_interval_list(start, end, num):
    """
    生成在指定区间内的对数间隔列表。
    
    参数:
    start (int or float): 区间开始值。
    end (int or float): 区间结束值。
    num (int): 列表中值的数量。
    
    返回:
    list: 对数间隔的列表，包含指定数量的值。
    """
    return np.logspace(np.log10(start), np.log10(end), num=num).tolist()

def generate_lin_interval_list(start, end, num):
    """
    生成在指定区间内的线性间隔列表。

    参数:
    start (int or float): 区间开始值。
    end (int or float): 区间结束值。
    num (int): 列表中值的数量。

    返回:
    list: 线性间隔的列表，包含指定数量的值。
    """
    return np.linspace(start, end, num=num).tolist()

# Define the possible values for each parameter
# test_lable = "Ori_VGGFace2_r6_eval0_idx50_textnoise_total480_20250218_part1"
# test_lable = "PAN_VGGFace2_r6rd6_eval0_idx50_lambdaS_5e-5_omigax10_total120_20250325"
# test_lable = "Orimetacloak_total480_id0_20250216_edgefilter_compare_partyes"
# test_lable = "newloss_vfromclass2target_noisetext_idx50_r6_20250224"
# test_lable = "Orimetacloak_dataMetaCloak480_r6_train0eval0_id0_total120_timeselect03_classv_prompt2Monet_20250317"
# test_lable = "Purl_PGDpeg_r30_20250318"
# test_lable = "Orimetacloak_VGGFace2_r6_eval0_idx50_total480_timestepfinal200_vpre_20250409_p2"
# test_lable = "PAN_VGGFace2_r7rd7_eval0_idx50_lambdaS_1e-4_omiga0599_total120_20250401"
# test_lable = "PAN_VGGFace2_r10rd10_eval0_idx50_lambdaS_1e-4_omiga0599_total120_20250401_x-22"
# test_lable = "LF_test_sametime07_r11_20250411_p3"
test_lable = "MetaCloak_SD21_VGGFace2_random50_r16_50~28"
# test_lable = "CW_test3"
repeat_times = 1
params_options = {
    "MODEL_ROOT": ["${ADB_PROJECT_ROOT}"],
    "gen_model_path": ["${MODEL_ROOT}/SD/stable-diffusion-2-1-base"],
    "init_model_state_pool_pth_path": [
        "None"
        # "${MODEL_ROOT}/SD/init_model_state_pool_sd2-1.pth"
    ],
    "dataset_name": ["VGGFace2-clean"],
    "instance_name": [i for i in range(49,28,-1)],
    # "instance_name": [0,1],
    "model_select_mode":["order"],
    "wandb_project_name": ["metacloak_PAN"],
    "mixed_precision": ["fp16"],
    "advance_steps": [2],
    "total_trail_num":[4],
    "total_train_steps": [1000],
    "total_gan_step":[0],
    "interval": [200],
    "dreambooth_training_steps": [1000],
    "step_size": [1],
    "unroll_steps": [1],
    "defense_sample_num": [1],
    "defense_pgd_step_num": [6],
    # "sampling_times_delta": [1],
    # "sampling_times_theta": [1],
    # "sampling_noise_ratio": [0.2],
    # "sampling_step_delta":  [1e-3],
    # "sampling_step_theta":  [1e-5],
    # "beta_s": [0.3],
    # "beta_p": [0.3],
    # "mat_lambda_s": [0],
    "attack_pgd_step_num": [0],
    "attack_pgd_radius": [7],
    "r": [16],
    # "rd": [11],
    "time_select": [1],
    "SGLD_method": ["noSGLD"],
    "gauK": [7],
    "eval_gen_img_num": [16],
    "train_mode": ["gau"],
    "eval_mode": ["no"],
    "img_save_interval":[600],
    "select_model_index":[0],
    "attack_mode": ["pgd"],
    "loss_mode":["mse"],
    # "WANDB_MODE": ["disabled"],
    # "diff_time_diff_loss":["0"],
    # "time_window_start":['0.2'],
    # "time_window_end":['0.4'],
    # "time_window_len":['0.02'],
    # # "classv_prompt": ["Oil painting in Monet's Water Lilies style"],
    # "low_f_filter": [-1],
    # # "classv_prompt": ["The metaphysical essence of nonexistence","A person captured in extreme motion blur","Oil painting in Monet's Water Lilies style","Charcoal sketch with rough hatching","a pho to ofs ksp son","random noise","Solid color canvas","Jackie Chan","a big blue door","a baby with delicate skin","a black man"],
    # # "classv_prompt":["a photo of sks person","a big blue door","The metaphysical essence of nonexistence","Oil painting in Monet's Water Lilies style","Jackie Chan"],
    # # "prediction_type":["v_prediction"],
    # "pan_lambda_D": [0],
    # # 【1e-4，2e-4，4e-4，8e-4，16e-4】
    # "pan_lambda_S": [1e-4],  # Multiple values for pan_lambda_S
    # "use_edge_filter":[0],
    # "use_unet_noise":[0],
    # "use_text_noise":[0],
    # "unet_noise_r":[0.034866576443999764/1.8],
    # "text_noise_r":[0.023099106894149842],
    # # 0.13894954943731375
    # # [0.1, 0.129, 0.167, 0.215, 0.278, 0.359, 0.464, 0.599]
    # # [0.774,1]
    # #  [0.1, 0.129, 0.167, 0.215, 0.278, 0.359, 0.464, 0.599,0.774,1]
    # "pan_omiga": [0.599],
    # "pan_k": [2],  # Multiple values for pan_k
    # "pan_mode": ["S"],
    # "pan_use_val": ["last"],
    # "Ltype":['ciede2000'],
    # "interval_L": [0],
    # "min_L": [0],
    # "hpara_update_interval":[5],
    # "dynamic_mode":['L_only'],
    # "omiga_strength": [2e-5]
    
}

# test_lable = "Orimetacloak_VGGFace2_r6_eval0_idx50_total480_randomtimeattack_mse_20250326"
# repeat_times = 1
# params_options = {
#     "MODEL_ROOT": ["${ADB_PROJECT_ROOT}"],
#     "gen_model_path": ["${MODEL_ROOT}/SD/stable-diffusion-2-1-base"],
#     "init_model_state_pool_pth_path": [
#         "None"
#         # "${MODEL_ROOT}/SD/init_model_state_pool_sd2-1.pth"
#     ],
#     "dataset_name": ["VGGFace2-clean"],
#     "instance_name": [i for i in range(50)],
#     "model_select_mode":["order"],
#     "wandb_project_name": ["metacloak_PAN"],
#     "mixed_precision": ["fp16"],
#     "advance_steps": [2],
#     "total_trail_num":[4],
#     "total_train_steps": [1000],
#     "total_gan_step":[480],
#     "interval": [200],
#     "dreambooth_training_steps": [1000],
#     "step_size": [1],
#     "unroll_steps": [1],
#     "defense_sample_num": [1],
#     "defense_pgd_step_num": [6],
#     "sampling_times_delta": [1],
#     "sampling_times_theta": [1],
#     "sampling_noise_ratio": [0.2],
#     "sampling_step_delta":  [1e-3],
#     "sampling_step_theta":  [1e-5],
#     "beta_s": [0.3],
#     "beta_p": [0.3],
#     "mat_lambda_s": [0],
#     "attack_pgd_step_num": [0],
#     "attack_pgd_radius": [7],
#     "r": [6],
#     "rd": [6],
#     "time_select": [1],
#     "SGLD_method": ["noSGLD"],
#     "gauK": [7],
#     "eval_gen_img_num": [16],
#     "train_mode": ["gau"],
#     "eval_mode": ["no"],
#     "img_save_interval":[120],
#     "select_model_index":[0],
#     "attack_mode": ["pgd"],
#     "loss_mode":["mse"],
#     "diff_time_diff_loss":["2"],
#     "time_window_start":['0.05'],
#     "time_window_end":['0.8'],
#     "time_window_len":['0.04'],
#     # "classv_prompt": ["Oil painting in Monet's Water Lilies style"],
#     "low_f_filter": [-1],
#     # "classv_prompt": ["The metaphysical essence of nonexistence","A person captured in extreme motion blur","Oil painting in Monet's Water Lilies style","Charcoal sketch with rough hatching","a pho to ofs ksp son","random noise","Solid color canvas","Jackie Chan","a big blue door","a baby with delicate skin","a black man"],
#     # "classv_prompt":["a photo of sks person","a big blue door","The metaphysical essence of nonexistence","Oil painting in Monet's Water Lilies style","Jackie Chan"],
#     # "prediction_type":["v_prediction"],
#     "pan_lambda_D": [0],
#     # 【1e-4，2e-4，4e-4，8e-4，16e-4】
#     "pan_lambda_S": [5e-5],  # Multiple values for pan_lambda_S
#     "use_edge_filter":[0],
#     "use_unet_noise":[0],
#     "use_text_noise":[0],
#     "unet_noise_r":[0.034866576443999764/1.8],
#     "text_noise_r":[0.023099106894149842],
#     # 0.13894954943731375
#     # [0.1, 0.129, 0.167, 0.215, 0.278, 0.359, 0.464, 0.599]
#     # [0.774,1]
#     #  [0.1, 0.129, 0.167, 0.215, 0.278, 0.359, 0.464, 0.599,0.774,1]
#     "pan_omiga": [0.1],
#     "pan_k": [2],  # Multiple values for pan_k
#     "pan_mode": ["S"],
#     "pan_use_val": ["last"],
#     "Ltype":['ciede2000'],
#     "interval_L": [0],
#     "min_L": [0],
#     "hpara_update_interval":[5],
#     "dynamic_mode":['L_only'],
#     "omiga_strength": [2e-5]
    
# }



for key, value in params_options.items():
    use_log = True
    if type(value) is list:
        continue
    if value.startswith("lin:") or value.startswith("log:"):
        if value.startswith("lin:"):
            use_log = False
        args = value.split(":")[1:]
        assert len(args) == 3, "Invalid format for linear/log interval list"
        start, end, num = map(float, args)
        num = int(num)
        if use_log:
            params_options[key] = generate_log_interval_list(start, end, num)
        else:
            params_options[key] = generate_lin_interval_list(start, end, num)

# Generate all combinations
experiments = generate_combinations(params_options, repeat_times=repeat_times)

# Wrap in "untest_args_list" for proper JSON structure
output = {"test_lable":test_lable ,"settings":params_options,"untest_args_list": experiments}

# Print the generated combinations as JSON
print(json.dumps(output, indent=4))

py_path = os.path.dirname(os.path.abspath(__file__))
# Save to a JSON file
with open(f"{py_path}/{test_lable}.json", "w") as outfile:
    json.dump(output, outfile, indent=4)
