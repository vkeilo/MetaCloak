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
test_lable = "PAN_r11rd11_ciede2000_s8e-8~12e-8_omiga200~500_sameseed"
params_options = {
    "MODEL_ROOT": ["${ADB_PROJECT_ROOT}"],
    "gen_model_path": ["${MODEL_ROOT}/SD/stable-diffusion-2-1-base"],
    "init_model_state_pool_pth_path": [
        "${MODEL_ROOT}/robust_facecloak/attacks/algs/tmpdata/init_model_state_pool_sd2-1.pth"
    ],
    "model_select_mode":["order"],
    "wandb_project_name": ["metacloak_PAN"],
    "mixed_precision": ["fp16"],
    "advance_steps": [2],
    "total_trail_num": [4],
    "total_train_steps": [1000],
    "total_gan_step":[120],
    "interval": [200],
    "dreambooth_training_steps": [1000],
    "unroll_steps": [1],
    "defense_sample_num": [1],
    "defense_pgd_step_num": [6],
    "sampling_times_delta": [1],
    "sampling_times_theta": [1],
    "attack_pgd_step_num": [0],
    "attack_pgd_radius": [7],
    "r": [40],
    "SGLD_method": ["noSGLD"],
    "gauK": [7],
    "eval_gen_img_num": [16],
    "attack_mode": ["pan"],
    "pan_lambda_D": [0],
    "pan_lambda_S": [8e-8,1e-7,12e-8],  # Multiple values for pan_lambda_S
    "pan_omiga": "log:200:500:5",
    "pan_k": [2],  # Multiple values for pan_k
    "pan_mode": ["S"],
    "pan_use_val": ["last"],
    "img_save_interval":[10],
    "select_model_index":[1]
}

# Number of times to repeat each configuration
repeat_times = 5


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