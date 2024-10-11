import itertools
import json
import copy
import os

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

# Define the possible values for each parameter
test_lable = "100-20-pan"
params_options = {
    "MODEL_ROOT": ["${ADB_PROJECT_ROOT}"],
    "gen_model_path": ["${MODEL_ROOT}/SD/stable-diffusion-2-1-base"],
    "init_model_state_pool_pth_path": [
        "${MODEL_ROOT}/robust_facecloak/attacks/algs/tmpdata/init_model_state_pool_sd2-1.pth"
    ],
    "model_select_mode":["order"],
    "wandb_project_name": ["metacloak_PAN"],
    "mixed_precision": ["bf16"],
    "advance_steps": [2],
    "total_trail_num": [2],
    "interval": [20],
    "dreambooth_training_steps": [100],
    "total_train_steps": [1000],
    "unroll_steps": [1],
    "defense_sample_num": [1],
    "defense_pgd_step_num": [1,3,9],
    "sampling_times_delta": [1],
    "sampling_times_theta": [1],
    "attack_pgd_step_num": [1],
    "attack_pgd_radius": [0],
    "r": [11],
    "SGLD_method": ["noSGLD","deltaSGLD"],
    "gauK": [7],
    "eval_gen_img_num": [16],
    "attack_mode": ["pan"],
    "pan_lambda_D": [0.01, 0.001],
    "pan_lambda_S": [1000,  10, 0],  # Multiple values for pan_lambda_S
    "pan_omiga": [0.5,1],
    "pan_k": [2],  # Multiple values for pan_k
    "pan_mode": ["S"],
    "pan_use_val": ["last"]
}

# Number of times to repeat each configuration
repeat_times = 3

# Generate all combinations
experiments = generate_combinations(params_options, repeat_times=repeat_times)

# Wrap in "untest_args_list" for proper JSON structure
output = {"test_lable":test_lable ,"untest_args_list": experiments}

# Print the generated combinations as JSON
print(json.dumps(output, indent=4))

py_path = os.path.dirname(os.path.abspath(__file__))
# Save to a JSON file
with open(f"{py_path}/experiment_params.json", "w") as outfile:
    json.dump(output, outfile, indent=4)