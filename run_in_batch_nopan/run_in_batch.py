import os
import json
import time

def test_one_args(args):
    for k,v in args.items():
        os.environ[k] = str(v)
    # os.chdir("..")
    # bash run : nohup bash script/gen_and_eval_vk.sh > output_MAT-1000-200-6-6-x1x1-radius11-allSGLD-rubust0.log 2>&1
    os.environ["test_timestamp"] = str(int(time.time()))
    os.environ["wandb_run_name"] = f"MAT-{os.getenv('total_train_steps')}-{os.getenv('interval')}-{os.getenv('sampling_times_delta')}-{os.getenv('sampling_times_theta')}-x{os.getenv('defense_pgd_step_num')}x{os.getenv('attack_pgd_step_num')}-radius{os.getenv('r')}-{os.getenv('SGLD_method')}-robust{os.getenv('attack_pgd_radius')}-{os.getenv('test_timestamp')}"
    # python 实现 export test_timestamp=$(date +%s)
    run_name = os.getenv("wandb_run_name")
    print(f"run_name: {run_name}")
    os.system(f"nohup bash script/gen_and_eval_vk_batch.sh > output_{run_name}.log 2>&1")
    check_file_for_pattern(f"output_{run_name}.log","find function last")
    # rename dir exp_data to exp_data_{run_name}
    os.system(f"mv exp_data exp_datas_output/exp_data_{run_name}")
    os.system(f"mv output_{run_name}.log logs_output/output_{run_name}.log")
    return run_name

def update_finished_json(finished_log_json_path, run_name):
    finished_file = json.load(open(finished_log_json_path))
    # if json is empty, add key finished_args_list and value []
    if "finished_args_list" not in finished_file:
        finished_file["finished_args_list"] = []
    finished_file["finished_args_list"].append(run_name)
    json.dump(finished_file, open(finished_log_json_path, "w"))

def update_untest_json(untest_args_json_path):
    json_dict = json.load(open(untest_args_json_path))
    json_dict["untest_args_list"].pop(0)
    json.dump(json_dict, open(untest_args_json_path, "w"))

def check_file_for_pattern(file_path, pattern="find function last"):
    while True:
        try:
            # 打开文件并读取最后一行
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if lines:
                    last_line = lines[-1].strip()  # 获取最后一行并去除两边空白
                    print(f"检测到的最后一行: {last_line}")
                    # 检查最后一行是否以指定模式开头
                    if last_line.startswith(pattern):
                        print("找到匹配的行，退出检测。")
                        return last_line
        except Exception as e:
            print(f"读取文件时出错: {e}")
        
        # 等待 3 分钟（180 秒）
        print("未找到匹配的行，等待 3 分钟后重新检测...")
        time.sleep(180)

if __name__ == "__main__":
    untest_args_json_path  = "run_in_batch/untest.json"
    finished_log_json_path = "run_in_batch/finished.json"
    untest_file_con = json.load(open(untest_args_json_path))
    untest_args_list = untest_file_con["untest_args_list"].copy()
    for args in untest_args_list:
        print(f"start run :{args}")
        finished_name = test_one_args(args)
        print(f"finished run :{finished_name}")
        update_untest_json(untest_args_json_path)
        update_finished_json(finished_log_json_path, finished_name)



    