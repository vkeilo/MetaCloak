
if [ "$method_select" = "metacloak" ]; then
    echo "method_select = metacloak"
    export method_name=MetaCloak
    export gen_file_name="metacloak_vk.sh"
    export advance_steps=2; 
    export total_trail_num=4; 
    # export interval=200; 
    export interval=2; 
    # export total_train_steps=1000; 
    export total_train_steps=10; 
    export unroll_steps=1;  # 向前预训练多步，默认为1
    # vkeilo change it from default 1 to 10
    export defense_sample_num=1; # 这个应该为1，每个step更新一次扰动
    export defense_pgd_step_num=1; # 这个是一轮扰动计算的扰动更新次数,默认为6
    export sampling_times_theta=1;
    export sampling_times_delta=2;
    export attack_pgd_step_num=1; # 鲁棒攻击步数,默认为3
    export attack_pgd_radius=0;   # 鲁棒性对抗训练，设置为0就是不需要进行鲁棒性对抗，默认为0
    export SGLD_method="deltaSGLD";
    export ref_model_path=$gen_model_path;
    export test_timestamp=$(date +%s)
    export method_hyper=advance_steps-$advance_steps-total_trail_num-$total_trail_num-unroll_steps-$unroll_steps-interval-$interval-total_train_steps-$total_train_steps-$model_name
    export wandb_project_name="metacloak_test1"
    export wandb_run_name="MAT-${total_train_steps}-${interval}-${sampling_times_delta}-${sampling_times_theta}-x${defense_pgd_step_num}x${attack_pgd_step_num}-radius${r}-${SGLD_method}-robust${attack_pgd_radius}-${test_timestamp}"
    if [ "$train_mode" = "gau" ]; then
        export method_hyper=$method_hyper-robust-gauK-$gauK
    fi
elif [ "$method_select" = "clean" ]; then
    export method_name=Clean
    export gen_file_name="Clean.sh"
    export method_hyper=""
else
    echo "Invalid method name $method_select"
    exit 1
fi

# nohup bash script/gen_and_eval_vk.sh > output_MAT-1000-200-6-6-x1x1-radius11-allSGLD-rubust0.log 2>&1
# nohup bash script/gen_and_eval_vk.sh > output_MAT-1000-200-6-1-x1x1-radius11-deltaSGLD-rubust0-test3.log 2>&1