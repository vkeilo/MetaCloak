
if [ "$method_select" = "metacloak" ]; then
    echo "method_select = metacloak"
    export method_name=MetaCloak
    export gen_file_name="metacloak_vk.sh"
    export advance_steps=2; 
    export total_trail_num=4; 
    # export interval=200; 
    export interval=200; 
    # export total_train_steps=1000; 
    export total_train_steps=1000; 
    export unroll_steps=1;
    # vkeilo change it from default 1 to 10
    export defense_sample_num=10; # 这个应该为1，每个step更新一次扰动
    export defense_pgd_step_num=6; # 这个是一轮扰动计算的扰动更新次数
    export sampling_times_theta=10;
    export attack_pgd_step_num=3; #这个才是一次扰动更新的采样数量
    export ref_model_path=$gen_model_path
    export method_hyper=advance_steps-$advance_steps-total_trail_num-$total_trail_num-unroll_steps-$unroll_steps-interval-$interval-total_train_steps-$total_train_steps-$model_name
    
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