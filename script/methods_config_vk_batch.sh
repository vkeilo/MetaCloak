if [ "$method_select" = "metacloak" ]; then
    echo "method_select = metacloak"
    export method_name=MetaCloak
    export gen_file_name="metacloak_vk.sh"
    export ref_model_path=$gen_model_path;
    echo "ref_model_path inconfig_vk.sh:${ref_model_path}"
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

# nohup bash script/gen_and_eval_vk.sh > output_MAT-1000-200-6-6-x1x1-radius11-allSGLD-rubust0.log 2>&1
# nohup bash script/gen_and_eval_vk.sh > output_MAT-1000-200-6-1-x1x1-radius11-deltaSGLD-rubust0-test3.log 2>&1