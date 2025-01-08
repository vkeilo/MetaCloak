
export max_train_steps=$dreambooth_training_steps
export lr=5e-7
echo set up the model path
export MODEL_PATH=$eval_model_path
echo $MODEL_PATH

export WANDB_MODE=offline
# export CLEAN_REF="$ADB_PROJECT_ROOT/dataset/$dataset_name/${instance_name}/set_C"
export CLEAN_REF="$ADB_PROJECT_ROOT/exp_datas_output_antidrm/$exp_batch_name/$exp_run_name/image_clean/"
class_name=$(cat $ADB_PROJECT_ROOT/dataset/$dataset_name/${instance_name}/class.txt)
# map class_name from face to person
if [ "$class_name" = "face" ]; then
  class_name="person"
fi

class_name_fixed=$(echo $class_name | sed "s/ /-/g")
export CLASS_DIR="$ADB_PROJECT_ROOT/prior-data/$eval_model_name/class-$class_name_fixed"
eval_prompts=$(head -n 2 $ADB_PROJECT_ROOT/dataset/$dataset_name/${instance_name}/prompts.txt | tr '\n' ';' | sed "s/@@@/$eval_prompt/g")
# export train_exp_name_prefix=$prefix_name_train
# train_exp_name=$gen_exp_name-$train_exp_name_prefix-$train_mode-eval
# train_hyper=gen-$gen_exp_name-$gen_exp_hyper-eval-$train_exp_name_prefix-rate-$poison_rate

export DREAMBOOTH_OUTPUT_DIR="$ADB_PROJECT_ROOT/exp_datas_output_antidrm/$exp_batch_name/$exp_run_name/train_output/"
echo "dreambooth output to $DREAMBOOTH_OUTPUT_DIR"
# export TEXTUAL_INVERSION_OUTPUT_DIR="$ADB_PROJECT_ROOT/exp_data-$test_timestamp/train_output/$train_exp_name/$train_hyper/${instance_name}_TEXTUAL_INVERSION"

# export DREAMBOOTH_OUTPUT_DIR="$ADB_PROJECT_ROOT/exp_data-$test_timestamp/train_output/$train_exp_name/$train_hyper/${instance_name}_DREAMBOOTH"
# export TEXTUAL_INVERSION_OUTPUT_DIR="$ADB_PROJECT_ROOT/exp_data-$test_timestamp/train_output/$train_exp_name/$train_hyper/${instance_name}_TEXTUAL_INVERSION"

cd $ADB_PROJECT_ROOT/robust_facecloak
instance_prompt="a photo of $eval_prompt $class_name"
source activate $ADB_ENV_NAME;
# vkeilo add it
conda activate Metacloak

# this is to indicate that whether we have finished the training before 
# training_finish_indicator=$DREAMBOOTH_OUTPUT_DIR/finished.txt

echo $INSTANCE_DIR
# skip training if instance data not exist 
if [ ! -d "$INSTANCE_DIR" ]; then
  echo "instance data not exist, skip training"
  exit 1
fi

# if select_model_index is -1, then nothing todo
if [ "$select_model_index" -eq -1 ]; then
    echo "Environment variable select_model_index is -1. Exited"
    exit 0
fi

echo "start dreambooth training"
if [ ${#select_model_index} -eq 1 ]; then
  command="""python train_dreambooth.py --clean_img_dir $CLEAN_INSTANCE_DIR --clean_ref_db $CLEAN_REF  --instance_name $instance_name --dataset_name $dataset_name --class_name '$class_name' \
  --wandb_entity_name $wandb_entity_name \
  --seed $seed \
  --train_text_encoder \
  --gradient_checkpointing \
  --pretrained_model_name_or_path='$MODEL_PATH'  \
  --instance_data_dir='$INSTANCE_DIR' \
  --class_data_dir='$CLASS_DIR' \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt='${instance_prompt}' \
  --class_prompt='a photo of ${class_name}' \
  --inference_prompts='${eval_prompts}' \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=$lr \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=$max_train_steps \
  --center_crop \
  --sample_batch_size=4 \
  --use_8bit_adam \
  --log_score \
  --eval_gen_img_num=$eval_gen_img_num \
  --wandb_project_name $wandb_project_name \
  --wandb_run_name $wandb_run_name \
  --poison_rate 1.0 \
  --select_model_index $select_model_index \
  --attack_mode $attack_mode 
  """

  if [ $eval_model_name = "SD21base" ] || [ $eval_model_name = "SD21" ]; then
    command="$command --enable_xformers_memory_efficient_attention --mixed_precision=bf16 --prior_generation_precision=bf16"
  else 
    command="$command --mixed_precision=bf16 --prior_generation_precision=bf16"
  fi 

  # check variable more_defense_name 
  if [ -z "$more_defense_name" ]; then
    more_defense_name="none"
  fi

  # if more_defense_name is not none, then add the flag
  if [ "$more_defense_name" != "none" ]; then
    # if more_defense_name is sr, then add the flag
    if [ "$more_defense_name" = "sr" ]; then
      command="$command --transform_sr"
    fi
    # transform_tvm
    if [ "$more_defense_name" = "tvm" ]; then
      command="$command --transform_tvm"
    fi
    # jpeg_transform
    if [ "$more_defense_name" = "jpeg" ]; then
      command="$command --jpeg_transform"
    fi
  fi

  if [ "$eval_mode" = "gau" ]; then
    command="$command --transform_defense --transform_gau --gau_kernel_size $gauK --transform_hflip"
  fi
  echo $command
  eval $command
else
  for (( i=0; i<${#select_model_index}; i++ )); do
    use_model_index="${select_model_index:i:1}"
    tmp_run_name="$wandb_run_name-model$use_model_index"
    command="""python train_dreambooth.py --clean_img_dir $CLEAN_INSTANCE_DIR --clean_ref_db $CLEAN_REF  --instance_name $instance_name --dataset_name $dataset_name --class_name '$class_name' \
    --wandb_entity_name $wandb_entity_name \
    --seed $seed \
    --train_text_encoder \
    --exp_name $train_exp_name \
    --gradient_checkpointing \
    --exp_hyper $train_hyper \
    --pretrained_model_name_or_path='$MODEL_PATH'  \
    --instance_data_dir='$INSTANCE_DIR' \
    --class_data_dir='$CLASS_DIR' \
    --output_dir=$DREAMBOOTH_OUTPUT_DIR \
    --with_prior_preservation \
    --prior_loss_weight=1.0 \
    --instance_prompt='${instance_prompt}' \
    --class_prompt='a photo of ${class_name}' \
    --inference_prompts='${eval_prompts}' \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=$lr \
    --lr_scheduler=constant \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=$max_train_steps \
    --center_crop \
    --sample_batch_size=4 \
    --use_8bit_adam \
    --log_score \
    --eval_gen_img_num=$eval_gen_img_num \
    --wandb_project_name $wandb_project_name \
    --wandb_run_name $tmp_run_name \
    --poison_rate 1.0 \
    --select_model_index $use_model_index \
    --attack_mode $attack_mode 
    """

    if [ $eval_model_name = "SD21base" ] || [ $eval_model_name = "SD21" ]; then
      command="$command --enable_xformers_memory_efficient_attention --mixed_precision=bf16 --prior_generation_precision=bf16"
    else 
      command="$command --mixed_precision=bf16 --prior_generation_precision=bf16"
    fi 

    # check variable more_defense_name 
    if [ -z "$more_defense_name" ]; then
      more_defense_name="none"
    fi

    # if more_defense_name is not none, then add the flag
    if [ "$more_defense_name" != "none" ]; then
      # if more_defense_name is sr, then add the flag
      if [ "$more_defense_name" = "sr" ]; then
        command="$command --transform_sr"
      fi
      # transform_tvm
      if [ "$more_defense_name" = "tvm" ]; then
        command="$command --transform_tvm"
      fi
      # jpeg_transform
      if [ "$more_defense_name" = "jpeg" ]; then
        command="$command --jpeg_transform"
      fi
    fi
    if [ "$train_mode" = "gau" ]; then
      command="$command --transform_defense --transform_gau --gau_kernel_size $gauK --transform_hflip"
    fi
    eval $command
  done
fi

mv $DREAMBOOTH_OUTPUT_DIR/checkpoint-$dreambooth_training_steps/dreambooth $DREAMBOOTH_OUTPUT_DIR
rm -r $DREAMBOOTH_OUTPUT_DIR/checkpoint-$dreambooth_training_steps