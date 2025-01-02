#### Variables that you should set before running the script
# setting the GPU id
# export CUDA_VISIBLE_DEVICES=0
# for logging , please set your own wandb entity name
export wandb_entity_name=vkeilo
# export wandb_project_name="metacloak_test1"
# for reproducibility
export seed=0
# for other models, change the name and path correspondingly
# SD21, SD21base, SD15, SD14
export gen_model_name=SD21base # the model to generate the noise 
# export MODEL_ROOT=$ADB_PROJECT_ROOT
# export gen_model_path=$MODEL_ROOT/SD/stable-diffusion-2-1-base,$MODEL_ROOT/SD/stable-diffusion-2-1-base
# export gen_model_path="${MODEL_ROOT}/SD/stable-diffusion-2-1-base,${MODEL_ROOT}/SD/stable-diffusion-v1-5"
# echo $gen_model_path
export eval_model_name=SD21base # the model that performs the evaluation on noise 
export eval_model_path=$MODEL_ROOT/SD/stable-diffusion-2-1-base
# export r=11 # the noise level of perturbation
export gen_prompt=sks # the prompt used to craft the noise
export eval_prompt=sks # the prompt used to do dreambooth training
export dataset_name=VGGFace2-clean # for the Cele, change it to CelebA-HQ-clean
# support method name: metacloak, clean
export method_select=metacloak
# the training mode, can be std or gau
# export train_mode=gau
# if gau is selected, the gauK is the kernel size of guassian filter 
# export gauK=7
# export eval_gen_img_num=16 # the number of images to generate per prompt 
# export round=final # which round of noise to use for evaluation
# the instance name of image, for loop all of them to measure the performance of MetaCloak on whole datasets
export instance_name=0
# this is for indexing and retriving the noise for evaluation
export prefix_name_gen=release
export prefix_name_train=gau
##############################################################


# for later usage 
export model_name=$gen_model_name
# go to main directory
pwdd=$(cd "$(dirname "$0")";pwd)
cd $pwdd 
. ./methods_config_vk_batch.sh

export gen_exp_name_prefix=$prefix_name_gen
export prefix_name_train=$prefix_name_train
export method_hyper_name=$method_name-$method_hyper
export gen_exp_name=$gen_exp_name_prefix-$method_hyper_name
export gen_exp_hyper=dataset-$dataset_name-r-$r-model-$gen_model_name-gen_prompt-$gen_prompt
export OUTPUT_DIR="$ADB_PROJECT_ROOT/exp_datas_output_antidrm/$exp_batch_name/$exp_run_name"
export INSTANCE_DIR=$OUTPUT_DIR/noise-ckpt/${round}
export CLEAN_INSTANCE_DIR=$OUTPUT_DIR/image_before_addding_noise/
export INSTANCE_DIR_CHECK=$INSTANCE_DIR

# vkeilo add it
export PYTHONPATH=$ADB_PROJECT_ROOT:$PYTHONPATH

echo "INSTANCE_DIR is set to ${INSTANCE_DIR}"

echo "PYTHONPATH:${PYTHONPATH}"
# generate the noise 
# bash ./sub/gen/$gen_file_name

# evaluate the noise with dreambooth training
bash ./sub/eval/generic_antidrm.sh



