#!/bin/bash
#SBATCH -o /aifs4su/rubickjiang/logs/job.%j.out.log
#SBATCH --error /aifs4su/rubickjiang/logs/job.%j.err.log
#SBATCH -p batch
#SBATCH -J test_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:8
#SBATCH -c 32
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MACLAB_NAS_NAME="maclabcv2"

export TORCH_USE_CUDA_DSA=1
# export CUDA_VISIBLE_DEVICES=0
# what matters: model_name_or_path, peft_model_path, eval_data_path, per_device_eval_batch_size(fixed)
accelerate launch --config_file "/mnt/${MACLAB_NAS_NAME}/rubickjiang/codes/accelerate_config/config_acc.yaml" src/open_r1/evaluation.py \
    --model_name_or_path "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/models/Qwen2.5-1.5B-DPO-wmt24pp-ensembled-5-2" \
    --tokenizer_path "" \
    --output_dir "" \
    --mode "chat" \
    --dataset_name "wmt24pp_de" \
    --bf16 True \
    --few_shot_cot False \
    --per_device_eval_batch_size 8 \
    --model_max_length 2048 \
    --max_new_tokens 512
exit 0
# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json