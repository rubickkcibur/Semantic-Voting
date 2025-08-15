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
export SEED=42
python src/SRLM/generate.py \
    --model_name_or_path "/mnt/${MACLAB_NAS_NAME}/rubickjiang/proj_storage/huggingface_models/Qwen2.5-1.5B-Instruct" \
    --tokenizer_path "" \
    --output_dir "/mnt/${MACLAB_NAS_NAME}/rubickjiang/codes/open-r1/data/SR_candidates" \
    --mode "chat" \
    --dataset_name "wmt24pp_zh" \
    --few_shot_cot False \
    --batch_size 4 \
    --return_sequences 256 \
    --max_model_len 2048 \
    --max_new_tokens 512 \
    --seed $SEED
exit 0
# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json