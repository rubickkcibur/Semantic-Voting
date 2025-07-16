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
python src/SRLM/cluster_score.py \
    --candidate_path "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/SR_candidates/cnn_dailymail_output_64.jsonl" \
    --output_path_scored_file "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/SR_candidates/cnn_dailymail_scored.jsonl" \
    --output_path_dpo_file "/mnt/maclabcv2/rubickjiang/codes/open-r1/data/SR_candidates/cnn_dailymail_dpo.jsonl" \
    --min_cluster_size 5 \
    --min_samples 2 \
    --filter_length 5 \
    --seed $SEED
exit 0
# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json