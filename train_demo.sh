#!/bin/bash
export MACLAB_NAS_NAME="maclabcv2"
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file /mnt/maclabcv2/rubickjiang/codes/accelerate_config/config_ds.yaml \
    src/open_r1/grpo.py --config /mnt/maclabcv2/rubickjiang/codes/open-r1/recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml \
    --vllm_mode colocate