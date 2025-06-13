#!/bin/bash
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file /mnt/maclabcv2/rubickjiang/codes/accelerate_config/config_ds.yaml \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
    --vllm_mode colocate