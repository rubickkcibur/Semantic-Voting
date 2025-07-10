#!/bin/bash
export MACLAB_NAS_NAME="maclabcv2"
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file /mnt/maclabcv2/rubickjiang/codes/accelerate_config/config_ds3.yaml \
    src/SRLM/dpo.py --config /mnt/maclabcv2/rubickjiang/codes/open-r1/src/configs/dpo/Qwen2.5-1.5B-Instruct.yaml