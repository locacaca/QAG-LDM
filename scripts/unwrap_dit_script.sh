#!/bin/bash

GPU=${GPU:-1}
# CONFIG_NAME="default_dit_mix_pretrain"
# CONFIG_NAME="default_dit"
# CONFIG_NAME="default_dit_scratch"
CONFIG_NAME=${CONFIG_NAME:-dit}
# The checkpoint under gate_quality_* was trained with the gated trainer config.
# Override the trainer explicitly here so unwrap uses the same architecture.
TRAINER_NAME=${TRAINER_NAME:-dit}

export PYTHONPATH=/app/data/code/MGE-LDM-main
export http_proxy="http://10.242.11.163:7890"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

#CKPT_DIR="/app/data/code/test_code/MGE-LDM-main/ckps/dit_random_weigth/dit/checkpoints/"
CKPT_DIR="/app/data/code/test_code/MGE-LDM-main/ckps/gate_quality_aux_loss_X_HiddenState/dit/checkpoints/"
CKPT_PATH=$CKPT_DIR"last.ckpt"

# OUTPUT_PATH=$CKPT_DIR"unwrapped_last"
# OUTPUT_PATH=$CKPT_DIR"unwrapped_DiT"
OUTPUT_PATH=$CKPT_DIR"unwrapped_DiT_31"

CUDA_VISIBLE_DEVICES=$GPU \
python unwrap_model.py \
    --config-name $CONFIG_NAME \
    trainer=${TRAINER_NAME} \
    +type=mgeldm \
    ckpt_path=${CKPT_PATH} \
    +use_safetensors=false \
    +output_name=${OUTPUT_PATH}
