#!/bin/bash
export PYTHONPATH=/app/data/code/MGE-LDM-main
# 在 torchrun 命令前添加
# EXPNAME=$1
# GPU=$2
# GPU=5
export http_proxy="http://10.242.26.231:7890"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
GPU="0"
NUM_GPU=1

## Load from mix-only pretrained model
# MIX_CKPT_DIR="/data2/yoongi/MGE_LDM/default_dit_mix_pretrain/checkpoints/"
# MIX_CKPT_PATH=$MIX_CKPT_DIR"unwrapped_last.ckpt"
# MIX_CKPT_PATH=null ## Can be null if you want to train from scratch

### Load from resume checkpoint
# CKPT_DIR="/data2/yoongi/MGE_LDM/default_dit/checkpoints/"
# CKPT_PATH=$CKPT_DIR"last.ckpt"

SAVE_DIR="/app/data/code/test_code/MGE-LDM-main/ckps/test"
CONFIG_NAME="dit"


# AE_CKPT_PATH=".../unwrapped_AE.ckpt"

AE_CKPT_PATH="/app/data/code/MGE-LDM-main/ckps/ae/unwrapped_AE.ckpt"
CLAP_CKPT_PATH="/app/data/code/MGE-LDM-main/ckps/clap/music_audioset_epoch_15_esc_90.14.pt"
#CKPT_PATH="/app/data/code/test_code/MGE-LDM-main/ckps/nongate_quality_aux_loss_X_HiddenState/dit/checkpoints/last.ckpt"
#CKPT_PATH="/app/data/code/test_code/MGE-LDM-main/ckps/slakh_musdb18_moisesdb/dit/checkpoints/last-v1.ckpt"
# CUDA_VISIBLE_DEVICES=$GPU \
# taskset -c 16-79 \
# python train_dit.py \
# --config-name $CONFIG_NAME \
# save_dir=$SAVE_DIR \
# autoencoder_ckpt_path=$AE_CKPT_PATH
# # # pretrained_ckpt_path=$MIX_CKPT_PATH
# # # ckpt_path=$CKPT_PATH


## Mutlti-GPU Training
CUDA_VISIBLE_DEVICES=$GPU \
torchrun --master_port 29502 --nproc_per_node gpu train_dit.py \
--config-name $CONFIG_NAME \
save_dir=$SAVE_DIR \
autoencoder_ckpt_path=$AE_CKPT_PATH \
ckpt_path=$CKPT_PATH
