#!/bin/bash
export PYTHONPATH=/app/data/code/MGE-LDM-main
export http_proxy="http://10.242.2.130:7890"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

# 设置离线模式，避免网络请求
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
set -e

EXPNAME="default_ae" ## Set your experiment name here
DATASET=${1:-"all"}  ## Dataset to encode. Set to "all" to encode all datasets
GPU=${2:-"0"} ## GPU ID to use

CKPT_DIR="/app/data/code/MGE-LDM-main/ckps/ae/"
CKPT_PATH=$CKPT_DIR"unwrapped_AE.ckpt"

CLAP_CKPT_PATH="/app/data/code/MGE-LDM-main/ckps/clap/music_audioset_epoch_15_esc_90.14.pt"

SAVE_ROOT_DIR="/app/data/data/pre_extracted_latents/"

if [ "$DATASET" = "all" ]; then
    ## You should set valid paths to your dataset in pre_encode.py
    echo "No dataset specified. Using all datasets"
    DATASET_LIST=(
        # slakh2100
        musdb18hq 
        moisesdb 
        # medleydbV2
        # mtgjamendo 
        # other_tracks
        )
else
    echo "Using specified dataset list: $DATASET"
    IFS=',' read -r -a DATASET_LIST <<< "$DATASET"
    echo "Extracting from datasets: ${DATASET_LIST[*]}"
fi

echo "Number of datasets: ${#DATASET_LIST[@]}"
# exit 0

JOINED=$(IFS=,; echo "${DATASET_LIST[*]}")    # "slakh2100,musdb18hq,moisesdb,..."
LIST_OVERRIDE="[$JOINED]"

# 如果用户已经在外部设置了 CUDA_VISIBLE_DEVICES，则使用它
# 否则使用传入的 GPU 参数设置 CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU
    echo "Setting CUDA_VISIBLE_DEVICES=$GPU"
else
    echo "Using existing CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

python pre_encode.py \
    --config-name $EXPNAME \
    +clap_ckpt_path=${CLAP_CKPT_PATH} \
    ckpt_path=${CKPT_PATH} \
    +save_root_dir=${SAVE_ROOT_DIR} \
    +extract_dataset=$LIST_OVERRIDE