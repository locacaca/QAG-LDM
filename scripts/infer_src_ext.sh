#!/bin/bash
export PYTHONPATH=/app/data/code/MGE-LDM-main
export http_proxy="http://10.242.101.185:7890"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

# 设置离线模式，避免网络请求
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
set -e
GPU=1
# CONFIG_NAME="default_dit"
# CONFIG_NAME="default_dit_scratch"
CONFIG_NAME="dit"
CKPT_DIR="/app/data/code/test_code/MGE-LDM-main/ckps/dit_random_weigth/dit/checkpoints/"
CKPT_PATH=$CKPT_DIR"unwrapped_DiT_31.ckpt"

OUTPUT_DIR="./outputs_infer/"

## Inference Condition
TASK="source_extract"

GIVEN_WAV_PATH="/app/data/data/slakh/wav/test/Track01876/mix.wav"
# GIVEN_WAV_PATH="data_sample/bruno_24kmagic_seg.wav"
# GIVEN_WAV_PATH="data_sample/passo_bem_solto_seg.wav"
# GIVEN_WAV_PATH="data_sample/charlie_attention_seg.wav"
# GIVEN_WAV_PATH="data_sample/charlie_wedont_seg.wav"
# GIVEN_WAV_PATH="data_sample/haruharu_seg.wav"
# GIVEN_WAV_PATH="data_sample/nell_103_seg.wav"
# GIVEN_WAV_PATH="data_sample/iu_lilac_seg.wav"
# GIVEN_WAV_PATH="data_sample/dontwannacry_seg.wav"
# GIVEN_WAV_PATH="data_sample/aot_seg.wav"
# GIVEN_WAV_PATH="data_sample/vaundy_kaiju_seg.wav"
# GIVEN_WAV_PATH="data_sample/oneokrock_dreamer_seg.wav"
# GIVEN_WAV_PATH="data_sample/born_hater_seg.wav"


# TEXT_PROMPT="The sound of vocals"
# TEXT_PROMPT="The sound of synthesizer"
# TEXT_PROMPT="Synthesizer audio"
# TEXT_PROMPT="The sound of drums"
# TEXT_PROMPT="The sound of drum beat"
# TEXT_PROMPT="The sound of bass"
# TEXT_PROMPT="The sound of bass guitar"
# TEXT_PROMPT="The sound of distorted guitar"
# TEXT_PROMPT="The sound of overdrive guitar"
# TEXT_PROMPT="The sound of acoustic guitar"
TEXT_PROMPT="The sound of piano"

## GEN / Inpaint Condition
NUM_STEPS=250
# NUM_STEPS=100
CFG_SCALE=10.0
# CFG_SCALE=8.0
# CFG_SCALE=4.0
# CFG_SCALE=1.0

OVERLAP_DUR=3.0
REPAINT_N=1



CUDA_VISIBLE_DEVICES=$GPU \
python infer.py \
    --config-name $CONFIG_NAME \
    +task=$TASK \
    ckpt_path=${CKPT_PATH} \
    +given_wav_path=${GIVEN_WAV_PATH} \
    "+text_prompt='${TEXT_PROMPT}'" \
    +num_steps=${NUM_STEPS} \
    +cfg_scale=${CFG_SCALE} \
    +overlap_dur=${OVERLAP_DUR} \
    +repaint_n=${REPAINT_N} \
    +output_dir=${OUTPUT_DIR}