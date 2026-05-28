#!/bin/bash
export PYTHONPATH=/app/data/code/MGE-LDM-main
export http_proxy="http://10.242.26.231:7890"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

# 离线模式（如需联网可注释）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
set -e

GPU=${1:-0}

echo "Using GPU: ${GPU}"

set -e

ROOT="/app/data/data/slakh/wav"
MANIFEST_OUT="./manifests"
mkdir -p ${MANIFEST_OUT}

# python make_manifest_slakh.py --root ${ROOT} --out_dir ${MANIFEST_OUT} --splits train,validation,test

CKPT_PATH="/app/data/code/test_code/MGE-LDM-main/ckps/gate_quality_aux_loss_X_HiddenState/dit/checkpoints/unwrapped_DiT_31.ckpt"
OUTPUT_ROOT="./outputs_total_gen"
mkdir -p ${OUTPUT_ROOT}

NUM_STEPS=250
CFG_SCALE=10.0
OVERLAP_DUR=7.0
REPAINT_N=1
QUALITY_SCORE=0.9

echo "Running total_gen..."
CUDA_VISIBLE_DEVICES=${GPU} python eval_total_gen.py \
  ckpt_path=${CKPT_PATH} \
  +task=total_gen +text_prompt="'The sound of the bass, drums, guitar, and piano'" \
  +gen_audio_dur=30 +num_steps=${NUM_STEPS} +cfg_scale=${CFG_SCALE} +overlap_dur=${OVERLAP_DUR} +repaint_n=${REPAINT_N} \
  +quality_score=${QUALITY_SCORE} \
  +output_root=${OUTPUT_ROOT} \
  +eval.manifest_path=./manifests/total_gen.jsonl \
  +eval.embed_backend=mfcc +eval.compute_kid=false

echo "Finished. Outputs saved to ${OUTPUT_ROOT}"
echo "Quality score: ${QUALITY_SCORE}"
echo "Generation params: num_steps=${NUM_STEPS}, cfg_scale=${CFG_SCALE}, overlap_dur=${OVERLAP_DUR}, repaint_n=${REPAINT_N}"
