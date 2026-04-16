#!/bin/bash
#
# 高质量评估脚本 - 支持可选质量控制
# 使用方法：
#   ./scripts/eval_all_quality.sh [GPU编号] [质量控制开关]
#
# 示例：
#   ./scripts/eval_all_quality.sh 0 true   # 使用GPU 0，启用质量控制（默认）
#   ./scripts/eval_all_quality.sh 0 false  # 使用GPU 0，禁用质量控制
#   ./scripts/eval_all_quality.sh 1        # 使用GPU 1，启用质量控制
#
export PYTHONPATH=/app/data/code/code/MGE-LDM-main
export http_proxy="http://10.242.13.204:7890"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

# GPU选择，默认使用GPU 0
GPU=${1:-1}

# 质量控制开关，第二个参数控制是否启用质量控制
# 使用方法: ./scripts/eval_all_quality.sh [GPU编号] [质量控制开关]
# 示例: ./scripts/eval_all_quality.sh 0 true   # 使用GPU 0，启用质量控制
#       ./scripts/eval_all_quality.sh 0 false  # 使用GPU 0，禁用质量控制
ENABLE_QUALITY_CONTROL=${2:-true}

echo "使用GPU: ${GPU}"
echo "质量控制: ${ENABLE_QUALITY_CONTROL}"

set -e

ROOT="/app/data/data/slakh/wav"
MANIFEST_OUT="./manifests"
mkdir -p ${MANIFEST_OUT}

#echo "生成 manifests..."
#python make_manifest_slakh.py --root ${ROOT} --out_dir ${MANIFEST_OUT} --splits train,validation,test

CKPT_PATH="/app/data/code/test_code/MGE-LDM-main/ckps/test20260309/dit/checkpoints/unwrapped_DiT_31.ckpt"

# 根据质量控制开关设置输出目录
if [ "${ENABLE_QUALITY_CONTROL}" = "true" ]; then
    OUTPUT_ROOT="./outputs_eval_quality_with_control_se"
else
    OUTPUT_ROOT="./outputs_eval_quality_no_control_se"
fi
mkdir -p ${OUTPUT_ROOT}

# 基本参数（可按需修改）
NUM_STEPS=250
CFG_SCALE=10.0
OVERLAP_DUR=7.0
REPAINT_N=1

# 质量控制参数
QUALITY_SCORE=0.2
ENABLE_QUALITY_CONTROL=true  # 设置为false可禁用质量控制

# 1) total generation with quality control
echo "运行 total_gen (质量控制版本)..."
CUDA_VISIBLE_DEVICES=${GPU} python eval_total_gen.py \
  ckpt_path=${CKPT_PATH} \
  +task=total_gen +text_prompt="'The sound of the bass, drums, guitar, and piano'" \
  +gen_audio_dur=30 +num_steps=${NUM_STEPS} +cfg_scale=${CFG_SCALE} +overlap_dur=${OVERLAP_DUR} +repaint_n=${REPAINT_N} \
  +quality_score=${QUALITY_SCORE} \
  +enable_quality_control=${ENABLE_QUALITY_CONTROL} \
  +output_root=${OUTPUT_ROOT} \
  +eval.manifest_path=./manifests/total_gen.jsonl \
  +eval.embed_backend=mfcc +eval.compute_kid=false

## 2) partial generation with quality control
#echo "运行 partial_gen (质量控制版本)..."
#python eval_total_gen.py \
#  ckpt_path=${CKPT_PATH} \
#  +task=partial_gen +text_prompt="bass" \
#  +gen_audio_dur=30.0 +num_steps=${NUM_STEPS} +cfg_scale=${CFG_SCALE} +overlap_dur=${OVERLAP_DUR} +repaint_n=${REPAINT_N} \
#  +quality_score=${QUALITY_SCORE} \
#  +output_root=${OUTPUT_ROOT} \
#  +eval.manifest_path=${MANIFEST_OUT}/partial_gen.jsonl \
#  +eval.embed_backend=mfcc +eval.compute_kid=false
#

## 3) source extraction with quality control
#echo "运行 source_extract (质量控制版本)..."
#MANIFEST_FILE=${MANIFEST_OUT}/source_extract_merged.jsonl
#
#if [ -f "$MANIFEST_FILE" ]; then
#  for inst in Piano; do
#    echo "  -> 处理乐器: $inst"
#    python eval_source_extract.py \
#      ckpt_path=${CKPT_PATH} \
#      +task=source_extract +text_prompt="${inst}" \
#      +num_steps=${NUM_STEPS} +cfg_scale=${CFG_SCALE} \
#      +overlap_dur=${OVERLAP_DUR} +repaint_n=${REPAINT_N} \
#      +quality_score=${QUALITY_SCORE} \
#      +output_root=${OUTPUT_ROOT} \
#      +eval.manifest_path=${MANIFEST_FILE}
#  done
#else
#  echo "  [跳过] 未找到 $MANIFEST_FILE"
#fi
#

echo "评估任务完成。输出保存在 ${OUTPUT_ROOT}"
echo "质量控制状态: ${ENABLE_QUALITY_CONTROL}"
echo "使用的质量分数: ${QUALITY_SCORE}"
echo "使用的生成参数: num_steps=${NUM_STEPS}, cfg_scale=${CFG_SCALE}, overlap_dur=${OVERLAP_DUR}, repaint_n=${REPAINT_N}"