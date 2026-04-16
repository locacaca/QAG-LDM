#!/bin/bash
export PYTHONPATH=/app/data/code/MGE-LDM-main
export http_proxy="http://10.242.13.204:7890"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

# 设置离线模式，避免网络请求
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
set -e

GPU=4
CONFIG_NAME="dit"
#CKPT_DIR="/app/data/code/test_code/MGE-LDM-main/ckps/dit_random_weigth/dit/checkpoints/"
CKPT_DIR="/app/data/code/test_code/MGE-LDM-main/ckps/test20260309/dit/checkpoints/"

CKPT_PATH=$CKPT_DIR"unwrapped_DiT_31.ckpt"

## 推断任务
TASK="source_extract"

## 推理参数
NUM_STEPS=100
CFG_SCALE=10.0
OVERLAP_DUR=6.0
REPAINT_N=1

# 质量控制分数
QUALITY_SCORES=(0.2 0.5 0.8)

# 指定要分离的音轨（逗号分隔）。设置为 "random" 则在 Bass,Drums,Guitar,Piano 四种中随机选择
# 为空则在每个样本中随机挑选一个已有音轨
# 例如：TARGET_INSTRUMENTS="Bass,Drums" 或 TARGET_INSTRUMENTS="random"
TARGET_INSTRUMENTS="random"
# 输入和输出路径
INPUT_JSONL="manifests/merged.jsonl"
MAIN_OUTPUT_DIR="./outputs_batch_src_extract_quality_Bregman"
RESULTS_CSV="${MAIN_OUTPUT_DIR}/Bregman_logmel_l1_results.csv"

# 创建输出目录
mkdir -p ${MAIN_OUTPUT_DIR}

# 初始化评估结果文件
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "uid,quality,instrument,output_dir,logmel_l1" > "${RESULTS_CSV}"
fi

echo "[start] 批量质量控制源分离"

# 统计任务数量
if [ ! -f "${INPUT_JSONL}" ]; then
    echo "错误: 输入文件 ${INPUT_JSONL} 不存在"
    exit 1
fi
TASK_COUNT=$(wc -l < ${INPUT_JSONL})
echo "tasks=${TASK_COUNT} ckpt=$(basename ${CKPT_PATH}) qualities=[${QUALITY_SCORES[*]}] out=${MAIN_OUTPUT_DIR}"
if [ -n "${TARGET_INSTRUMENTS}" ]; then
    echo "target_instruments=${TARGET_INSTRUMENTS}"
else
    echo "target_instruments=random_per_track"
fi

# 对每个质量控制分数进行批量生成
for QUALITY_SCORE in "${QUALITY_SCORES[@]}"; do
    echo "[quality=${QUALITY_SCORE}]"

    QUALITY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}/quality_${QUALITY_SCORE}"
    mkdir -p ${QUALITY_OUTPUT_DIR}

    TASK_INDEX=0
    SKIPPED_COUNT=0

    while IFS= read -r line; do
        if [ -z "$line" ]; then
            continue
        fi

        # 解析 JSON 行
        TRACK_UID=$(echo "$line" | python -c "import sys, json; d=json.load(sys.stdin); print(d['uid'])")
        GIVEN_WAV_PATH=$(echo "$line" | python -c "import sys, json; d=json.load(sys.stdin); print(d['given_wav'])")

        # 根据用户指定/随机策略确定需要分离的音轨集合
        INSTRUMENT_LIST=$(JSON_LINE="$line" USER_INSTRUMENTS="${TARGET_INSTRUMENTS}" python - <<'PY'
import json, os, random
line = os.environ.get("JSON_LINE", "").strip()
if not line:
    print("")
    raise SystemExit
d = json.loads(line)
ref = list(d.get("ref_sources", {}).keys())
user = os.environ.get("USER_INSTRUMENTS", "").strip()

if user == "random":
    # 从四种乐器中随机选择一种（如果样本中存在）
    available_instruments = ["Bass", "Drums", "Guitar", "Piano"]
    instruments = [inst for inst in available_instruments if inst in ref]
    if instruments:
        instruments = [random.choice(instruments)]
    else:
        instruments = []
elif user:
    targets = [inst.strip() for inst in user.split(",") if inst.strip()]
    instruments = [inst for inst in targets if inst in ref]
else:
    instruments = [random.choice(ref)] if ref else []
print("\n".join(instruments))
PY
)

        if [ -z "${INSTRUMENT_LIST}" ]; then
            if [ -n "${TARGET_INSTRUMENTS}" ]; then
                echo "跳过任务 ${TASK_INDEX}/${TASK_COUNT}: ${TRACK_UID} (无匹配的音轨: ${TARGET_INSTRUMENTS})"
            else
                echo "跳过任务 ${TASK_INDEX}/${TASK_COUNT}: ${TRACK_UID} (ref_sources 为空)"
            fi
            TASK_INDEX=$((TASK_INDEX + 1))
            continue
        fi

        mapfile -t TRACK_INSTRUMENTS <<< "${INSTRUMENT_LIST}"

        for SELECTED_INSTRUMENT in "${TRACK_INSTRUMENTS[@]}"; do
            TEXT_PROMPT="the sound of ${SELECTED_INSTRUMENT}"

            # 为不同音轨建立独立的输出目录，避免互相覆盖
            SAFE_INST=$(python - "$SELECTED_INSTRUMENT" <<'PY'
import re, sys
inst = sys.argv[1]
safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', inst).strip('_') or "inst"
print(safe)
PY
)
            TASK_OUTPUT_DIR="${QUALITY_OUTPUT_DIR}/${TRACK_UID}/${SAFE_INST}"

            # 断点续传: 如果已有任意一次输出存在则跳过（源分离检查 src_extract/output_XXXX/gen_src.wav）
            if [ -d "${TASK_OUTPUT_DIR}/src_extract" ]; then
                EXISTING_OUT=$(find "${TASK_OUTPUT_DIR}/src_extract" -maxdepth 1 -type d -name 'output_[0-9][0-9][0-9][0-9]' -print -quit 2>/dev/null)
                if [ -n "${EXISTING_OUT}" ]; then
                    echo "跳过任务 ${TASK_INDEX}/${TASK_COUNT}: ${TRACK_UID} inst=${SELECTED_INSTRUMENT} (已完成)"
                    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
                    continue
                fi
            fi

            mkdir -p ${TASK_OUTPUT_DIR}

            # 打印参考原音轨统计
            STEM_META=$(INST="${SELECTED_INSTRUMENT}" bash -c 'echo "$0" | python -c "import sys, json, os; d=json.load(sys.stdin); inst=os.environ.get(\"INST\", \"\"); stems=d.get(\"ref_sources\", {}).get(inst, []); print(str(len(stems))+\"|\"+(stems[0] if stems else \"\"))"' "$line")
            STEM_CNT=$(echo "$STEM_META" | cut -d'|' -f1)
            STEM_EXAMPLE=$(echo "$STEM_META" | cut -d'|' -f2-)
            echo "(${TASK_INDEX}/${TASK_COUNT}) uid=${TRACK_UID} inst=${SELECTED_INSTRUMENT} stems=${STEM_CNT} example=${STEM_EXAMPLE}"

            # 运行推理 - 带重试机制
            MAX_RETRIES=3
            RETRY_COUNT=0

            while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
                echo "  infer try=$((RETRY_COUNT + 1))"
                if CUDA_VISIBLE_DEVICES=$GPU \
                python infer.py \
                    --config-name $CONFIG_NAME \
                    +task=$TASK \
                    ckpt_path=${CKPT_PATH} \
                    +given_wav_path=${GIVEN_WAV_PATH} \
                    "+text_prompt='${TEXT_PROMPT}'" \
                    +segment_duration=10.0 \
                    +random_segment=true \
                    +quality_score=${QUALITY_SCORE} \
                    +num_steps=${NUM_STEPS} \
                    +cfg_scale=${CFG_SCALE} \
                    +overlap_dur=${OVERLAP_DUR} \
                    +repaint_n=${REPAINT_N} \
                    +output_dir=${TASK_OUTPUT_DIR}; then
                    echo "  ok"
                    # 定位 src_extract 下的最新输出子目录 output_XXXX
                    OUT_BASE="${TASK_OUTPUT_DIR}/src_extract"
                    LATEST_OUT_DIR=""
                    if [ -d "${OUT_BASE}" ]; then
                        LATEST_OUT_DIR=$(ls -1 ${OUT_BASE} | grep -E '^output_[0-9]{4}$' | sort -n | tail -n 1)
                    fi
                    if [ -z "${LATEST_OUT_DIR}" ]; then
                        echo "  warn: 未找到输出目录，跳过评估"
                        echo "  looked_in=${OUT_BASE} entries=$(ls -1 ${OUT_BASE} 2>/dev/null | tr '\n' ' ')"
                    else
                        OUT_DIR_FULL="${OUT_BASE}/${LATEST_OUT_DIR}"
                        GEN_SRC_PATH="${OUT_DIR_FULL}/gen_src.wav"
                        GIVEN_MIX_PATH="${OUT_DIR_FULL}/given_mix.wav"
                        if [ -f "${GEN_SRC_PATH}" ] && [ -f "${GIVEN_MIX_PATH}" ]; then
                            echo "  eval: Log-Mel L1"
                            python tools/logmel_l1_eval.py \
                              --uid "${TRACK_UID}" \
                              --quality "${QUALITY_SCORE}" \
                              --instrument "${SELECTED_INSTRUMENT}" \
                              --out_dir "${OUT_DIR_FULL}" \
                              --manifest "${INPUT_JSONL}" \
                              --results_csv "${RESULTS_CSV}"
                        else
                            echo "  warn: 缺少 gen_src.wav/given_mix.wav，跳过评估"
                        fi
                    fi
                    break
                else
                    RETRY_COUNT=$((RETRY_COUNT + 1))
                    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                        echo "  retry in 5s..."
                        sleep 5
                    else
                        echo "  fail: 达到最大重试次数，跳过"
                        break
                    fi
                fi
            done

            echo "  done uid=${TRACK_UID} inst=${SELECTED_INSTRUMENT} q=${QUALITY_SCORE}"
        done

        # 清理GPU缓存
        if command -v nvidia-smi &> /dev/null; then
            python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
        fi

        TASK_INDEX=$((TASK_INDEX + 1))

    done < ${INPUT_JSONL}

    echo "[quality=${QUALITY_SCORE}] skip=${SKIPPED_COUNT}"
done

echo "[done] input=${INPUT_JSONL} out=${MAIN_OUTPUT_DIR}"

# 统计 Log-Mel L1 结果
echo ""
echo "=========================================="
echo "Log-Mel L1 统计分析"
echo "=========================================="

if [ -f "${RESULTS_CSV}" ]; then
    echo "统计文件: ${RESULTS_CSV}"

    # 计算整体统计
    TOTAL_COUNT=$(tail -n +2 "${RESULTS_CSV}" | wc -l)
    echo "总样本数: ${TOTAL_COUNT}"

    if [ $TOTAL_COUNT -gt 0 ]; then
        # 按乐器分组统计
        echo ""
        echo "按乐器分组统计:"

        # 获取所有乐器类型
        INSTRUMENTS=$(tail -n +2 "${RESULTS_CSV}" | cut -d',' -f3 | sort | uniq)

        for inst in $INSTRUMENTS; do
            if [ "$inst" != "instrument" ]; then
                COUNT=$(grep ",${inst}," "${RESULTS_CSV}" | wc -l)
                VALUES=$(grep ",${inst}," "${RESULTS_CSV}" | cut -d',' -f5 | grep -v "^$" | grep -v "logmel_l1")
                if [ -n "$VALUES" ]; then
                    MEAN=$(echo "$VALUES" | awk '{sum+=$1; count+=1} END {if(count>0) printf "%.4f", sum/count}')
                    MIN=$(echo "$VALUES" | sort -n | head -n1)
                    MAX=$(echo "$VALUES" | sort -n | tail -n1)
                    echo "  ${inst}: 样本数=${COUNT}, 平均L1=${MEAN}, 范围=[${MIN}, ${MAX}]"
                fi
            fi
        done

        # 计算总体平均值
        echo ""
        echo "总体统计:"
        ALL_VALUES=$(tail -n +2 "${RESULTS_CSV}" | cut -d',' -f5 | grep -v "^$" | grep -v "logmel_l1")
        if [ -n "$ALL_VALUES" ]; then
            OVERALL_MEAN=$(echo "$ALL_VALUES" | awk '{sum+=$1; count+=1} END {if(count>0) printf "%.4f", sum/count}')
            OVERALL_MIN=$(echo "$ALL_VALUES" | sort -n | head -n1)
            OVERALL_MAX=$(echo "$ALL_VALUES" | sort -n | tail -n1)
            echo "  总样本数: ${TOTAL_COUNT}"
            echo "  平均 L1: ${OVERALL_MEAN}"
            echo "  最小 L1: ${OVERALL_MIN}"
            echo "  最大 L1: ${OVERALL_MAX}"

            # L1 质量评估
            if [ $(echo "${OVERALL_MEAN} < 0.1" | bc -l) -eq 1 ]; then
                QUALITY="优秀"
            elif [ $(echo "${OVERALL_MEAN} < 0.3" | bc -l) -eq 1 ]; then
                QUALITY="良好"
            elif [ $(echo "${OVERALL_MEAN} < 0.5" | bc -l) -eq 1 ]; then
                QUALITY="一般"
            else
                QUALITY="需要改进"
            fi
            echo "  质量等级: ${QUALITY} (L1 越小越好)"
        fi
    fi
else
    echo "未找到统计文件: ${RESULTS_CSV}"
fi

echo ""
echo "🎉 源分离任务完成！"
echo "   查看 ${RESULTS_CSV} 获取详细结果"


