#!/bin/bash
export PYTHONPATH=/app/data/code/MGE-LDM-main
export http_proxy="http://10.242.26.231:7890"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
set -e

GPU=${GPU:-4}
CONFIG_NAME="dit"
CKPT_DIR=${CKPT_DIR:-"/app/data/code/test_code/MGE-LDM-main/ckps/gate_quality_aux_loss_X_HiddenState/dit/checkpoints/"}
CKPT_PATH=$CKPT_DIR"unwrapped_DiT_31.ckpt"

TASK="source_extract"
NUM_STEPS=${NUM_STEPS:-100}
CFG_SCALE=${CFG_SCALE:-10.0}
OVERLAP_DUR=${OVERLAP_DUR:-6.0}
REPAINT_N=${REPAINT_N:-1}

if [ -z "${QUALITY_SCORES+x}" ] || [ -z "${QUALITY_SCORES}" ]; then
  QUALITY_SCORES=(0.2 0.5 0.8)
else
  QUALITY_SCORES=(${QUALITY_SCORES})
fi

TARGET_INSTRUMENTS=${TARGET_INSTRUMENTS:-"random"}

INPUT_JSONL=${INPUT_JSONL:-"manifests/merged.jsonl"}
MAIN_OUTPUT_DIR=${MAIN_OUTPUT_DIR:-"./outputs_batch_src_extract"}
RESULTS_DIR=${RESULTS_DIR:-"./results"}

mkdir -p ${MAIN_OUTPUT_DIR}
mkdir -p ${RESULTS_DIR}

if [ ! -f "${INPUT_JSONL}" ]; then
  echo "错误: 输入文件 ${INPUT_JSONL} 不存在"
  exit 1
fi
TASK_COUNT=$(wc -l < ${INPUT_JSONL})
echo "[Start] 批量源分离 | GPU=${GPU} | Qualities=${QUALITY_SCORES[*]} | Tasks=${TASK_COUNT}"

# -----------------------------------------------------------------------
# 生成阶段：只做生成，不计算指标
# -----------------------------------------------------------------------

for QUALITY_SCORE in "${QUALITY_SCORES[@]}"; do
  echo ""
  echo "[quality=${QUALITY_SCORE}]"
  QUALITY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}/quality_${QUALITY_SCORE}"
  mkdir -p ${QUALITY_OUTPUT_DIR}

  TASK_INDEX=0
  SKIPPED_COUNT=0

  while IFS= read -r line; do
    [ -z "$line" ] && continue

    TRACK_UID=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(d['uid'])")
    TRACK_UID=$(printf "%s" "$TRACK_UID" | tr -d '{}" \t\r\n' | sed 's/[^a-zA-Z0-9_-]//g')
    GIVEN_WAV_PATH=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(d['given_wav'])")

    INSTRUMENT_LIST=$(JSON_LINE="$line" USER_INSTRUMENTS="${TARGET_INSTRUMENTS}" python - <<'PY'
import json, os, random
line = os.environ.get("JSON_LINE", "").strip()
if not line:
    print(""); raise SystemExit
d = json.loads(line)
ref = list(d.get("ref_sources", {}).keys())
user = os.environ.get("USER_INSTRUMENTS", "").strip()
if user == "random":
    available = ["Bass", "Drums", "Guitar", "Piano"]
    instruments = [inst for inst in available if inst in ref]
    instruments = [random.choice(instruments)] if instruments else []
elif user:
    targets = [i.strip() for i in user.split(",") if i.strip()]
    instruments = [i for i in targets if i in ref]
else:
    instruments = [random.choice(ref)] if ref else []
print("\n".join(instruments))
PY
)

    if [ -z "${INSTRUMENT_LIST}" ]; then
      echo "[Skip] ${TRACK_UID} - 无匹配音轨"
      TASK_INDEX=$((TASK_INDEX + 1))
      continue
    fi

    mapfile -t TRACK_INSTRUMENTS <<< "${INSTRUMENT_LIST}"

    for SELECTED_INSTRUMENT in "${TRACK_INSTRUMENTS[@]}"; do
      [ -z "${SELECTED_INSTRUMENT}" ] && continue

      SAFE_INST=$(python - "$SELECTED_INSTRUMENT" <<'PY'
import re, sys
inst = sys.argv[1]
safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', inst).strip('_') or "inst"
print(safe)
PY
)
      TASK_OUTPUT_DIR="${QUALITY_OUTPUT_DIR}/${TRACK_UID}/${SAFE_INST}"

      # 断点续传
      if [ -d "${TASK_OUTPUT_DIR}/src_extract" ]; then
        EXISTING_OUT=$(find "${TASK_OUTPUT_DIR}/src_extract" -maxdepth 1 -type d \
          -name 'output_[0-9][0-9][0-9][0-9]' -print -quit 2>/dev/null)
        if [ -n "${EXISTING_OUT}" ] && [ -f "${EXISTING_OUT}/gen_src.wav" ]; then
          echo "[SkipGen] ${TRACK_UID}/${SAFE_INST} q=${QUALITY_SCORE} (exists)"
          SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
          continue
        fi
      fi

      mkdir -p ${TASK_OUTPUT_DIR}
      echo "[Gen] ${TASK_INDEX}/${TASK_COUNT} ${TRACK_UID}/${SAFE_INST} q=${QUALITY_SCORE}"

      MAX_RETRIES=3
      RETRY_COUNT=0
      while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if CUDA_VISIBLE_DEVICES=$GPU \
        python infer.py \
          --config-name $CONFIG_NAME \
          +task=$TASK \
          ckpt_path=${CKPT_PATH} \
          +given_wav_path=${GIVEN_WAV_PATH} \
          "+text_prompt='the sound of ${SELECTED_INSTRUMENT}'" \
          +segment_duration=10.0 \
          +random_segment=true \
          +quality_score=${QUALITY_SCORE} \
          +enable_attention_gating=true \
          +num_steps=${NUM_STEPS} \
          +cfg_scale=${CFG_SCALE} \
          +overlap_dur=${OVERLAP_DUR} \
          +repaint_n=${REPAINT_N} \
          +output_dir=${TASK_OUTPUT_DIR}; then
          echo "  [OK] ${TRACK_UID}/${SAFE_INST}"
          # 定位最新输出子目录
          OUT_BASE="${TASK_OUTPUT_DIR}/src_extract"
          LATEST_OUT_DIR=""
          if [ -d "${OUT_BASE}" ]; then
            LATEST_OUT_DIR=$(ls -1 "${OUT_BASE}" | grep -E '^output_[0-9]{4}$' | sort -n | tail -n1)
          fi
          if [ -z "${LATEST_OUT_DIR}" ]; then
            echo "  [Warn] 未找到输出子目录，跳过评估"
          else
            OUT_DIR_FULL="${OUT_BASE}/${LATEST_OUT_DIR}"
            if [ -f "${OUT_DIR_FULL}/gen_src.wav" ]; then
              echo "  [Eval] Mel MSE for ${TRACK_UID}/${SAFE_INST}"
              CUDA_VISIBLE_DEVICES=$GPU python tools/logmel_l1_eval.py \
                --uid "${TRACK_UID}" \
                --quality "${QUALITY_SCORE}" \
                --instrument "${SELECTED_INSTRUMENT}" \
                --out_dir "${OUT_DIR_FULL}" \
                --manifest "${INPUT_JSONL}" \
                --results_csv "${QUALITY_OUTPUT_DIR}/mel_mse_results.csv" \
                --device "cuda" || true
            else
              echo "  [Warn] 缺少 gen_src.wav，跳过评估"
            fi
          fi
          break
        else
          RETRY_COUNT=$((RETRY_COUNT + 1))
          if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "  [Retry] ${RETRY_COUNT}/${MAX_RETRIES} in 5s..."
            sleep 5
          else
            echo "  [Fail] ${TRACK_UID}/${SAFE_INST} 达到最大重试次数"
          fi
        fi
      done

      python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    done

    TASK_INDEX=$((TASK_INDEX + 1))
  done < ${INPUT_JSONL}

  echo "[DoneGen] quality=${QUALITY_SCORE}  skipped=${SKIPPED_COUNT}"
done

echo ""
echo "[DoneGen] All qualities finished."

# -----------------------------------------------------------------------
# 汇总：按 quality 统计 mel_mse_results.csv
# -----------------------------------------------------------------------

echo ""
echo "========== Mel MSE Summary =========="
for QUALITY_SCORE in "${QUALITY_SCORES[@]}"; do
  QUALITY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}/quality_${QUALITY_SCORE}"
  CSV="${QUALITY_OUTPUT_DIR}/mel_mse_results.csv"
  if [ ! -f "${CSV}" ]; then
    echo "quality=${QUALITY_SCORE}: no results"
    continue
  fi
  python - "${CSV}" "${QUALITY_SCORE}" <<'PY'
import sys, csv, numpy as np
csv_path, q = sys.argv[1], sys.argv[2]
vals = []
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            vals.append(float(row['mel_mse']))
        except (KeyError, ValueError):
            pass
if vals:
    print(f"quality={q}: mean_MSE={np.mean(vals):.6f}  "
          f"std={np.std(vals):.6f}  n={len(vals)}")
else:
    print(f"quality={q}: no valid rows")
PY
done

echo ""
echo "[Done] Output: ${MAIN_OUTPUT_DIR}"
