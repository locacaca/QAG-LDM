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

GPU=${GPU:-1}
CONFIG_NAME=${CONFIG_NAME:-"dit"}
CKPT_DIR=${CKPT_DIR:-"/app/data/code/test_code/MGE-LDM-main/ckps/gate_quality_aux_loss_X_HiddenState/dit/checkpoints/"}
CKPT_PATH=${CKPT_PATH:-${CKPT_DIR}"unwrapped_DiT_31.ckpt"}

# 生成参数
TASK="partial_gen"
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

# 输入/输出路径
INPUT_JSONL=${INPUT_JSONL:-"manifests/merged.jsonl"}
TASK_CONFIG_JSONL=${TASK_CONFIG_JSONL:-"batch_partial_gen_tasks.jsonl"}
AUDIO_OUTPUT_DIR=${AUDIO_OUTPUT_DIR:-"./batch_partial_gen_data"}
MAIN_OUTPUT_DIR=${MAIN_OUTPUT_DIR:-"./outputs_batch_partial_gen_quality"}

# FAD 计算配置
COMPUTE_FAD_AFTER_GEN=${COMPUTE_FAD_AFTER_GEN:-true}
PANNS_CHECKPOINT=${PANNS_CHECKPOINT:-"/root/panns_data/Cnn14_mAP=0.431.pth"}
FAD_DEVICE="cuda:${GPU}"
FAD_RESULTS_DIR=${FAD_RESULTS_DIR:-"./results"}

mkdir -p ${MAIN_OUTPUT_DIR}
mkdir -p ${FAD_RESULTS_DIR}

echo "[Start] Batch partial-gen with FAD | GPU=${GPU} | Qualities=${QUALITY_SCORES[*]}"
echo "Output: ${MAIN_OUTPUT_DIR}"

# 根据是否指定音轨决定使用哪种模式
if [ -n "${TARGET_INSTRUMENTS}" ]; then
  USE_DYNAMIC_TASKS=true
  INPUT_FILE="${INPUT_JSONL}"
  echo "[Mode] 指定音轨模式: ${TARGET_INSTRUMENTS}"
else
  USE_DYNAMIC_TASKS=false
  INPUT_FILE="${TASK_CONFIG_JSONL}"
  echo "[Mode] 预设任务模式: ${TASK_CONFIG_JSONL}"
fi

if [ ! -f "${INPUT_FILE}" ]; then
  echo "错误: 输入文件 ${INPUT_FILE} 不存在"
  exit 1
fi
TASK_COUNT=$(wc -l < ${INPUT_FILE})
echo "Tasks: ${TASK_COUNT}"

# -----------------------------------------------------------------------
# 生成阶段：只做生成，把对应关系写入 fad_pairs.jsonl
# -----------------------------------------------------------------------

for QUALITY_SCORE in "${QUALITY_SCORES[@]}"; do
  QUALITY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}/quality_${QUALITY_SCORE}"
  mkdir -p ${QUALITY_OUTPUT_DIR}

  # 每个 quality 对应一个 pairs 文件（追加模式，支持断点续跑）
  FAD_PAIRS_JSONL="${QUALITY_OUTPUT_DIR}/fad_pairs.jsonl"

  _PROCESS_PARTIAL_GEN_TASK() {
    local OUT_DIR_VAR="$OUT_DIR"
    local TRACK_UID_VAR="$TRACK_UID"
    local GIVEN_WAV_PATH_VAR="$GIVEN_WAV_PATH"
    local TEXT_PROMPT_VAR="$TEXT_PROMPT"
    local UNKNOWN_LIST_VAR="$UNKNOWN_LIST"
    local INSTRUMENT_VAR="$INSTRUMENT_NAME"

    # 如已有 gen_src.wav 则跳过生成
    EXIST_GEN=""
    if [ -d "${OUT_DIR_VAR}" ]; then
      EXIST_GEN=$(find "${OUT_DIR_VAR}" -maxdepth 3 -type f -name "gen_src.wav" | head -n1 || true)
    fi

    if [ -n "${EXIST_GEN}" ]; then
      echo "[SkipGen] ${TRACK_UID_VAR}/${INSTRUMENT_VAR} q=${QUALITY_SCORE} (exists)"
      GEN_SRC="${EXIST_GEN}"
    else
      mkdir -p ${OUT_DIR_VAR}
      echo "[Gen] ${INDEX}/${TASK_COUNT} ${TRACK_UID_VAR}/${INSTRUMENT_VAR} q=${QUALITY_SCORE}"
      if CUDA_VISIBLE_DEVICES=$GPU \
      python infer.py \
        --config-name $CONFIG_NAME \
        +task=$TASK \
        ckpt_path=${CKPT_PATH} \
        +given_wav_path=${GIVEN_WAV_PATH_VAR} \
        "+text_prompt='${TEXT_PROMPT_VAR}'" \
        +quality_score=${QUALITY_SCORE} \
        +enable_attention_gating=true \
        +num_steps=${NUM_STEPS} \
        +cfg_scale=${CFG_SCALE} \
        +overlap_dur=${OVERLAP_DUR} \
        +repaint_n=${REPAINT_N} \
        +output_dir=${OUT_DIR_VAR} \
        +segment_duration=10.0 \
        +random_segment=true; then
        GEN_SRC=$(find "${OUT_DIR_VAR}" -maxdepth 3 -type f -name "gen_src.wav" | head -n1 || true)
        if [ -z "$GEN_SRC" ]; then
          echo "[NoSrc] ${TRACK_UID_VAR}/${INSTRUMENT_VAR} q=${QUALITY_SCORE}"
          return 0
        fi
      else
        echo "[GenFail] ${TRACK_UID_VAR}/${INSTRUMENT_VAR} q=${QUALITY_SCORE}"
        return 0
      fi
    fi

    # 把对应关系追加写入 fad_pairs.jsonl（用 Python 保证 JSON 格式正确）
    python - "${TRACK_UID_VAR}" "${INSTRUMENT_VAR}" "${QUALITY_SCORE}" \
            "${GEN_SRC}" "${FAD_PAIRS_JSONL}" ${UNKNOWN_LIST_VAR} <<'PY'
import json, sys, os
uid        = sys.argv[1]
instrument = sys.argv[2]
quality    = sys.argv[3]
gen_src    = sys.argv[4]
pairs_file = sys.argv[5]
ref_paths  = sys.argv[6:]

# 检查是否已记录（断点续跑时避免重复）
if os.path.isfile(pairs_file):
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("uid") == uid and d.get("instrument") == instrument:
                    sys.exit(0)  # 已存在，跳过
            except Exception:
                pass

entry = {"uid": uid, "instrument": instrument, "quality": quality,
         "gen_src": gen_src, "ref_paths": ref_paths}
with open(pairs_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY
    echo "[Recorded] ${TRACK_UID_VAR}/${INSTRUMENT_VAR} -> ${GEN_SRC}"
  }

  INDEX=0
  while IFS= read -r line; do
    [ -z "$line" ] && continue

    if [ "${USE_DYNAMIC_TASKS}" = "true" ]; then
      TRACK_UID=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(d['uid'])")
      TRACK_UID=$(printf "%s" "$TRACK_UID" | tr -d '{}" \t\r\n' | sed 's/[^a-zA-Z0-9_-]//g')

      INSTRUMENT_LIST=$(JSON_LINE="$line" USER_INSTRUMENTS="${TARGET_INSTRUMENTS}" python - <<'PY'
import json, os, random
line = os.environ.get("JSON_LINE", "").strip()
user_insts_str = os.environ.get("USER_INSTRUMENTS", "").strip()
try:
    d = json.loads(line)
    ref_sources = d.get("ref_sources", {})
    available = list(ref_sources.keys())
    if user_insts_str == "random":
        target_instruments = ["Bass", "Drums", "Guitar", "Piano"]
        instruments = [inst for inst in target_instruments if inst in available]
        if instruments:
            instruments = [random.choice(instruments)]
        else:
            instruments = []
    elif user_insts_str:
        user_list = [inst.strip() for inst in user_insts_str.split(",") if inst.strip()]
        instruments = [inst for inst in user_list if inst in available]
    else:
        instruments = []
    print("\n".join(instruments))
except:
    print("")
PY
)

      mapfile -t TRACK_INSTRUMENTS <<< "${INSTRUMENT_LIST}"
      if [ ${#TRACK_INSTRUMENTS[@]} -eq 0 ]; then
        echo "[Skip] ${TRACK_UID} - 未找到指定的音轨"
        continue
      fi

      for UNKNOWN_INSTRUMENT in "${TRACK_INSTRUMENTS[@]}"; do
        [ -z "${UNKNOWN_INSTRUMENT}" ] && continue

        SAFE_INST=$(python - "$UNKNOWN_INSTRUMENT" <<'PY'
import re, sys
inst = sys.argv[1]
safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', inst).strip('_') or "inst"
print(safe)
PY
)

        MIXED_AUDIO_PATH="${AUDIO_OUTPUT_DIR}/${TRACK_UID}/${SAFE_INST}/given_audio.wav"
        mkdir -p "$(dirname "${MIXED_AUDIO_PATH}")"

        if [ ! -f "${MIXED_AUDIO_PATH}" ]; then
          MIX_RESULT=$(python - "${line}" "${UNKNOWN_INSTRUMENT}" "${MIXED_AUDIO_PATH}" <<'PY'
import json, sys, subprocess, os
from pathlib import Path
line = sys.argv[1]; unknown_inst = sys.argv[2]; output_path = sys.argv[3]
try:
    d = json.loads(line)
    ref_sources = d.get("ref_sources", {})
    available = list(ref_sources.keys())
    if unknown_inst not in available:
        print("SKIP"); sys.exit(0)
    remaining_files = []
    for inst in available:
        if inst != unknown_inst:
            remaining_files.extend(ref_sources.get(inst, []))
    if not remaining_files:
        print("SKIP"); sys.exit(0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if len(remaining_files) == 1:
        subprocess.run(['cp', remaining_files[0], output_path], check=True)
    else:
        cmd = ['ffmpeg']
        for f in remaining_files:
            cmd.extend(['-i', f])
        cmd.extend(['-filter_complex', f"amix=inputs={len(remaining_files)}:duration=longest",
                    '-c:a', 'pcm_s16le', '-y', output_path])
        subprocess.run(cmd, check=True, capture_output=True)
    print("OK")
except Exception as e:
    print(f"ERROR: {e}"); sys.exit(1)
PY
) 2>&1 || MIX_RESULT="ERROR"

          if [[ "${MIX_RESULT}" == *"SKIP"* ]] || [[ "${MIX_RESULT}" == *"ERROR"* ]]; then
            echo "[Skip] ${TRACK_UID}/${UNKNOWN_INSTRUMENT}: ${MIX_RESULT}"
            continue
          fi
          if [ ! -f "${MIXED_AUDIO_PATH}" ]; then
            echo "[Error] ${TRACK_UID} - 无法创建混合音频"
            continue
          fi
        fi

        UNKNOWN_LIST=$(python - "$line" "$UNKNOWN_INSTRUMENT" <<'PY'
import json, sys
try:
    d = json.loads(sys.argv[1])
    files = d.get("ref_sources", {}).get(sys.argv[2], [])
    if not files: sys.exit(1)
    print(" ".join(files))
except: sys.exit(1)
PY
) 2>&1
        if [ -z "${UNKNOWN_LIST}" ] || [[ "${UNKNOWN_LIST}" == *"ERROR"* ]]; then
          echo "[Error] ${TRACK_UID}/${UNKNOWN_INSTRUMENT} - 文件列表获取失败"
          continue
        fi

        GIVEN_WAV_PATH="${MIXED_AUDIO_PATH}"
        TEXT_PROMPT="the sound of ${UNKNOWN_INSTRUMENT}"
        INSTRUMENT_NAME="${SAFE_INST}"
        OUT_DIR="${QUALITY_OUTPUT_DIR}/${TRACK_UID}/${SAFE_INST}"
        OUT_DIR=$(printf "%s" "$OUT_DIR" | tr -d '{}"')

        _PROCESS_PARTIAL_GEN_TASK
      done

    else
      # 预设任务模式
      TRACK_UID=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(d['uid'])")
      TRACK_UID=$(printf "%s" "$TRACK_UID" | tr -d '{}" \t\r\n' | sed 's/[^a-zA-Z0-9_-]//g')
      GIVEN_WAV_PATH=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(d['given_wav_path'])")
      UNKNOWN_LIST=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(' '.join(d['unknown_audio_files']))")
      TEXT_PROMPT="the sound of $(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(d['text_prompt'])")"
      INSTRUMENT_NAME=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(d.get('instrument','unknown'))" 2>/dev/null || echo "unknown")
      OUT_DIR="${QUALITY_OUTPUT_DIR}/${TRACK_UID}"
      OUT_DIR=$(printf "%s" "$OUT_DIR" | tr -d '{}"')

      _PROCESS_PARTIAL_GEN_TASK
    fi

    INDEX=$((INDEX+1))
  done < ${INPUT_FILE}

  echo "[DoneGen] quality=${QUALITY_SCORE}  pairs recorded: $(wc -l < ${FAD_PAIRS_JSONL} 2>/dev/null || echo 0)"
done

echo ""
echo "[DoneGen] All qualities finished."

# -----------------------------------------------------------------------
# FAD 阶段：所有生成完成后，对每个 quality 做 population-level FAD
# -----------------------------------------------------------------------

if [ "${COMPUTE_FAD_AFTER_GEN}" = "true" ]; then
  echo ""
  echo "[StartFAD] Population-level FAD computation"

  for QUALITY_SCORE in "${QUALITY_SCORES[@]}"; do
    QUALITY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}/quality_${QUALITY_SCORE}"
    FAD_PAIRS_JSONL="${QUALITY_OUTPUT_DIR}/fad_pairs.jsonl"

    if [ ! -f "${FAD_PAIRS_JSONL}" ]; then
      echo "[FADSkip] q=${QUALITY_SCORE} - fad_pairs.jsonl not found"
      continue
    fi

    PAIR_COUNT=$(wc -l < "${FAD_PAIRS_JSONL}")
    echo "[FAD] q=${QUALITY_SCORE}  pairs=${PAIR_COUNT}"

    python tools/batch_fad_from_pairs.py \
      --pairs_jsonl "${FAD_PAIRS_JSONL}" \
      --panns_checkpoint "${PANNS_CHECKPOINT}" \
      --device "${FAD_DEVICE}" \
      --batch_size 16 \
      --output_json "${FAD_RESULTS_DIR}/fad_partial_gen_q${QUALITY_SCORE}.json"

    echo "[FAD] q=${QUALITY_SCORE} done -> ${FAD_RESULTS_DIR}/fad_partial_gen_q${QUALITY_SCORE}.json"
  done

  # 汇总打印
  echo ""
  echo "========== FAD Summary =========="
  for QUALITY_SCORE in "${QUALITY_SCORES[@]}"; do
    RESULT_JSON="${FAD_RESULTS_DIR}/fad_partial_gen_q${QUALITY_SCORE}.json"
    if [ -f "${RESULT_JSON}" ]; then
      python - "${RESULT_JSON}" "${QUALITY_SCORE}" <<'PY'
import sys, json
path = sys.argv[1]; q = sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)
fad = d.get("population_fad", "N/A")
n_gen = d.get("gen_count", "?")
n_ref = d.get("ref_count", "?")
print(f"quality={q}: population_FAD={fad:.6f}  gen_n={n_gen}  ref_n={n_ref}")
PY
    fi
  done
else
  echo "[FAD] skipped (COMPUTE_FAD_AFTER_GEN=false)"
fi

echo ""
echo "[Done] Output: ${MAIN_OUTPUT_DIR}"
