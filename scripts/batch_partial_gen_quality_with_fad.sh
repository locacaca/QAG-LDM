#!/bin/bash
export PYTHONPATH=/app/data/code/MGE-LDM-main
export http_proxy="http://10.242.13.204:7890"
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
#CKPT_DIR=${CKPT_DIR:-"/app/data/code/test_code/MGE-LDM-main/ckps/dit_random_weigth/dit/checkpoints/"}
CKPT_DIR=${CKPT_DIR:-"/app/data/code/test_code/MGE-LDM-main/ckps/test20260309/dit/checkpoints/"}

CKPT_PATH=${CKPT_PATH:-${CKPT_DIR}"unwrapped_DiT_31.ckpt"}

# 生成参数
TASK="partial_gen"
NUM_STEPS=${NUM_STEPS:-100}
CFG_SCALE=${CFG_SCALE:-10.0}
OVERLAP_DUR=${OVERLAP_DUR:-6.0}
REPAINT_N=${REPAINT_N:-1}

# 质量控制分数（与生成脚本保持一致）
# - 若未传入环境变量 QUALITY_SCORES，则默认使用 0.8
# - 若通过环境变量传入，例如：export QUALITY_SCORES="0.1 0.9"
if [ -z "${QUALITY_SCORES+x}" ] || [ -z "${QUALITY_SCORES}" ]; then
  QUALITY_SCORES=(0.2 0.5 0.8)
else
  QUALITY_SCORES=(${QUALITY_SCORES})
fi

# 指定要生成的部分生成音轨（逗号分隔）。设置为 "random" 则在 Bass,Drums,Guitar,Piano 四种中随机选择
# 为空则使用 TASK_CONFIG_JSONL 中的预设任务
# 例如：TARGET_INSTRUMENTS="Bass,Drums" 或 TARGET_INSTRUMENTS="random"
# 如果指定了音轨，将从 INPUT_JSONL 读取原始数据并动态生成任务
TARGET_INSTRUMENTS=${TARGET_INSTRUMENTS:-"random"}

# 输入/输出路径
INPUT_JSONL=${INPUT_JSONL:-"manifests/merged.jsonl"}
TASK_CONFIG_JSONL=${TASK_CONFIG_JSONL:-"batch_partial_gen_tasks.jsonl"}
AUDIO_OUTPUT_DIR=${AUDIO_OUTPUT_DIR:-"./batch_partial_gen_data"}
MAIN_OUTPUT_DIR=${MAIN_OUTPUT_DIR:-"./outputs_batch_partial_gen_quality"}

# FAD 计算配置
COMPUTE_FAD_AFTER_GEN=${COMPUTE_FAD_AFTER_GEN:-true}
PANNS_CHECKPOINT=${PANNS_CHECKPOINT:-"/root/panns_data/Cnn14_mAP=0.431.pth"}
FAD_DEVICE="cuda:${GPU}"  # FAD 使用与推理相同的 GPU
FAD_SR=${FAD_SR:-48000}
FAD_SUMMARY_PATH=${FAD_SUMMARY_PATH:-"${MAIN_OUTPUT_DIR}/fad_summary.json"}

mkdir -p ${MAIN_OUTPUT_DIR}

echo "[Start] Batch partial-gen with FAD | GPU=${GPU} | Qualities=${QUALITY_SCORES[*]}"
echo "Output: ${MAIN_OUTPUT_DIR}"
echo "静音检查: ${ENABLE_SILENCE_CHECK}"

# 音频质量检测参数（在计算FAD前应用于 gen_src.wav）
# 是否启用静音检查（设置为 false/true 可禁用静音检查，直接进行FAD计算）
ENABLE_SILENCE_CHECK=${ENABLE_SILENCE_CHECK:-false}
SILENCE_THRESHOLD=0.01
MIN_SILENCE_DUR=0.5
MAX_SILENCE_RATIO=0.4
MIN_ENERGY_RATIO=0.1

# 根据是否指定音轨决定使用哪种模式
if [ -n "${TARGET_INSTRUMENTS}" ]; then
  # 模式1: 从原始数据读取并动态生成任务（指定音轨）
  USE_DYNAMIC_TASKS=true
  INPUT_FILE="${INPUT_JSONL}"
  echo "[Mode] 指定音轨模式: ${TARGET_INSTRUMENTS}"
else
  # 模式2: 使用预设任务配置文件（原有模式）
  USE_DYNAMIC_TASKS=false
  INPUT_FILE="${TASK_CONFIG_JSONL}"
  echo "[Mode] 预设任务模式: ${TASK_CONFIG_JSONL}"
fi

# 统计任务数量
if [ ! -f "${INPUT_FILE}" ]; then
  echo "错误: 输入文件 ${INPUT_FILE} 不存在"
  exit 1
fi
TASK_COUNT=$(wc -l < ${INPUT_FILE})
echo "Tasks: ${TASK_COUNT}"

for QUALITY_SCORE in "${QUALITY_SCORES[@]}"; do
  QUALITY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}/quality_${QUALITY_SCORE}"
  mkdir -p ${QUALITY_OUTPUT_DIR}

  # 定义任务处理函数
  _PROCESS_PARTIAL_GEN_TASK() {
    local OUT_DIR_VAR="$OUT_DIR"
    local TRACK_UID_VAR="$TRACK_UID"
    local GIVEN_WAV_PATH_VAR="$GIVEN_WAV_PATH"
    local TEXT_PROMPT_VAR="$TEXT_PROMPT"
    local UNKNOWN_LIST_VAR="$UNKNOWN_LIST"

    # 如已有 gen_src.wav 则跳过生成（兼容嵌套子目录，如 partial_gen_single/output_0001）
    EXIST_MIX=""
    if [ -d "${OUT_DIR_VAR}" ]; then
      EXIST_MIX=$(find "${OUT_DIR_VAR}" -maxdepth 3 -type f -name "gen_src.wav" | head -n1 || true)
    fi
    if [ -n "${EXIST_MIX}" ]; then
      echo "[SkipGen] ${TRACK_UID_VAR} q=${QUALITY_SCORE} (exists)"
      # 仍然尝试输出/计算该任务的 FAD
      GEN_MIX="${EXIST_MIX}"
      if [ -n "$GEN_MIX" ]; then
        # 质量检测：静音/能量检查（不合格则跳过FAD）
        if [ "${ENABLE_SILENCE_CHECK}" = "true" ]; then
          CHECK_RET=$(python - "${GEN_MIX}" ${SILENCE_THRESHOLD} ${MIN_SILENCE_DUR} ${MAX_SILENCE_RATIO} ${FAD_SR} << 'PY'
import sys, soundfile as sf, numpy as np
path = sys.argv[1]
sil_th = float(sys.argv[2]); min_sil_dur = float(sys.argv[3]); max_sil_ratio = float(sys.argv[4]); sr = int(sys.argv[5])
x, file_sr = sf.read(path)
if x.ndim > 1:
    x = x.mean(axis=1)
x = x.astype('float32')
abs_x = np.abs(x)
# 静音比例（考虑最小静音时长）
win = max(1, int(min_sil_dur * (file_sr if file_sr>0 else sr)))
if len(x) >= win:
    from numpy.lib.stride_tricks import sliding_window_view as swv
    sil = (abs_x < sil_th).astype(np.uint8)
    sw = swv(sil, win)
    long_sil = (sw.sum(axis=1) == win).astype(np.uint8)
    sil_ratio = long_sil.mean()
else:
    sil_ratio = float((abs_x < sil_th).mean())
ok = (sil_ratio <= max_sil_ratio)
print('OK' if ok else f'BAD sil={sil_ratio:.2f}')
PY
)
          if [[ "${CHECK_RET}" != OK* ]]; then
            echo "[AudioBad] ${TRACK_UID_VAR} q=${QUALITY_SCORE} ${CHECK_RET}"
            return 0
          fi
        else
          echo "[SkipCheck] 静音检查已禁用，直接进行FAD计算"
        fi
        if [ ! -f "${OUT_DIR_VAR}/fad_result.json" ]; then
          mkdir -p "${OUT_DIR_VAR}"
          python compute_fad_partial_gen.py \
            --gen_mix_path "$GEN_MIX" \
            --unknown_audio_files ${UNKNOWN_LIST_VAR} \
            --sample_rate ${FAD_SR} \
            --panns_checkpoint ${PANNS_CHECKPOINT} \
            --device ${FAD_DEVICE} \
            --output_json "${OUT_DIR_VAR}/fad_result.json" || true
        fi
        if [ -f "${OUT_DIR_VAR}/fad_result.json" ]; then
          FAD_JSON_PATH="${OUT_DIR_VAR}/fad_result.json"
          FAD_JSON_PATH=$(printf "%s" "$FAD_JSON_PATH" | tr -d '{}"')
          FAD_VAL=$(python - "$FAD_JSON_PATH" << 'PY'
import sys, json, os
path = sys.argv[1]
if not os.path.isfile(path):
    print('')
    sys.exit(0)
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)
print(d.get('results',{}).get('FAD',''))
PY
)
          echo "[FAD] ${TRACK_UID_VAR} q=${QUALITY_SCORE} = ${FAD_VAL}"
          echo "$FAD_VAL" >> "${QUALITY_OUTPUT_DIR}/fad_values.txt"
        else
          echo "[FADSkip] ${TRACK_UID_VAR} q=${QUALITY_SCORE} no fad_result.json"
        fi
      else
        echo "[NoSrc] ${TRACK_UID_VAR} q=${QUALITY_SCORE}"
      fi
    else
      mkdir -p ${OUT_DIR_VAR}
      echo "[Gen] ${INDEX}/${TASK_COUNT} ${TRACK_UID_VAR} q=${QUALITY_SCORE}"
      if CUDA_VISIBLE_DEVICES=$GPU \
      python infer.py \
        --config-name $CONFIG_NAME \
        +task=$TASK \
        ckpt_path=${CKPT_PATH} \
        +given_wav_path=${GIVEN_WAV_PATH_VAR} \
        "+text_prompt='${TEXT_PROMPT_VAR}'" \
        +quality_score=${QUALITY_SCORE} \
        +num_steps=${NUM_STEPS} \
        +cfg_scale=${CFG_SCALE} \
        +overlap_dur=${OVERLAP_DUR} \
        +repaint_n=${REPAINT_N} \
        +output_dir=${OUT_DIR_VAR} \
        +segment_duration=10.0 \
        +random_segment=true; then
        # 定位 gen_src 路径（兼容嵌套子目录）
        GEN_MIX=$(find "${OUT_DIR_VAR}" -maxdepth 3 -type f -name "gen_src.wav" | head -n1 || true)
        if [ -n "$GEN_MIX" ]; then
          # 质量检测
          if [ "${ENABLE_SILENCE_CHECK}" = "true" ]; then
            CHECK_RET=$(python - "${GEN_MIX}" ${SILENCE_THRESHOLD} ${MIN_SILENCE_DUR} ${MAX_SILENCE_RATIO} ${FAD_SR} << 'PY'
import sys, soundfile as sf, numpy as np
path = sys.argv[1]
sil_th = float(sys.argv[2]); min_sil_dur = float(sys.argv[3]); max_sil_ratio = float(sys.argv[4]); sr = int(sys.argv[5])
x, file_sr = sf.read(path)
if x.ndim > 1:
    x = x.mean(axis=1)
x = x.astype('float32')
abs_x = np.abs(x)
# 静音比例（考虑最小静音时长）
win = max(1, int(min_sil_dur * (file_sr if file_sr>0 else sr)))
if len(x) >= win:
    from numpy.lib.stride_tricks import sliding_window_view as swv
    sil = (abs_x < sil_th).astype(np.uint8)
    sw = swv(sil, win)
    long_sil = (sw.sum(axis=1) == win).astype(np.uint8)
    sil_ratio = long_sil.mean()
else:
    sil_ratio = float((abs_x < sil_th).mean())
ok = (sil_ratio <= max_sil_ratio)
print('OK' if ok else f'BAD sil={sil_ratio:.2f}')
PY
)
            if [[ "${CHECK_RET}" != OK* ]]; then
              echo "[AudioBad] ${TRACK_UID_VAR} q=${QUALITY_SCORE} ${CHECK_RET}"
              return 0
            fi
          else
            echo "[SkipCheck] 静音检查已禁用，直接进行FAD计算"
          fi
          # 计算并输出当前任务的 FAD
          python compute_fad_partial_gen.py \
            --gen_mix_path "$GEN_MIX" \
            --unknown_audio_files ${UNKNOWN_LIST_VAR} \
            --sample_rate ${FAD_SR} \
            --panns_checkpoint ${PANNS_CHECKPOINT} \
            --device ${FAD_DEVICE} \
            --output_json "${OUT_DIR_VAR}/fad_result.json" || true
          if [ -f "${OUT_DIR_VAR}/fad_result.json" ]; then
            FAD_JSON_PATH="${OUT_DIR_VAR}/fad_result.json"
            FAD_JSON_PATH=$(printf "%s" "$FAD_JSON_PATH" | tr -d '{}"')
            FAD_VAL=$(python - "$FAD_JSON_PATH" << 'PY'
import sys, json, os
path = sys.argv[1]
if not os.path.isfile(path):
    print('')
    sys.exit(0)
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)
print(d.get('results',{}).get('FAD',''))
PY
)
            echo "[FAD] ${TRACK_UID_VAR} q=${QUALITY_SCORE} = ${FAD_VAL}"
            echo "$FAD_VAL" >> "${QUALITY_OUTPUT_DIR}/fad_values.txt"
          else
            echo "[FADSkip] ${TRACK_UID_VAR} q=${QUALITY_SCORE} no fad_result.json"
          fi
        else
          echo "[NoSrc] ${TRACK_UID_VAR} q=${QUALITY_SCORE}"
        fi
      else
        echo "[GenFail] ${TRACK_UID_VAR} q=${QUALITY_SCORE}"
      fi
    fi
  }

  INDEX=0
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    
    if [ "${USE_DYNAMIC_TASKS}" = "true" ]; then
      # 动态任务模式：从 merged.jsonl 读取并处理指定音轨
      TRACK_UID=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(d['uid'])")
      TRACK_UID=$(printf "%s" "$TRACK_UID" | tr -d '{}" \t\r\n' | sed 's/[^a-zA-Z0-9_-]//g')
      REF_SOURCES=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);import json as j;print(j.dumps(d.get('ref_sources',{})))")
      
      # 获取可用的音轨并筛选指定的音轨
      INSTRUMENT_LIST=$(JSON_LINE="$line" USER_INSTRUMENTS="${TARGET_INSTRUMENTS}" python - <<'PY'
import json, os, random
line = os.environ.get("JSON_LINE", "").strip()
user_insts_str = os.environ.get("USER_INSTRUMENTS", "").strip()
try:
    d = json.loads(line)
    ref_sources = d.get("ref_sources", {})
    available = list(ref_sources.keys())

    if user_insts_str == "random":
        # 从四种乐器中随机选择一种（如果样本中存在）
        target_instruments = ["Bass", "Drums", "Guitar", "Piano"]
        instruments = [inst for inst in target_instruments if inst in available]
        if instruments:
            instruments = [random.choice(instruments)]
        else:
            instruments = []  # 没有目标乐器
    elif user_insts_str:
        # 解析用户指定的音轨（逗号分隔）
        user_list = [inst.strip() for inst in user_insts_str.split(",") if inst.strip()]
        # 筛选出在可用音轨中的指定音轨
        instruments = [inst for inst in user_list if inst in available]
        if not instruments:
            instruments = []  # 没有匹配的音轨
    else:
        instruments = []

    print("\n".join(instruments))
except:
    print("")
PY
)
      
      mapfile -t TRACK_INSTRUMENTS <<< "${INSTRUMENT_LIST}"
      
      if [ ${#TRACK_INSTRUMENTS[@]} -eq 0 ]; then
        echo "[Skip] ${TRACK_UID} - 未找到指定的音轨或音轨列表为空"
        continue
      fi
      
      # 为每个指定音轨创建任务
      for UNKNOWN_INSTRUMENT in "${TRACK_INSTRUMENTS[@]}"; do
        # 清理音轨名称
        SAFE_INST=$(python - "$UNKNOWN_INSTRUMENT" <<'PY'
import re, sys
inst = sys.argv[1]
safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', inst).strip('_') or "inst"
print(safe)
PY
)
        
        # 获取剩余音轨并混合
        MIXED_AUDIO_PATH="${AUDIO_OUTPUT_DIR}/${TRACK_UID}/${SAFE_INST}/given_audio.wav"
        mkdir -p "$(dirname "${MIXED_AUDIO_PATH}")"
        
        # 检查是否已存在混合音频，如果不存在则创建
        if [ ! -f "${MIXED_AUDIO_PATH}" ]; then
          MIX_RESULT=$(python - "${line}" "${UNKNOWN_INSTRUMENT}" "${MIXED_AUDIO_PATH}" <<'PY'
import json, sys, subprocess, os
from pathlib import Path

line = sys.argv[1]
unknown_inst = sys.argv[2]
output_path = sys.argv[3]

try:
    d = json.loads(line)
    ref_sources = d.get("ref_sources", {})
    available = list(ref_sources.keys())

    if unknown_inst not in available:
        print(f"SKIP: 音轨 '{unknown_inst}' 不存在于当前样本", file=sys.stderr)
        print("SKIP", file=sys.stdout)
        sys.exit(0)  # 正常退出但标记为跳过

    # 获取剩余音轨的音频文件
    remaining_files = []
    for inst in available:
        if inst != unknown_inst:
            remaining_files.extend(ref_sources.get(inst, []))

    if not remaining_files:
        print(f"SKIP: 没有剩余音轨可以混合（样本可能只有 '{unknown_inst}' 一个音轨）", file=sys.stderr)
        print("SKIP", file=sys.stdout)
        sys.exit(0)  # 正常退出但标记为跳过

    # 使用 ffmpeg 混合音频
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if len(remaining_files) == 1:
        subprocess.run(['cp', remaining_files[0], output_path], check=True)
    else:
        cmd = ['ffmpeg']
        for f in remaining_files:
            cmd.extend(['-i', f])
        filter_complex = f"amix=inputs={len(remaining_files)}:duration=longest"
        cmd.extend(['-filter_complex', filter_complex])
        cmd.extend(['-c:a', 'pcm_s16le', '-y', output_path])
        subprocess.run(cmd, check=True, capture_output=True)

    print("OK", file=sys.stdout)
    sys.exit(0)
except Exception as e:
    print(f"ERROR: 处理音轨 '{unknown_inst}' 时出错: {e}", file=sys.stderr)
    print("ERROR", file=sys.stdout)
    sys.exit(1)
PY
) 2>&1 || MIX_RESULT="ERROR"
          
          # 检查混合结果
          if [[ "${MIX_RESULT}" == *"SKIP"* ]] || [[ "${MIX_RESULT}" == *"ERROR"* ]]; then
            if [[ "${MIX_RESULT}" == *"SKIP"* ]]; then
              echo "[Skip] ${TRACK_UID} - 音轨 '${UNKNOWN_INSTRUMENT}' 跳过: ${MIX_RESULT}"
            else
              echo "[Error] ${TRACK_UID} - 音轨 '${UNKNOWN_INSTRUMENT}' 处理失败: ${MIX_RESULT}"
            fi
            continue
          fi
          
          # 验证混合音频文件是否成功创建
          if [ ! -f "${MIXED_AUDIO_PATH}" ]; then
            echo "[Error] ${TRACK_UID} - 无法创建混合音频: ${MIXED_AUDIO_PATH}"
            continue
          fi
        fi
        
        GIVEN_WAV_PATH="${MIXED_AUDIO_PATH}"
        PROMPT="${UNKNOWN_INSTRUMENT}"
        TEXT_PROMPT="the sound of ${UNKNOWN_INSTRUMENT}"
        
        # 获取未知音轨的音频文件列表
        UNKNOWN_LIST=$(python - "$line" "$UNKNOWN_INSTRUMENT" <<'PY'
import json, sys
try:
    line = sys.argv[1]
    unknown_inst = sys.argv[2]
    d = json.loads(line)
    ref_sources = d.get("ref_sources", {})
    unknown_files = ref_sources.get(unknown_inst, [])
    if not unknown_files:
        print("ERROR: 音轨文件列表为空", file=sys.stderr)
        sys.exit(1)
    print(" ".join(unknown_files))
except Exception as e:
    print(f"ERROR: 获取音轨文件列表失败: {e}", file=sys.stderr)
    sys.exit(1)
PY
) 2>&1
        
        # 检查是否成功获取文件列表
        if [ -z "${UNKNOWN_LIST}" ] || [[ "${UNKNOWN_LIST}" == *"ERROR"* ]]; then
          echo "[Error] ${TRACK_UID} - 音轨 '${UNKNOWN_INSTRUMENT}' 的文件列表为空或获取失败，跳过"
          continue
        fi
        
        # 创建输出目录（按音轨分组）
        OUT_DIR="${QUALITY_OUTPUT_DIR}/${TRACK_UID}/${SAFE_INST}"
        OUT_DIR=$(printf "%s" "$OUT_DIR" | tr -d '{}"')
        
        # 处理单个音轨任务
        _PROCESS_PARTIAL_GEN_TASK
        
      done
    else
      # 原有模式：从任务配置文件读取
      TRACK_UID=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(d['uid'])")
      TRACK_UID=$(printf "%s" "$TRACK_UID" | tr -d '{}" \t\r\n' | sed 's/[^a-zA-Z0-9_-]//g')
      GIVEN_WAV_PATH=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(d['given_wav_path'])")
      PROMPT=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(d['text_prompt'])")
      UNKNOWN_LIST=$(echo "$line" | python -c "import sys,json;d=json.load(sys.stdin);print(' '.join(d['unknown_audio_files']))")
      TEXT_PROMPT="the sound of ${PROMPT}"
      OUT_DIR="${QUALITY_OUTPUT_DIR}/${TRACK_UID}"
      OUT_DIR=$(printf "%s" "$OUT_DIR" | tr -d '{}"')
      
      # 处理预设任务
      _PROCESS_PARTIAL_GEN_TASK
    fi
    
    INDEX=$((INDEX+1))
  done < ${INPUT_FILE}

  # 该质量分数下的 FAD 均值
  if [ -f "${QUALITY_OUTPUT_DIR}/fad_values.txt" ]; then
    MEAN_FAD=$(awk '{s+=$1; n+=1} END {if(n>0) printf("%f", s/n);}' "${QUALITY_OUTPUT_DIR}/fad_values.txt")
    echo "[MeanFAD] q=${QUALITY_SCORE} -> ${MEAN_FAD}"
  fi
done

echo "[DoneGen]"

if [ "${COMPUTE_FAD_AFTER_GEN}" = "true" ] && [ "${USE_DYNAMIC_TASKS}" != "true" ]; then
  # 批量FAD计算仅适用于预设任务模式（使用任务配置文件）
  echo "[StartBulkFAD]"
  python batch_compute_fad_partial_gen.py \
    --tasks_jsonl ${TASK_CONFIG_JSONL} \
    --output_root ${MAIN_OUTPUT_DIR} \
    --quality_scores ${QUALITY_SCORES[@]} \
    --sample_rate ${FAD_SR} \
    --panns_checkpoint ${PANNS_CHECKPOINT} \
    --device ${FAD_DEVICE} \
    --output_summary ${FAD_SUMMARY_PATH} \
    --skip_existing
  echo "[BulkFAD] summary: ${FAD_SUMMARY_PATH}"
else
  if [ "${USE_DYNAMIC_TASKS}" = "true" ]; then
    echo "[BulkFAD] skipped (指定音轨模式下，FAD已在生成时计算)"
else
  echo "[BulkFAD] skipped"
  fi
fi

echo "[Done] Output: ${MAIN_OUTPUT_DIR}"


