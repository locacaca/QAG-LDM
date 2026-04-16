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

GPU=4
CONFIG_NAME="dit"
CKPT_DIR="/app/data/code/MGE-LDM-main/ckps/dit_epoch200/dit/checkpoints/"
CKPT_PATH=$CKPT_DIR"unwrapped_DiT_31.ckpt"

## 推断条件
TASK="partial_gen"

## 生成配置
NUM_STEPS=100
CFG_SCALE=10.0
OVERLAP_DUR=6.0
REPAINT_N=1

# 输入和输出路径
INPUT_JSONL="manifests/merged.jsonl"
TASK_CONFIG_JSONL="batch_partial_gen_tasks.jsonl"
AUDIO_OUTPUT_DIR="/app/data/code/test_code/MGE-LDM-main/batch_partial_gen_data"
MAIN_OUTPUT_DIR="./outputs_batch_partial_gen_no_quality"

# 创建输出目录
mkdir -p ${MAIN_OUTPUT_DIR}

echo "=========================================="
echo "批量部分生成开始 (无质量控制)"
echo "=========================================="
echo "本脚本将完成以下步骤:"
echo "1. ✓ 使用已处理的数据 (merged.jsonl 已处理完成)"
echo "2. ✓ 使用已混合的音频文件作为 GIVEN_WAV_PATH"
echo "3. ✓ 使用扩展文字提示: 'the sound of {乐器名}'"
echo "4. ✓ 设置离线模式，避免网络连接问题"
echo "5. ✓ 智能跳过已完成的任务 (断点续传)"
echo "6. → 随机截取模型训练长度片段进行部分生成 (约9.5秒，100个潜在帧)"
echo "7. → 批量生成 (带重试机制)"
echo ""
echo "配置信息:"
echo "输入文件: ${INPUT_JSONL}"
echo "模型检查点: ${CKPT_PATH}"
echo "生成参数: num_steps=${NUM_STEPS}, cfg_scale=${CFG_SCALE}, overlap_dur=${OVERLAP_DUR}, repaint_n=${REPAINT_N}"
echo "片段截取: 随机截取模型训练长度片段进行部分生成 (约9.5秒，100个潜在帧)"
echo "主输出目录: ${MAIN_OUTPUT_DIR}"
echo ""

# 步骤1: 处理merged.jsonl数据，生成任务配置
echo "步骤1: 处理merged.jsonl数据..."

# 检查输入文件是否存在
if [ ! -f "${INPUT_JSONL}" ]; then
    echo "错误: 输入文件 ${INPUT_JSONL} 不存在"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "batch_partial_gen_processor_no_quality.py" ]; then
    echo "错误: Python脚本 batch_partial_gen_processor_no_quality.py 不存在"
    exit 1
fi

# 创建音频输出目录
mkdir -p ${AUDIO_OUTPUT_DIR}

# 执行Python数据处理脚本
echo "正在执行Python数据处理脚本..."
python batch_partial_gen_processor_no_quality.py \
    --input ${INPUT_JSONL} \
    --output ${TASK_CONFIG_JSONL} \
    --output-dir ${AUDIO_OUTPUT_DIR} \
    --seed 42

# 检查任务配置文件是否生成成功
if [ ! -f "${TASK_CONFIG_JSONL}" ]; then
    echo "错误: 任务配置文件生成失败"
    exit 1
fi

# 统计任务数量 (使用已生成的任务配置文件)
TASK_COUNT=$(wc -l < ${TASK_CONFIG_JSONL})
echo "✓ 使用已生成的任务配置: ${TASK_COUNT} 个任务"
echo "✓ 音频文件位置: ${AUDIO_OUTPUT_DIR}"
echo ""

# 步骤2: 批量生成 (无质量控制)
echo "=========================================="
echo "开始批量生成 (无质量控制)"
echo "=========================================="

# 读取任务配置并逐个处理
TASK_INDEX=0
SKIPPED_COUNT=0
while IFS= read -r line; do
    if [ -z "$line" ]; then
        continue
    fi
    
    # 解析JSON行
    TRACK_UID=$(echo "$line" | python -c "import sys, json; data=json.load(sys.stdin); print(data['uid'])")
    GIVEN_WAV_PATH=$(echo "$line" | python -c "import sys, json; data=json.load(sys.stdin); print(data['given_wav_path'])")
    ORIGINAL_TEXT_PROMPT=$(echo "$line" | python -c "import sys, json; data=json.load(sys.stdin); print(data['text_prompt'])")
    UNKNOWN_INSTRUMENT=$(echo "$line" | python -c "import sys, json; data=json.load(sys.stdin); print(data['unknown_instrument'])")
    
    # 扩展文字提示为 "the sound of + 乐器名"
    TEXT_PROMPT="the sound of ${ORIGINAL_TEXT_PROMPT}"
    
    # 为当前任务创建输出目录
    TASK_OUTPUT_DIR="${MAIN_OUTPUT_DIR}/${TRACK_UID}"
    
    # 检查任务是否已经完成
    if [ -d "${TASK_OUTPUT_DIR}" ]; then
        # 检查是否已有生成结果
        if [ -f "${TASK_OUTPUT_DIR}/output_0001/gen_mix.wav" ] || [ -f "${TASK_OUTPUT_DIR}/output_0002/gen_mix.wav" ] || [ -f "${TASK_OUTPUT_DIR}/output_0003/gen_mix.wav" ]; then
            echo "跳过任务 ${TASK_INDEX}/${TASK_COUNT}: ${TRACK_UID} (已完成)"
            echo "  未知音轨: ${UNKNOWN_INSTRUMENT}"
            echo "  扩展提示: ${TEXT_PROMPT}"
            echo "  输出目录: ${TASK_OUTPUT_DIR}"
            echo "  ✓ 任务已完成，跳过"
            echo ""
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            continue
        fi
    fi
    
    # 创建输出目录
    mkdir -p ${TASK_OUTPUT_DIR}
    
    echo "处理任务 ${TASK_INDEX}/${TASK_COUNT}: ${TRACK_UID}"
    echo "  未知音轨: ${UNKNOWN_INSTRUMENT}"
    echo "  原始提示: ${ORIGINAL_TEXT_PROMPT}"
    echo "  扩展提示: ${TEXT_PROMPT}"
    echo "  给定音频: ${GIVEN_WAV_PATH}"
    echo "  输出目录: ${TASK_OUTPUT_DIR}"
    
    # 运行推理 (随机截取10秒片段) - 带重试机制，无质量控制
    MAX_RETRIES=3
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        echo "  尝试运行推理 (第 $((RETRY_COUNT + 1)) 次)..."
        
        if CUDA_VISIBLE_DEVICES=$GPU \
        python infer_no_quality.py \
            --config-name $CONFIG_NAME \
            --ckpt-path ${CKPT_PATH} \
            --task $TASK \
            --given-wav-path ${GIVEN_WAV_PATH} \
            --text-prompt "${TEXT_PROMPT}" \
            --output-dir ${TASK_OUTPUT_DIR} \
            --num-steps ${NUM_STEPS} \
            --cfg-scale ${CFG_SCALE} \
            --overlap-dur ${OVERLAP_DUR} \
            --repaint-n ${REPAINT_N} \
            --segment-duration 10.0 \
            --random-segment; then
            echo "  ✓ 推理成功完成"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                echo "  ✗ 推理失败，等待5秒后重试..."
                sleep 5
            else
                echo "  ✗ 推理失败，已达到最大重试次数，跳过此任务"
                continue
            fi
        fi
    done
    
    echo "  ✓ 完成: ${TRACK_UID} (无质量控制)"
    echo ""
    
    # 清理GPU缓存
    if command -v nvidia-smi &> /dev/null; then
        python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    fi
    
    TASK_INDEX=$((TASK_INDEX + 1))
    
done < ${TASK_CONFIG_JSONL}

echo "=========================================="
echo "批量部分生成完成！"
echo "=========================================="
echo "✓ 数据处理: 使用已处理的 ${TASK_COUNT} 个音轨"
echo "✓ 音频混合: 使用已混合的音频文件作为 GIVEN_WAV_PATH"
echo "✓ 文字提示: 使用扩展文字提示 'the sound of {乐器名}'"
echo "✓ 网络设置: 设置离线模式，避免网络连接问题"
echo "✓ 断点续传: 智能跳过已完成的任务"
echo "✓ 片段截取: 随机截取模型训练长度片段进行部分生成"
echo "✓ 批量生成: 完成所有任务生成 (带重试机制，无质量控制)"
echo ""
echo "输出目录: ${MAIN_OUTPUT_DIR}"
echo "处理的任务数量: ${TASK_COUNT}"
echo "跳过已完成的任务: ${SKIPPED_COUNT} 个"
echo "总共生成了 $((TASK_COUNT - SKIPPED_COUNT)) 个音频文件"
echo ""

# 创建汇总信息文件
SUMMARY_FILE="${MAIN_OUTPUT_DIR}/generation_summary.txt"
cat > ${SUMMARY_FILE} << EOF
批量部分生成汇总 (无质量控制)
=====================================

生成时间: $(date)
模型检查点: ${CKPT_PATH}
输入文件: ${INPUT_JSONL}
处理的任务数量: ${TASK_COUNT}

执行步骤:
1. ✓ 使用已处理的 merged.jsonl 数据 (预处理完成)
2. ✓ 使用已随机选择的未知音轨 (Bass/Drums/Guitar/Piano)
3. ✓ 使用已混合的音频文件作为 GIVEN_WAV_PATH
4. ✓ 使用扩展文字提示: 'the sound of {乐器名}'
5. ✓ 设置离线模式，避免网络连接问题
6. ✓ 智能跳过已完成的任务 (断点续传)
7. ✓ 随机截取模型训练长度片段进行部分生成 (约9.5秒，100个潜在帧)
8. ✓ 批量生成 (带重试机制，无质量控制)

生成参数:
- task: ${TASK}
- num_steps: ${NUM_STEPS}
- cfg_scale: ${CFG_SCALE}
- overlap_dur: ${OVERLAP_DUR}
- repaint_n: ${REPAINT_N}
- segment_duration: 10.0 (随机截取模型训练长度片段，约9.5秒，100个潜在帧)
- random_segment: true
- text_prompt: 扩展为 "the sound of {乐器名}" (如: "the sound of Bass")
- 质量控制: 禁用 (无质量控制)

输出目录结构:
${MAIN_OUTPUT_DIR}/
├── test_Track01892/
├── test_Track02029/
└── ...

每个任务目录包含:
- gen_mix.wav (生成的混合音轨)
- gen_src_*.wav (生成的各个音源)
- generation_info.txt (生成信息)

实验目的:
验证原始模型在部分生成任务中的性能，作为质量控制模型的基线对比。
不使用任何质量控制机制，生成标准质量的音频内容。

质量控制说明:
- 无质量控制: 使用模型默认生成质量
- 无质量分数参数: 不传递quality_score参数
- 标准生成: 使用原始模型的生成能力
EOF

echo "汇总信息已保存到: ${SUMMARY_FILE}"

# 显示最终统计
echo ""
echo "最终统计:"
echo "- 总任务数: ${TASK_COUNT}"
echo "- 跳过任务数: ${SKIPPED_COUNT}"
echo "- 实际生成数: $((TASK_COUNT - SKIPPED_COUNT))"
echo "- 输出目录: ${MAIN_OUTPUT_DIR}"
echo "- 汇总报告: ${SUMMARY_FILE}"
echo ""
echo "🎉 批量部分生成完成！"
echo "   现在您可以开始分析原始模型的生成效果了！"
