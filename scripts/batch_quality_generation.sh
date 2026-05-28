#!/bin/bash
export PYTHONPATH=/app/data/code/MGE-LDM-main
export http_proxy="http://10.242.26.231:7890"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

set -e

GPU=1
CONFIG_NAME="dit"
CKPT_DIR="/app/data/code/test_code/MGE-LDM-main/ckps/gate_quality_aux_loss_X_HiddenState/dit/checkpoints/"
CKPT_PATH=$CKPT_DIR"unwrapped_DiT_31.ckpt"

## 推断条件
TASK="total_gen"
GEN_AUDIO_DUR=30.0
GIVEN_WAV_PATH=null

#### 设置文本提示
TEXT_PROMPT="bass, drums, guitar, piano"

## 生成配置
NUM_STEPS=250
CFG_SCALE=10.0
OVERLAP_DUR=7.0
REPAINT_N=1

# 创建主输出目录
MAIN_OUTPUT_DIR="./outputs_batch_quality"
mkdir -p ${MAIN_OUTPUT_DIR}

echo "开始批量生成不同质量分数的音轨..."
echo "质量分数范围: 1.0 到 9.0，步长: 1.0"
echo "输出目录: ${MAIN_OUTPUT_DIR}"
echo "文本提示数量: ${#TEXT_PROMPTS[@]}"
echo "生成参数: num_steps=${NUM_STEPS}, cfg_scale=${CFG_SCALE}, overlap_dur=${OVERLAP_DUR}, repaint_n=${REPAINT_N}"
echo ""

# 定义所有要测试的文本提示
#    "bass, drums, guitar, piano"
TEXT_PROMPTS=(
    "Funky upbeat jazz with guitar, saxophone and piano"
    "Funky upbeat jazz with guitar and piano"
    "Lo-fi hip hop beat with mellow jazzy chords and a smooth bassline"
    "Relaxing acoustic guitar instrumental with soft percussion"
    "Metal guitar riff with heavy distortion and fast-paced drums"
    "Upbeat electronic dance music with catchy synth melodies and driving bass"
    "EDM music with energetic beats and vibrant synths"
    "Groovy bassline with funky guitar riffs"
    "Chill lo-fi beat with soft vinyl crackle, laid-back piano chords, and mellow drums"
    "Epic cinematic trailer with soaring strings, powerful brass hits, and pounding timpani"
    "Neo-soul ballad with warm Rhodes piano, smooth electric guitar chords, and subtle strings"
    "8-bit chiptune melody with catchy bleeps, retro drums, and arpeggiated bass"
)


# 使用预定义的质量分数数组，每隔1.0测试（0.0到9.0）
QUALITY_SCORES=(0.8)

for QUALITY_SCORE in "${QUALITY_SCORES[@]}"; do
    
    echo "=========================================="
    echo "正在生成质量分数: ${QUALITY_SCORE}"
    echo "=========================================="
    
    for PROMPT_INDEX in "${!TEXT_PROMPTS[@]}"; do
        TEXT_PROMPT="${TEXT_PROMPTS[$PROMPT_INDEX]}"
        
        # 为每个质量分数和文本提示创建单独的输出目录
        # 简化命名：q{质量分数}_p{提示索引}
        OUTPUT_DIR="${MAIN_OUTPUT_DIR}/q${QUALITY_SCORE}_p${PROMPT_INDEX}"
        
        echo "正在生成质量分数: ${QUALITY_SCORE}, 提示 ${PROMPT_INDEX}: ${TEXT_PROMPT}"
        echo "输出目录: ${OUTPUT_DIR}"
        
        # 运行推理
        CUDA_VISIBLE_DEVICES=$GPU \
        python infer.py \
            --config-name $CONFIG_NAME \
            +task=$TASK \
            ckpt_path=${CKPT_PATH} \
            +gen_audio_dur=${GEN_AUDIO_DUR} \
            +given_wav_path=${GIVEN_WAV_PATH} \
            "+text_prompt='${TEXT_PROMPT}'" \
            +quality_score=${QUALITY_SCORE} \
            +num_steps=${NUM_STEPS} \
            +cfg_scale=${CFG_SCALE} \
            +overlap_dur=${OVERLAP_DUR} \
            +repaint_n=${REPAINT_N} \
            +output_dir=${OUTPUT_DIR} \
            +enable_attention_gating=true
        
        echo "完成: 质量分数 ${QUALITY_SCORE}, 提示 ${PROMPT_INDEX}"
        echo ""
        
        # 清理GPU缓存
        if command -v nvidia-smi &> /dev/null; then
            echo "清理GPU缓存..."
            python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
        fi
    done
    
    echo "质量分数 ${QUALITY_SCORE} 的所有提示生成完成！"
    echo ""
done

echo "=========================================="
echo "所有质量分数的音轨生成完成！"
echo "输出目录: ${MAIN_OUTPUT_DIR}"
echo "生成的质量分数: 1.0 到 9.0 (共9个质量分数)"
echo "每个质量分数测试了 ${#TEXT_PROMPTS[@]} 个文本提示"
echo "总共生成了 $((9 * ${#TEXT_PROMPTS[@]})) 个音频文件"
echo "=========================================="

# 创建汇总信息文件
SUMMARY_FILE="${MAIN_OUTPUT_DIR}/generation_summary.txt"
cat > ${SUMMARY_FILE} << EOF
批量质量生成汇总
==================

生成时间: $(date)
模型检查点: ${CKPT_PATH}
文本提示数量: ${#TEXT_PROMPTS[@]}
生成音频时长: ${GEN_AUDIO_DUR} 秒

生成参数:
- num_steps: ${NUM_STEPS}
- cfg_scale: ${CFG_SCALE}
- overlap_dur: ${OVERLAP_DUR}
- repaint_n: ${REPAINT_N}

测试的文本提示:
$(for i in "${!TEXT_PROMPTS[@]}"; do echo "${i}: ${TEXT_PROMPTS[$i]}"; done)

生成的质量分数:
1.0 - 低质量
2.0 - 较低质量
3.0 - 中等质量
4.0 - 中高质量
5.0 - 高质量
6.0 - 较高质量
7.0 - 很高质量
8.0 - 极高质量
9.0 - 最高质量
(共9个质量分数，从1.0到9.0，步长1.0)

输出目录结构:
${MAIN_OUTPUT_DIR}/
├── q1.0_p0/  (质量1.0, 提示0)
├── q1.0_p1/  (质量1.0, 提示1)
├── ...
├── q9.0_p11/ (质量9.0, 提示11)
└── ...
(总共 $((9 * ${#TEXT_PROMPTS[@]})) 个目录，每个质量分数对应 ${#TEXT_PROMPTS[@]} 个文本提示)

命名规则:
- q{质量分数}: 质量分数 (1.0-9.0)
- p{提示索引}: 文本提示索引 (0-11)

每个目录包含:
- gen_mix.wav (生成的混合音轨)
- gen_src_*.wav (生成的各个音源)
- generation_info.txt (生成信息)

实验目的:
验证质量感知模型是否能够根据不同的质量分数生成相应质量的音频，
以及在不同文本提示下质量感知能力的一致性。
EOF

echo "汇总信息已保存到: ${SUMMARY_FILE}"
