# 无质量控制批量部分生成脚本

这个脚本集合用于测试原始MGE-LDM模型，不包含任何质量控制功能，作为质量控制模型的基线对比。

## 文件说明

### 1. 批量生成脚本
- **`scripts/batch_partial_gen_no_quality.sh`**: 主要的批量生成脚本
- **`batch_partial_gen_processor_no_quality.py`**: 数据处理脚本
- **`infer_no_quality.py`**: 无质量控制的推理脚本

### 2. 主要功能

#### 批量生成脚本 (`batch_partial_gen_no_quality.sh`)
- 处理 `manifests/merged.jsonl` 数据
- 为每个音轨随机选择未知乐器 (Bass/Drums/Guitar/Piano)
- 生成任务配置文件
- 批量执行部分生成任务
- 支持断点续传和重试机制
- **无质量控制参数**

#### 数据处理脚本 (`batch_partial_gen_processor_no_quality.py`)
- 加载并处理 merged.jsonl 数据
- 随机选择未知音轨
- 生成任务配置文件
- 统计处理结果

#### 推理脚本 (`infer_no_quality.py`)
- 简化的推理接口
- 不包含质量控制功能
- 支持部分生成和完全生成
- 自动保存生成结果

## 使用方法

### 1. 准备工作
```bash
# 确保以下文件存在
- manifests/merged.jsonl
- 模型检查点文件
- 配置文件 (configs/dit.yaml)
```

### 2. 运行批量生成
```bash
# 给脚本添加执行权限
chmod +x scripts/batch_partial_gen_no_quality.sh

# 运行批量生成
bash scripts/batch_partial_gen_no_quality.sh
```

### 3. 单独运行推理
```bash
python infer_no_quality.py \
    --config-name dit \
    --ckpt-path /path/to/checkpoint.ckpt \
    --task partial_gen \
    --given-wav-path /path/to/audio.wav \
    --text-prompt "the sound of Bass" \
    --output-dir ./output \
    --num-steps 100 \
    --cfg-scale 10.0
```

## 配置参数

### 批量生成脚本参数
- `GPU`: GPU设备ID (默认: 4)
- `CONFIG_NAME`: 配置名称 (默认: "dit")
- `CKPT_PATH`: 模型检查点路径
- `NUM_STEPS`: 扩散步数 (默认: 100)
- `CFG_SCALE`: CFG缩放 (默认: 10.0)
- `OVERLAP_DUR`: 重叠时长 (默认: 6.0)
- `REPAINT_N`: 重绘次数 (默认: 1)

### 推理脚本参数
- `--config-name`: 配置名称
- `--ckpt-path`: 模型检查点路径
- `--task`: 任务类型 (partial_gen/total_gen)
- `--given-wav-path`: 给定音频路径
- `--text-prompt`: 文本提示
- `--output-dir`: 输出目录
- `--num-steps`: 扩散步数
- `--cfg-scale`: CFG缩放
- `--overlap-dur`: 重叠时长
- `--repaint-n`: 重绘次数
- `--segment-duration`: 片段时长
- `--random-segment`: 随机片段截取
- `--device`: 设备 (默认: cuda:0)
- `--seed`: 随机种子 (默认: 42)

## 输出结构

```
outputs_batch_partial_gen_no_quality/
├── test_Track01892/
│   ├── gen_mix.wav
│   ├── gen_src_1.wav
│   ├── gen_src_2.wav
│   ├── gen_src_3.wav
│   └── generation_info.txt
├── test_Track02029/
│   └── ...
└── generation_summary.txt
```

## 与质量控制版本的对比

| 特性 | 无质量控制版本 | 质量控制版本 |
|------|----------------|--------------|
| 质量分数 | 无 | 0.0-1.0 |
| 质量编码器 | 无 | quality_scores -> quality_tokens |
| 条件器集成 | 无 | MultiConditioner |
| 权重混合 | 无 | 80%内容+20%质量 |
| 生成质量 | 模型默认 | 可调节 |
| 基线对比 | ✓ | 实验组 |

## 注意事项

1. **无质量控制**: 此版本不包含任何质量控制功能
2. **基线对比**: 用于与质量控制版本进行对比
3. **模型兼容**: 适用于原始MGE-LDM模型
4. **断点续传**: 支持中断后继续执行
5. **错误处理**: 包含重试机制和错误处理

## 故障排除

### 常见问题
1. **模型加载失败**: 检查检查点路径和配置文件
2. **音频加载失败**: 检查音频文件路径和格式
3. **内存不足**: 减少批处理大小或使用更小的模型
4. **设备错误**: 检查CUDA可用性和设备ID

### 调试模式
```bash
# 启用详细输出
export CUDA_LAUNCH_BLOCKING=1
bash scripts/batch_partial_gen_no_quality.sh
```

## 实验目的

这个无质量控制版本的主要目的是：
1. 建立基线性能
2. 与质量控制版本进行对比
3. 验证原始模型的生成能力
4. 为质量控制效果评估提供参考
