# 部分生成任务FAD计算使用说明

本文档说明如何使用新增的FAD计算功能来评估部分生成任务的质量。

## 文件说明

### 1. `compute_fad_partial_gen.py`
独立的FAD计算脚本，用于计算单个部分生成任务的FAD值。

**功能：**
- 读取生成的混合音频 (`gen_mix.wav`)
- 构建参考音频（给定音频 + 未知乐器音频）
- 使用PANNs模型提取音频嵌入
- 计算FAD (Fréchet Audio Distance) 值

**使用方法：**
```bash
python compute_fad_partial_gen.py \
    --gen_mix_path <生成的混合音频路径> \
    --given_wav_path <给定音频路径> \
    --unknown_audio_files <未知乐器音频文件1> <未知乐器音频文件2> ... \
    --sample_rate 48000 \
    --panns_checkpoint /root/panns_data/Cnn14_mAP=0.431.pth \
    --device cuda \
    --output_json <输出JSON路径（可选）>
```

**参数说明：**
- `--gen_mix_path`: 生成的混合音频文件路径（必需）
- `--given_wav_path`: 给定音频文件路径（必需）
- `--unknown_audio_files`: 未知乐器音频文件列表（必需，可多个）
- `--sample_rate`: 采样率（默认：48000）
- `--panns_checkpoint`: PANNs模型检查点路径（默认：/root/panns_data/Cnn14_mAP=0.431.pth）
- `--device`: 计算设备，cuda或cpu（默认：自动检测）
- `--output_json`: 输出JSON文件路径（可选）

### 2. `batch_compute_fad_partial_gen.py`
批量FAD计算脚本，用于批量处理已生成的结果。

**功能：**
- 读取任务配置文件 (`batch_partial_gen_tasks.jsonl`)
- 遍历所有任务和质量控制分数
- 为每个任务计算FAD值
- 生成汇总统计信息

**使用方法：**
```bash
python batch_compute_fad_partial_gen.py \
    --tasks_jsonl batch_partial_gen_tasks.jsonl \
    --output_root ./outputs_batch_partial_gen_quality \
    --quality_scores 0.1 0.9 \
    --sample_rate 48000 \
    --panns_checkpoint /root/panns_data/Cnn14_mAP=0.431.pth \
    --device cuda \
    --output_summary ./fad_summary.json \
    --skip_existing
```

**参数说明：**
- `--tasks_jsonl`: 任务配置JSONL文件路径（必需）
- `--output_root`: 批量生成输出根目录（必需）
- `--quality_scores`: 质量控制分数列表（默认：[0.1, 0.9]）
- `--sample_rate`: 采样率（默认：48000）
- `--panns_checkpoint`: PANNs模型检查点路径（默认：/root/panns_data/Cnn14_mAP=0.431.pth）
- `--device`: 计算设备（默认：自动检测）
- `--output_summary`: 输出汇总JSON文件路径（默认：./fad_summary.json）
- `--skip_existing`: 跳过已计算FAD的任务（可选）

**输出格式：**
生成的汇总JSON文件包含：
- `total_tasks`: 总任务数
- `total_quality_scores`: 质量控制分数数量
- `successful`: 成功计算的数量
- `failed`: 失败的数量
- `statistics_by_quality`: 按质量控制分数分组的统计信息（均值、最小值、最大值、标准差）
- `items`: 每个任务的详细结果

### 3. `scripts/batch_partial_gen_quality_with_fad.sh`
集成FAD计算的批量生成脚本。

**功能：**
- 执行批量部分生成任务
- 生成后立即计算FAD值（如果启用）
- 自动生成FAD汇总报告

**使用方法：**
```bash
bash scripts/batch_partial_gen_quality_with_fad.sh
```

**配置说明：**
脚本中的关键配置项：
- `COMPUTE_FAD_AFTER_GEN`: 是否在生成后立即计算FAD（默认：true）
- `PANNS_CHECKPOINT`: PANNs模型检查点路径
- `DEVICE`: FAD计算设备（默认：cuda）
- `QUALITY_SCORES`: 质量控制分数数组

**输出结构：**
```
outputs_batch_partial_gen_quality/
├── quality_0.1/
│   ├── test_Track01892/
│   │   ├── gen_mix.wav (或 output_0001/gen_mix.wav)
│   │   ├── fad_result.json (FAD计算结果)
│   │   └── ...
│   └── ...
├── quality_0.9/
│   └── ...
├── fad_summary.json (批量FAD汇总)
└── generation_summary.txt (生成汇总报告)
```

## 工作流程

### 方案1：生成时计算FAD（推荐）
使用 `batch_partial_gen_quality_with_fad.sh`，在生成过程中自动计算FAD：

```bash
# 1. 确保任务配置文件存在
# batch_partial_gen_tasks.jsonl

# 2. 运行批量生成（带FAD计算）
bash scripts/batch_partial_gen_quality_with_fad.sh
```

### 方案2：事后批量计算FAD
如果已经完成了批量生成，可以使用批量计算脚本：

```bash
# 1. 确保已生成所有音频文件
# 2. 运行批量FAD计算
python batch_compute_fad_partial_gen.py \
    --tasks_jsonl batch_partial_gen_tasks.jsonl \
    --output_root ./outputs_batch_partial_gen_quality \
    --quality_scores 0.1 0.9 \
    --output_summary ./fad_summary.json \
    --skip_existing
```

### 方案3：单独计算FAD
对于单个任务，可以使用独立脚本：

```bash
python compute_fad_partial_gen.py \
    --gen_mix_path ./outputs_batch_partial_gen_quality/quality_0.5/test_Track01892/gen_mix.wav \
    --given_wav_path ./batch_partial_gen_data/test_Track01892/given_audio.wav \
    --unknown_audio_files /path/to/unknown/instrument.wav \
    --output_json ./fad_result.json
```

## 注意事项

1. **PANNs模型检查点**：确保PANNs模型检查点文件存在且路径正确。默认路径为 `/root/panns_data/Cnn14_mAP=0.431.pth`。

2. **音频文件路径**：确保所有音频文件路径正确且文件存在。

3. **GPU内存**：FAD计算会使用GPU，如果GPU内存不足，可以设置 `--device cpu`。

4. **断点续传**：
   - 使用 `batch_partial_gen_quality_with_fad.sh` 时，脚本会自动跳过已生成的任务
   - 使用 `batch_compute_fad_partial_gen.py` 时，添加 `--skip_existing` 参数可以跳过已计算FAD的任务

5. **音频长度匹配**：FAD计算会自动处理音频长度不匹配的情况（取较短的长度）。

## 输出结果解读

### FAD值说明
- FAD值越小，表示生成音频与参考音频的分布越接近
- 通常FAD值在0-10之间，值越小越好
- 不同质量控制分数下的FAD值可以用于评估质量感知模型的效果

### 汇总统计信息
`fad_summary.json` 文件包含：
- 每个质量控制分数的平均FAD值
- 每个质量控制分数的最小/最大FAD值
- 每个质量控制分数的标准差

这些统计信息可以帮助分析：
- 不同质量控制分数对生成质量的影响
- 质量感知模型的一致性
- 不同未知乐器类型下的性能差异

## 故障排除

1. **PANNs模型加载失败**：
   - 检查检查点文件路径是否正确
   - 确保已安装 `panns_inference` 包

2. **音频文件找不到**：
   - 检查文件路径是否正确
   - 检查文件是否存在

3. **FAD计算失败**：
   - 检查音频文件是否损坏
   - 检查GPU内存是否充足
   - 尝试使用CPU计算（设置 `--device cpu`）

4. **ffmpeg混合失败**：
   - 确保已安装ffmpeg
   - 检查音频文件格式是否支持

## 示例输出

### 单个任务FAD结果 (fad_result.json)
```json
{
  "gen_mix_path": "./outputs/quality_0.5/test_Track01892/gen_mix.wav",
  "given_wav_path": "./batch_partial_gen_data/test_Track01892/given_audio.wav",
  "unknown_audio_files": ["/path/to/unknown.wav"],
  "results": {
    "FAD": 2.3456
  }
}
```

### 批量FAD汇总 (fad_summary.json)
```json
{
  "total_tasks": 152,
  "total_quality_scores": 2,
  "successful": 304,
  "failed": 0,
  "statistics_by_quality": {
    "0.1": {
      "count": 152,
      "mean": 2.1234,
      "min": 1.5678,
      "max": 3.9012,
      "std": 0.4567
    },
    "0.9": {
      "count": 152,
      "mean": 1.9876,
      "min": 1.2345,
      "max": 3.4567,
      "std": 0.3456
    }
  },
  "items": [...]
}
```

## 参考

- FAD (Fréchet Audio Distance) 论文：https://arxiv.org/abs/1812.08434
- PANNs模型：https://github.com/qiuqiangkong/panns_inference

