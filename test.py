import os; opj = os.path.join
import re
from omegaconf import OmegaConf

import hydra
from hydra.core.hydra_config import HydraConfig

import pytorch_lightning as pl

import json
import os


def fix_fad_score(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    # 1. 读取原始数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 提取所有 items 中的 FAD 值
    fad_values = []
    for item in data.get('items', []):
        fad = item.get('metrics', {}).get('FAD')
        # 排除 None 或 NaN 值
        if fad is not None and str(fad).lower() != 'nan':
            fad_values.append(float(fad))

    # 3. 计算平均值
    if fad_values:
        average_fad = sum(fad_values) / len(fad_values)
        print(f"成功计算! 样本数: {len(fad_values)}, 平均 FAD: {average_fad:.6f}")
    else:
        average_fad = 0.0
        print("警告: 未在 items 中找到有效的 FAD 数据")

    # 4. 更新 aggregate 字段
    data['aggregate']['FAD'] = average_fad

    # 5. 保存回文件 (或者另存为新文件)
    output_path = file_path  # 覆盖原文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"已更新文件: {output_path}")


if __name__ == "__main__":
    fix_fad_score('/app/data/code/test_code/MGE-LDM-main/outputs_eval_quality_with_control_se/summary_total_gen.json')