#!/usr/bin/env python
"""统计 slakh2100, musdb18, moisesdb 三个数据集的 cocola 分布"""

import os
import json
import numpy as np
import pandas as pd
import math

DATA_DIR = "/app/data/data/pre_extracted_latents"

def get_dataset_configs():
    """获取数据集配置"""
    base_dir = DATA_DIR
    return {
        "slakh2100": {
            "data_dir": f"{base_dir}/slakh2100",
            "split": "train",
        },
        "musdb18": {
            "data_dir": f"{base_dir}/musdb18",
            "split": "train",
        },
        "moisesdb": {
            "data_dir": f"{base_dir}/moisesdb",
            "split": "train",
        },
    }

def collect_all_cocola_scores(dataset_configs):
    """收集所有 cocola 分数"""
    dataset_samples = {}

    for dataset_name, dataset_cfg in dataset_configs.items():
        data_dir = dataset_cfg["data_dir"]
        split_name = dataset_cfg.get("split", None)

        print(f"扫描数据集 {dataset_name} ({split_name})...")

        if split_name:
            dataset_dir = os.path.join(data_dir, split_name)
        else:
            dataset_dir = data_dir

        if not os.path.exists(dataset_dir):
            print(f"警告：数据集目录不存在 {dataset_dir}")
            continue

        scores = []
        count = 0

        try:
            tracks = os.listdir(dataset_dir)
        except Exception as e:
            print(f"错误：无法读取目录 {dataset_dir}: {e}")
            continue

        for track in tracks:
            track_dir = os.path.join(dataset_dir, track)
            if not os.path.exists(track_dir) or not track.startswith("track"):
                continue

            try:
                comb_dirs = [d for d in os.listdir(track_dir) if d.startswith("comb")]
            except:
                continue

            for comb_dir in comb_dirs:
                comb_path = os.path.join(track_dir, comb_dir)
                comb_info_path = os.path.join(comb_path, "comb_info.json")

                if not os.path.exists(comb_info_path):
                    continue

                try:
                    with open(comb_info_path, "r") as f:
                        comb_info = json.load(f)

                    if "cocola_score" in comb_info:
                        src_label = comb_info.get("src_label", "")
                        if "other" not in src_label.lower():
                            scores.append(comb_info["cocola_score"])
                            count += 1
                except:
                    continue

        dataset_samples[dataset_name] = scores
        print(f"  收集到 {count} 条记录")

    return dataset_samples

def main():
    print("="*60)
    print("收集 slakh2100, musdb18, moisesdb 三个数据集的 cocola 数值")
    print("="*60)
    print(f"数据目录: {DATA_DIR}\n")

    dataset_configs = get_dataset_configs()
    dataset_samples = collect_all_cocola_scores(dataset_configs)

    if not dataset_samples:
        print("未找到任何 cocola 数据")
        return

    # 合并所有数据求全局范围
    all_scores = []
    for scores in dataset_samples.values():
        all_scores.extend(scores)

    total_count = len(all_scores)
    print(f"\n总计收集到 {total_count} 条记录")

    # 计算全局最小最大值，确定区间
    min_val = math.floor(min(all_scores))
    max_val = math.ceil(max(all_scores))

    # 固定区间大小为1，生成区间列表
    bins = list(range(min_val, max_val + 2))  # +2 确保包含最大值

    # 创建区间标签
    bin_labels = [f"{bins[i]}" for i in range(len(bins) - 1)]

    # 计算每个数据集在各区间的频率
    result_data = {"区间": bin_labels}

    for dataset_name, scores in dataset_samples.items():
        hist, _ = np.histogram(scores, bins=bins)
        result_data[dataset_name] = hist.tolist()

    result_df = pd.DataFrame(result_data)

    # 同时生成百分比版本
    result_pct_data = {"区间": bin_labels}
    for dataset_name, scores in dataset_samples.items():
        hist, _ = np.histogram(scores, bins=bins)
        pct = [round(x / len(scores) * 100, 2) if len(scores) > 0 else 0 for x in hist]
        result_pct_data[dataset_name] = pct

    result_pct_df = pd.DataFrame(result_pct_data)

    # 打印结果
    print("\n" + "="*60)
    print("频率分布（计数）：")
    print("="*60)
    print(result_df.to_string(index=False))

    print("\n" + "="*60)
    print("频率分布（百分比 %）：")
    print("="*60)
    print(result_pct_df.to_string(index=False))

    # 保存到 Excel
    output_dir = "/app/data/code/test_code/MGE-LDM-main/ckps"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cocola_distribution.xlsx")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='频率计数', index=False)
        result_pct_df.to_excel(writer, sheet_name='频率百分比', index=False)

    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
