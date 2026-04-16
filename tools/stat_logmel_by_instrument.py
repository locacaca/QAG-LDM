#!/usr/bin/env python3
"""
统计 logmel_l1_results.csv 中四种音轨各自的 logmel L1 结果
使用方法: python tools/stat_logmel_by_instrument.py --csv <csv_file_path>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='统计不同音轨的 logmel L1 结果')
    parser.add_argument(
        '--csv',
        type=str,
        default='outputs_batch_src_extract_quality/logmel_l1_results.csv',
        help='CSV 结果文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出统计结果的 JSON 文件路径（可选）'
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    
    # 检查文件是否存在
    if not csv_path.exists():
        print(f"错误: CSV 文件不存在: {csv_path}")
        print(f"请确认文件路径是否正确，或先运行 batch_src_extract_quality.sh 生成结果")
        return 1

    # 读取 CSV 文件
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"错误: 无法读取 CSV 文件: {e}")
        return 1

    # 检查必需的列
    required_cols = ['uid', 'quality', 'instrument', 'output_dir', 'logmel_l1']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误: CSV 文件缺少必需的列: {missing_cols}")
        print(f"当前列: {list(df.columns)}")
        return 1

    # 过滤掉无效值
    df_valid = df.dropna(subset=['logmel_l1']).copy()
    df_valid = df_valid[df_valid['logmel_l1'].notna()]

    if len(df_valid) == 0:
        print("警告: CSV 文件中没有有效的 logmel_l1 数据")
        return 1

    print("=" * 80)
    print(f"统计文件: {csv_path}")
    print(f"总记录数: {len(df)}, 有效记录数: {len(df_valid)}")
    print("=" * 80)
    print()

    # 按音轨分组统计
    if 'instrument' not in df_valid.columns:
        print("错误: CSV 文件中没有 'instrument' 列")
        return 1

    # 统计每个音轨
    stats_by_instrument = {}
    instruments = sorted(df_valid['instrument'].unique())

    print("按音轨统计 Log-Mel L1 结果:")
    print("-" * 80)

    all_stats = {}

    for instrument in instruments:
        inst_df = df_valid[df_valid['instrument'] == instrument]
        values = inst_df['logmel_l1'].values

        stats = {
            'count': len(inst_df),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
        }

        all_stats[instrument] = stats

        print(f"\n【{instrument}】")
        print(f"  样本数量: {stats['count']}")
        print(f"  均值    : {stats['mean']:.6f}")
        print(f"  中位数   : {stats['median']:.6f}")
        print(f"  标准差   : {stats['std']:.6f}")
        print(f"  最小值   : {stats['min']:.6f}")
        print(f"  最大值   : {stats['max']:.6f}")
        print(f"  25%分位数: {stats['q25']:.6f}")
        print(f"  75%分位数: {stats['q75']:.6f}")

    print()
    print("-" * 80)

    # 按质量控制分数分组统计（如果存在多个质量分数）
    if 'quality' in df_valid.columns:
        qualities = sorted(df_valid['quality'].unique())
        if len(qualities) > 1:
            print("\n按质量控制分数和音轨分组统计:")
            print("-" * 80)
            
            for quality in qualities:
                print(f"\n质量分数 = {quality}:")
                quality_df = df_valid[df_valid['quality'] == quality]
                
                for instrument in instruments:
                    inst_df = quality_df[quality_df['instrument'] == instrument]
                    if len(inst_df) > 0:
                        values = inst_df['logmel_l1'].values
                        mean_val = float(np.mean(values))
                        print(f"  {instrument:20s}: 均值={mean_val:.6f}, 数量={len(inst_df)}")

    # 输出到 JSON 文件（如果指定）
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'csv_file': str(csv_path),
            'total_records': len(df),
            'valid_records': len(df_valid),
            'statistics_by_instrument': all_stats
        }
        
        # 添加按质量分数分组统计
        if 'quality' in df_valid.columns:
            quality_stats = {}
            for quality in sorted(df_valid['quality'].unique()):
                quality_df = df_valid[df_valid['quality'] == quality]
                quality_instrument_stats = {}
                for instrument in instruments:
                    inst_df = quality_df[quality_df['instrument'] == instrument]
                    if len(inst_df) > 0:
                        values = inst_df['logmel_l1'].values
                        quality_instrument_stats[instrument] = {
                            'count': len(inst_df),
                            'mean': float(np.mean(values)),
                            'median': float(np.median(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                        }
                quality_stats[str(quality)] = quality_instrument_stats
            summary['statistics_by_quality'] = quality_stats
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n统计结果已保存到: {output_path}")

    print()
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())

