#!/usr/bin/env python3
"""
FAD 值统计分析工具

用于分析 fad_values.txt 文件，计算各种统计指标
"""

import argparse
import os
import sys
import json
from statistics import mean, median, pstdev, variance
from collections import defaultdict
import numpy as np


def load_fad_values(file_path):
    """加载 FAD 值文件"""
    values = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        value = float(line)
                        values.append(value)
                    except ValueError:
                        print(f"警告: 无法解析行 '{line}'，跳过", file=sys.stderr)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在", file=sys.stderr)
        return []
    except Exception as e:
        print(f"错误: 读取文件时出错: {e}", file=sys.stderr)
        return []

    return values


def calculate_statistics(values):
    """计算统计指标"""
    if not values:
        return None

    values_sorted = sorted(values)

    stats = {
        'count': len(values),
        'mean': mean(values),
        'median': median(values),
        'min': min(values),
        'max': max(values),
        'range': max(values) - min(values),
    }

    # 计算分位数
    if len(values) >= 4:
        stats['q1'] = values_sorted[len(values) // 4]  # 25% 分位数
        stats['q3'] = values_sorted[3 * len(values) // 4]  # 75% 分位数
        stats['iqr'] = stats['q3'] - stats['q1']  # 四分位距
    else:
        stats['q1'] = None
        stats['q3'] = None
        stats['iqr'] = None

    # 计算标准差和方差（需要至少2个值）
    if len(values) > 1:
        stats['std'] = pstdev(values)  # 总体标准差
        stats['var'] = variance(values)  # 方差
    else:
        stats['std'] = 0.0
        stats['var'] = 0.0

    # 计算变异系数 (CV)
    if stats['mean'] != 0:
        stats['cv'] = stats['std'] / abs(stats['mean'])
    else:
        stats['cv'] = float('inf')

    # 计算偏度和峰度（需要足够的数据）
    if len(values) >= 3:
        values_array = np.array(values)
        stats['skewness'] = np.mean(((values_array - stats['mean']) / stats['std']) ** 3)
        stats['kurtosis'] = np.mean(((values_array - stats['mean']) / stats['std']) ** 4) - 3
    else:
        stats['skewness'] = None
        stats['kurtosis'] = None

    return stats


def print_statistics(stats, file_path):
    """打印统计结果"""
    if not stats:
        print(f"文件 '{file_path}' 中没有有效的 FAD 值")
        return

    print("=" * 60)
    print(f"FAD 值统计分析")
    print(f"文件: {file_path}")
    print("=" * 60)

    print("基本统计:")
    print(f"  样本数量: {stats['count']}")
    print(f"  平均值: {stats['mean']:.4f}")
    print(f"  中位数: {stats['median']:.4f}")
    print(f"  最小值: {stats['min']:.4f}")
    print(f"  最大值: {stats['max']:.4f}")
    print(f"  范围: {stats['range']:.4f}")

    if stats['q1'] is not None:
        print("\n分位数统计:")
        print(f"  25%分位数: {stats['q1']:.4f}")
        print(f"  75%分位数: {stats['q3']:.4f}")
        print(f"  四分位距: {stats['iqr']:.4f}")

    print("\n分布统计:")
    print(f"  标准差: {stats['std']:.4f}")
    print(f"  方差: {stats['var']:.4f}")
    if stats['cv'] != float('inf'):
        print(f"  变异系数: {stats['cv']:.4f}")
    else:
        print("  变异系数: 未定义 (平均值为0)")

    if stats['skewness'] is not None:
        print("\n分布形状:")
        print(f"  偏度: {stats['skewness']:.4f}")
        print(f"  峰度: {stats['kurtosis']:.4f}")

    # FAD 值质量评估
    print("\n质量评估 (FAD 值越小越好):")
    fad_mean = stats['mean']
    if fad_mean < 5:
        quality = "优秀"
    elif fad_mean < 10:
        quality = "良好"
    elif fad_mean < 20:
        quality = "一般"
    elif fad_mean < 50:
        quality = "较差"
    else:
        quality = "很差"

    print(f"  平均 FAD: {fad_mean:.4f}")
    print(f"  质量等级: {quality}")

    # 分布分析
    if stats['std'] > 0:
        cv = stats['cv'] if stats['cv'] != float('inf') else 0
        if cv < 0.2:
            consistency = "很高"
        elif cv < 0.5:
            consistency = "较高"
        elif cv < 1.0:
            consistency = "一般"
        else:
            consistency = "较低"

        print(f"  一致性: {consistency} (变异系数: {cv:.4f})")


def analyze_multiple_files(base_dir):
    """分析多个 fad_values.txt 文件"""
    results = defaultdict(dict)

    # 遍历所有质量分数目录
    for item in os.listdir(base_dir):
        if item.startswith('quality_') and os.path.isdir(os.path.join(base_dir, item)):
            quality_dir = os.path.join(base_dir, item)
            fad_file = os.path.join(quality_dir, 'fad_values.txt')

            if os.path.exists(fad_file):
                quality_score = item.replace('quality_', '')
                values = load_fad_values(fad_file)
                if values:
                    stats = calculate_statistics(values)
                    if stats:
                        results[quality_score] = stats

    if results:
        print("=" * 80)
        print("多质量分数 FAD 对比分析")
        print(f"目录: {base_dir}")
        print("=" * 80)

        # 按质量分数排序
        sorted_qualities = sorted(results.keys(), key=float)

        print("<8")
        print("-" * 50)

        for quality in sorted_qualities:
            stats = results[quality]
            print("<8"
                  "<8.4f"
                  "<8.4f"
                  "<8.4f")

        # 找出最佳质量分数
        best_quality = min(results.keys(), key=lambda q: results[q]['mean'])
        print("\n最佳质量分数: {} (平均 FAD: {:.4f})".format(best_quality, results[best_quality]['mean']))

        return results

    return None


def save_results(stats, output_file, file_path):
    """保存结果到文件"""
    result = {
        'file': file_path,
        'statistics': stats,
        'timestamp': str(np.datetime64('now'))
    }

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果失败: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="FAD 值统计分析工具")
    parser.add_argument('input', help='fad_values.txt 文件路径，或包含多个质量目录的根目录')
    parser.add_argument('--output', '-o', help='输出 JSON 文件路径')
    parser.add_argument('--multi', '-m', action='store_true',
                       help='分析多个质量分数的目录（当输入为目录时）')

    args = parser.parse_args()

    if os.path.isdir(args.input):
        if args.multi:
            # 分析多个质量分数目录
            results = analyze_multiple_files(args.input)
            if results and args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump({
                        'analysis_type': 'multi_quality',
                        'base_directory': args.input,
                        'results': results,
                        'timestamp': str(np.datetime64('now'))
                    }, f, indent=2, ensure_ascii=False)
                print(f"\n多质量分析结果已保存到: {args.output}")
        else:
            # 查找默认的 fad_values.txt
            fad_file = os.path.join(args.input, 'fad_values.txt')
            if os.path.exists(fad_file):
                values = load_fad_values(fad_file)
                if values:
                    stats = calculate_statistics(values)
                    print_statistics(stats, fad_file)
                    if args.output:
                        save_results(stats, args.output, fad_file)
                else:
                    print(f"未找到有效的 FAD 值文件: {fad_file}")
            else:
                print(f"目录中未找到 fad_values.txt: {args.input}")
                print("使用 --multi 参数来分析多个质量分数目录")

    elif os.path.isfile(args.input):
        # 分析单个文件
        values = load_fad_values(args.input)
        if values:
            stats = calculate_statistics(values)
            print_statistics(stats, args.input)
            if args.output:
                save_results(stats, args.output, args.input)
        else:
            print(f"文件中没有有效的 FAD 值: {args.input}")
            sys.exit(1)
    else:
        print(f"输入路径不存在: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
