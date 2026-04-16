#!/usr/bin/env python3
"""
批量部分生成处理器 (无质量控制版本)
用于处理merged.jsonl数据，生成任务配置文件，用于批量部分生成任务
"""

import json
import os
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

def load_merged_data(input_file: str) -> List[Dict[str, Any]]:
    """加载merged.jsonl数据"""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def select_unknown_track(track_data: Dict[str, Any], seed: int = None) -> str:
    """随机选择一个未知音轨"""
    if seed is not None:
        random.seed(seed)
    
    # 可用的音轨类型
    available_tracks = ['Bass', 'Drums', 'Guitar', 'Piano']
    
    # 随机选择一个作为未知音轨
    unknown_track = random.choice(available_tracks)
    return unknown_track

def create_mixed_audio_path(track_data: Dict[str, Any], output_dir: str, unknown_track: str) -> str:
    """创建混合音频文件路径"""
    uid = track_data['uid']
    
    # 创建输出目录
    track_output_dir = os.path.join(output_dir, uid)
    os.makedirs(track_output_dir, exist_ok=True)
    
    # 混合音频文件路径 - 使用统一的文件名
    mixed_audio_path = os.path.join(track_output_dir, "given_audio.wav")
    
    return mixed_audio_path

def process_track_data(track_data: Dict[str, Any], output_dir: str, seed: int = None) -> Dict[str, Any]:
    """处理单个音轨数据"""
    # 随机选择未知音轨
    unknown_track = select_unknown_track(track_data, seed)
    
    # 创建混合音频路径
    given_wav_path = create_mixed_audio_path(track_data, output_dir, unknown_track)
    
    # 创建任务配置
    task_config = {
        'uid': track_data['uid'],
        'given_wav_path': given_wav_path,
        'text_prompt': unknown_track,  # 使用乐器名作为提示
        'unknown_instrument': unknown_track,
        'original_data': track_data  # 保留原始数据
    }
    
    return task_config

def main():
    parser = argparse.ArgumentParser(description='批量部分生成数据处理器 (无质量控制版本)')
    parser.add_argument('--input', type=str, required=True, help='输入merged.jsonl文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出任务配置文件路径')
    parser.add_argument('--output-dir', type=str, required=True, help='音频输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max-tracks', type=int, default=None, help='最大处理音轨数量')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 50)
    print("批量部分生成数据处理器 (无质量控制版本)")
    print("=" * 50)
    print(f"输入文件: {args.input}")
    print(f"输出配置: {args.output}")
    print(f"音频输出目录: {args.output_dir}")
    print(f"随机种子: {args.seed}")
    print("")
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("正在加载merged.jsonl数据...")
    track_data_list = load_merged_data(args.input)
    print(f"✓ 加载了 {len(track_data_list)} 个音轨")
    
    # 限制处理数量
    if args.max_tracks is not None:
        track_data_list = track_data_list[:args.max_tracks]
        print(f"✓ 限制处理数量为 {len(track_data_list)} 个音轨")
    
    # 处理每个音轨
    print("正在处理音轨数据...")
    task_configs = []
    
    for i, track_data in enumerate(track_data_list):
        try:
            task_config = process_track_data(track_data, args.output_dir, args.seed + i)
            task_configs.append(task_config)
            
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(track_data_list)} 个音轨")
                
        except Exception as e:
            print(f"  警告: 处理音轨 {track_data.get('uid', 'unknown')} 时出错: {e}")
            continue
    
    print(f"✓ 成功处理了 {len(task_configs)} 个音轨")
    
    # 保存任务配置
    print("正在保存任务配置...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for task_config in task_configs:
            f.write(json.dumps(task_config, ensure_ascii=False) + '\n')
    
    print(f"✓ 任务配置已保存到: {args.output}")
    
    # 统计信息
    print("")
    print("处理统计:")
    print(f"- 总音轨数: {len(track_data_list)}")
    print(f"- 成功处理: {len(task_configs)}")
    print(f"- 失败数量: {len(track_data_list) - len(task_configs)}")
    
    # 统计未知音轨分布
    unknown_track_counts = {}
    for task_config in task_configs:
        unknown_track = task_config['unknown_instrument']
        unknown_track_counts[unknown_track] = unknown_track_counts.get(unknown_track, 0) + 1
    
    print("")
    print("未知音轨分布:")
    for track, count in sorted(unknown_track_counts.items()):
        print(f"- {track}: {count} 个")
    
    print("")
    print("🎉 数据处理完成！")
    print("现在可以运行批量生成脚本了:")
    print(f"bash scripts/batch_partial_gen_no_quality.sh")
    
    return 0

if __name__ == "__main__":
    exit(main())
