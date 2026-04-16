#!/usr/bin/env python3
"""
批量质量控制部分生成数据处理器

该脚本读取merged.jsonl文件，为每个条目随机选择一个音轨作为未知音轨，
将剩余音轨按时间轴混合成一个音轨作为partial_gen任务的GIVEN_WAV_PATH，
并将未知音轨的乐器名作为TEXT_PROMPT。

使用方法:
python batch_partial_gen_processor.py --input manifests/merged.jsonl --output batch_partial_gen_tasks.jsonl
"""

import json
import random
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def load_merged_data(input_file):
    """加载merged.jsonl文件"""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def mix_audio_files(audio_files, output_path):
    """
    使用ffmpeg将多个音频文件按时间轴混合成一个音轨
    """
    if len(audio_files) == 1:
        # 如果只有一个文件，直接复制
        subprocess.run(['cp', audio_files[0], output_path], check=True)
        return
    
    # 构建ffmpeg命令来混合多个音频文件
    # 使用amix滤镜将多个音频输入混合成一个输出
    cmd = ['ffmpeg']
    
    # 添加所有输入文件
    for audio_file in audio_files:
        cmd.extend(['-i', audio_file])
    
    # 添加amix滤镜来混合音频
    # inputs=len(audio_files) 表示输入文件数量
    # duration=longest 表示输出时长取最长的输入文件
    filter_complex = f"amix=inputs={len(audio_files)}:duration=longest"
    cmd.extend(['-filter_complex', filter_complex])
    
    # 输出设置
    cmd.extend(['-c:a', 'pcm_s16le', '-y', output_path])
    
    # 执行命令
    subprocess.run(cmd, check=True, capture_output=True)


def process_track_data(track_data, output_dir):
    """
    处理单个音轨数据，随机选择一个音轨作为未知音轨
    """
    uid = track_data['uid']
    ref_sources = track_data['ref_sources']
    
    # 获取所有可用的乐器类型
    available_instruments = list(ref_sources.keys())
    
    if len(available_instruments) < 2:
        print(f"警告: {uid} 只有 {len(available_instruments)} 个乐器，跳过")
        return None
    
    # 随机选择一个乐器作为未知音轨
    unknown_instrument = random.choice(available_instruments)
    
    # 获取剩余乐器
    remaining_instruments = [inst for inst in available_instruments if inst != unknown_instrument]
    
    # 收集剩余乐器的所有音频文件
    remaining_audio_files = []
    for instrument in remaining_instruments:
        remaining_audio_files.extend(ref_sources[instrument])
    
    # 创建输出目录
    track_output_dir = Path(output_dir) / uid
    track_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 混合剩余音轨
    mixed_audio_path = track_output_dir / "given_audio.wav"
    mix_audio_files(remaining_audio_files, str(mixed_audio_path))
    
    # 创建任务配置
    task_config = {
        'uid': uid,
        'unknown_instrument': unknown_instrument,
        'remaining_instruments': remaining_instruments,
        'given_wav_path': str(mixed_audio_path),
        'text_prompt': unknown_instrument,
        'ref_sources': ref_sources,
        'unknown_audio_files': ref_sources[unknown_instrument]
    }
    
    return task_config


def main():
    parser = argparse.ArgumentParser(description='批量质量控制部分生成数据处理器')
    parser.add_argument('--input', required=True, help='输入的merged.jsonl文件路径')
    parser.add_argument('--output', required=True, help='输出的任务配置文件路径')
    parser.add_argument('--output-dir', default='./batch_partial_gen_data', 
                       help='音频文件输出目录 (默认: ./batch_partial_gen_data)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--max-tracks', type=int, help='最大处理音轨数量 (用于测试)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"正在处理文件: {args.input}")
    print(f"输出目录: {args.output_dir}")
    print(f"随机种子: {args.seed}")
    
    # 加载数据
    track_data_list = load_merged_data(args.input)
    
    if args.max_tracks:
        track_data_list = track_data_list[:args.max_tracks]
        print(f"限制处理音轨数量: {args.max_tracks}")
    
    print(f"总共需要处理 {len(track_data_list)} 个音轨")
    
    # 处理每个音轨
    processed_tasks = []
    failed_count = 0
    
    for i, track_data in enumerate(track_data_list):
        print(f"处理进度: {i+1}/{len(track_data_list)} - {track_data['uid']}")
        
        try:
            task_config = process_track_data(track_data, args.output_dir)
            if task_config:
                processed_tasks.append(task_config)
                print(f"  ✓ 未知音轨: {task_config['unknown_instrument']}")
                print(f"  ✓ 剩余音轨: {', '.join(task_config['remaining_instruments'])}")
            else:
                failed_count += 1
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            failed_count += 1
    
    # 保存任务配置
    with open(args.output, 'w', encoding='utf-8') as f:
        for task in processed_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')
    
    print(f"\n处理完成!")
    print(f"成功处理: {len(processed_tasks)} 个音轨")
    print(f"失败: {failed_count} 个音轨")
    print(f"任务配置已保存到: {args.output}")
    
    # 统计信息
    instrument_stats = {}
    for task in processed_tasks:
        inst = task['unknown_instrument']
        instrument_stats[inst] = instrument_stats.get(inst, 0) + 1
    
    print(f"\n未知音轨统计:")
    for inst, count in sorted(instrument_stats.items()):
        print(f"  {inst}: {count} 次")


if __name__ == '__main__':
    main()
