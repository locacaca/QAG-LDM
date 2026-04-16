#!/usr/bin/env python3
"""
无质量控制的推理脚本
用于测试原始模型，不包含任何质量控制功能
"""

import os
import argparse
import torch
import json
from pathlib import Path
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from multi_track_stable_audio.models.factory import create_mgeldm_from_config
from multi_track_stable_audio.inference.task_wrapper import InferenceTaskWrapper
from multi_track_stable_audio.utils import load_ckpt_state_dict
from multi_track_stable_audio.utils import set_seed

def load_audio_file(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """加载音频文件"""
    import librosa
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=False)
    
    # 确保是2D张量 (channels, samples)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    
    return torch.tensor(audio, dtype=torch.float32)

def save_generation_info(output_dir: str, config: dict):
    """保存生成信息"""
    info_file = os.path.join(output_dir, "generation_info.txt")
    
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("生成信息 (无质量控制)\n")
        f.write("=" * 50 + "\n")
        f.write(f"任务类型: {config.get('task', 'unknown')}\n")
        f.write(f"给定音频: {config.get('given_wav_path', 'N/A')}\n")
        f.write(f"文本提示: {config.get('text_prompt', 'N/A')}\n")
        f.write(f"生成时长: {config.get('gen_audio_dur', 'N/A')} 秒\n")
        f.write(f"CFG缩放: {config.get('cfg_scale', 'N/A')}\n")
        f.write(f"扩散步数: {config.get('num_steps', 'N/A')}\n")
        f.write(f"重叠时长: {config.get('overlap_dur', 'N/A')} 秒\n")
        f.write(f"重绘次数: {config.get('repaint_n', 'N/A')}\n")
        f.write(f"质量控制: 禁用\n")
        f.write(f"质量分数: N/A\n")
        f.write(f"模型检查点: {config.get('ckpt_path', 'N/A')}\n")
        f.write(f"输出目录: {output_dir}\n")

def main():
    parser = argparse.ArgumentParser(description='无质量控制的推理脚本')
    parser.add_argument('--config-name', type=str, default='dit', help='配置名称')
    parser.add_argument('--ckpt-path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--task', type=str, default='partial_gen', help='任务类型')
    parser.add_argument('--given-wav-path', type=str, required=True, help='给定音频路径')
    parser.add_argument('--text-prompt', type=str, required=True, help='文本提示')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--num-steps', type=int, default=100, help='扩散步数')
    parser.add_argument('--cfg-scale', type=float, default=10.0, help='CFG缩放')
    parser.add_argument('--overlap-dur', type=float, default=6.0, help='重叠时长')
    parser.add_argument('--repaint-n', type=int, default=1, help='重绘次数')
    parser.add_argument('--segment-duration', type=float, default=10.0, help='片段时长')
    parser.add_argument('--random-segment', action='store_true', help='随机片段截取')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    print("=" * 60)
    print("无质量控制的推理脚本")
    print("=" * 60)
    print(f"配置名称: {args.config_name}")
    print(f"模型检查点: {args.ckpt_path}")
    print(f"任务类型: {args.task}")
    print(f"给定音频: {args.given_wav_path}")
    print(f"文本提示: {args.text_prompt}")
    print(f"输出目录: {args.output_dir}")
    print(f"扩散步数: {args.num_steps}")
    print(f"CFG缩放: {args.cfg_scale}")
    print(f"重叠时长: {args.overlap_dur}")
    print(f"重绘次数: {args.repaint_n}")
    print(f"片段时长: {args.segment_duration}")
    print(f"随机片段: {args.random_segment}")
    print(f"设备: {args.device}")
    print(f"随机种子: {args.seed}")
    print("质量控制: 禁用")
    print("")
    
    # 检查文件
    if not os.path.exists(args.ckpt_path):
        print(f"错误: 模型检查点不存在: {args.ckpt_path}")
        return 1
    
    if not os.path.exists(args.given_wav_path):
        print(f"错误: 给定音频文件不存在: {args.given_wav_path}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    try:
        config = OmegaConf.load(f"configs/{args.config_name}.yaml")
    except Exception as e:
        print(f"错误: 无法加载配置文件: {e}")
        return 1
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载模型...")
    try:
        model = create_mgeldm_from_config(config.model)
        
        # 加载检查点
        if config.autoencoder_ckpt_path:
            print(f"加载自编码器权重: {config.autoencoder_ckpt_path}")
            model.pretransform.load_state_dict(
                load_ckpt_state_dict(config.autoencoder_ckpt_path),
                strict=True
            )
        
        # 加载扩散模型权重
        print(f"加载扩散模型权重: {args.ckpt_path}")
        model.load_state_dict(load_ckpt_state_dict(args.ckpt_path), strict=False)
        
        model.eval()
        print("✓ 模型加载完成")
        
    except Exception as e:
        print(f"错误: 模型加载失败: {e}")
        return 1
    
    # 创建推理包装器
    print("正在创建推理包装器...")
    try:
        task_wrapper = InferenceTaskWrapper(
            model=model,
            segment_length_trained=config.model.segment_length,
            timestep_eps=config.model.timestep_eps,
            device=device
        )
        print("✓ 推理包装器创建完成")
        
    except Exception as e:
        print(f"错误: 推理包装器创建失败: {e}")
        return 1
    
    # 加载给定音频
    print("正在加载给定音频...")
    try:
        given_wav = load_audio_file(args.given_wav_path, config.model.sample_rate)
        print(f"✓ 音频加载完成，形状: {given_wav.shape}")
        
    except Exception as e:
        print(f"错误: 音频加载失败: {e}")
        return 1
    
    # 执行推理
    print("开始推理...")
    try:
        if args.task == "partial_gen":
            # 部分生成
            output = task_wrapper.partial_generation(
                given_wav=given_wav,
                text_cond_src=args.text_prompt,
                overlap_dur=args.overlap_dur,
                cfg_scale=args.cfg_scale,
                num_timesteps=args.num_steps,
                repaint_n=args.repaint_n,
                verbose=True,
                return_full_output=True,
                # 无质量控制参数
                quality_score=None,
                enable_quality_control=False
            )
            
        elif args.task == "total_gen":
            # 完全生成
            output = task_wrapper.total_mixture_generation(
                text_conds_mix=[args.text_prompt],
                audio_dur=args.segment_duration,
                overlap_dur=args.overlap_dur,
                cfg_scale=args.cfg_scale,
                num_timesteps=args.num_steps,
                repaint_n=args.repaint_n,
                verbose=True,
                return_submix_src=True,
                # 无质量控制参数
                quality_score=None
            )
            
        else:
            print(f"错误: 不支持的任务类型: {args.task}")
            return 1
        
        print("✓ 推理完成")
        
    except Exception as e:
        print(f"错误: 推理失败: {e}")
        return 1
    
    # 保存结果
    print("正在保存结果...")
    try:
        import soundfile as sf
        
        # 保存生成的混合音轨
        if 'generated_mixture' in output:
            mix_wav = output['generated_mixture']
            mix_path = os.path.join(args.output_dir, "gen_mix.wav")
            sf.write(mix_path, mix_wav.cpu().numpy().T, config.model.sample_rate)
            print(f"✓ 混合音轨已保存: {mix_path}")
        
        # 保存生成的源音轨
        if 'generated_sources' in output:
            for i, src_wav in enumerate(output['generated_sources']):
                src_path = os.path.join(args.output_dir, f"gen_src_{i+1}.wav")
                sf.write(src_path, src_wav.cpu().numpy().T, config.model.sample_rate)
                print(f"✓ 源音轨 {i+1} 已保存: {src_path}")
        
        # 保存生成信息
        config_dict = {
            'task': args.task,
            'given_wav_path': args.given_wav_path,
            'text_prompt': args.text_prompt,
            'gen_audio_dur': args.segment_duration,
            'cfg_scale': args.cfg_scale,
            'num_steps': args.num_steps,
            'overlap_dur': args.overlap_dur,
            'repaint_n': args.repaint_n,
            'ckpt_path': args.ckpt_path
        }
        save_generation_info(args.output_dir, config_dict)
        
        print("✓ 结果保存完成")
        
    except Exception as e:
        print(f"错误: 结果保存失败: {e}")
        return 1
    
    print("")
    print("🎉 推理完成！")
    print(f"输出目录: {args.output_dir}")
    print("生成的文件:")
    for file in os.listdir(args.output_dir):
        if file.endswith('.wav'):
            print(f"- {file}")
    
    return 0

if __name__ == "__main__":
    exit(main())
