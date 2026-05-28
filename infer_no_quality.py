#!/usr/bin/env python3
"""
无质量控制的推理脚本（适配质量控制+门控模型架构）

与新模型兼容：强制 enable_quality_control=False, enable_attention_gating=False，
使用 null quality token，适合加载 baseline checkpoint。
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
import soundfile as sf
from omegaconf import OmegaConf

from multi_track_stable_audio.models.factory import create_mgeldm_from_config
from multi_track_stable_audio.inference.task_wrapper import InferenceTaskWrapper
from multi_track_stable_audio.utils import load_ckpt_state_dict, to_numpy, set_seed


def proc_audio(wav: torch.Tensor, downsample_ratio: int) -> torch.Tensor:
    wav = wav.mean(dim=0, keepdim=True)
    wav_len = wav.shape[-1]
    if wav_len % downsample_ratio != 0:
        wav = wav[:, :(wav_len // downsample_ratio) * downsample_ratio]
    return wav


def main():
    parser = argparse.ArgumentParser(description='无质量控制的推理脚本')
    parser.add_argument('--config-name', type=str, default='dit')
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--task', type=str, default='partial_gen',
                        choices=['partial_gen', 'total_gen'])
    parser.add_argument('--given-wav-path', type=str, default=None)
    parser.add_argument('--text-prompt', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--gen-audio-dur', type=float, default=10.0)
    parser.add_argument('--num-steps', type=int, default=100)
    parser.add_argument('--cfg-scale', type=float, default=10.0)
    parser.add_argument('--overlap-dur', type=float, default=6.0)
    parser.add_argument('--repaint-n', type=int, default=1)
    parser.add_argument('--segment-duration', type=float, default=None)
    parser.add_argument('--random-segment', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("无质量控制的推理脚本")
    print(f"模型检查点: {args.ckpt_path}")
    print(f"任务类型: {args.task}")
    print(f"文本提示: {args.text_prompt}")
    print(f"输出目录: {args.output_dir}")
    print("质量控制: 禁用 (质量分数 = None)")
    print("=" * 60)

    # 用 Hydra compose 加载配置，正确解析跨文件插值
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize(config_path="configs", version_base=None):
        config = compose(config_name=args.config_name)

    # 强制禁用质量控制和门控，与 baseline checkpoint 匹配
    OmegaConf.set_struct(config, False)
    if "MGELDM" in config.model:
        config.model.MGELDM.enable_quality_control = False
        config.model.MGELDM.enable_attention_gating = False
    OmegaConf.set_struct(config, True)

    print(f"enable_quality_control: False")
    print(f"enable_attention_gating: False")

    # 加载模型
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = create_mgeldm_from_config(config.model)
    ckpt = load_ckpt_state_dict(args.ckpt_path)
    model.load_state_dict(ckpt, strict=True)
    del ckpt
    torch.cuda.empty_cache()
    model.prepare_for_inference(device)
    print(f"模型已加载: {args.ckpt_path}，设备: {device}")

    task_wrapper = InferenceTaskWrapper(
        model=model,
        segment_length_trained=config.model.segment_length,
        timestep_eps=config.model.timestep_eps,
        clap_ckpt_path=None,
        load_clap_audio=False,
        device=device,
    )

    func_args = dict(
        overlap_dur=args.overlap_dur,
        cfg_scale=args.cfg_scale,
        num_timesteps=args.num_steps,
        repaint_n=args.repaint_n,
        quality_score=None,
        verbose=True,
    )

    given_wav = None

    if args.task == "partial_gen":
        assert args.given_wav_path, "partial_gen 需要 --given-wav-path"
        wav, sr_ori = torchaudio.load(args.given_wav_path)
        wav = torchaudio.transforms.Resample(sr_ori, config.model.sample_rate)(wav)
        wav = proc_audio(wav, downsample_ratio=2048)  # (1, T)

        if args.segment_duration is not None and args.random_segment:
            target_samples = config.model.segment_length * 2048
            if wav.shape[1] > target_samples:
                max_start = wav.shape[1] - target_samples
                start_idx = np.random.randint(0, max_start + 1)
                wav = wav[:, start_idx:start_idx + target_samples]
                print(f"随机截取片段: {start_idx/config.model.sample_rate:.2f}s ~ "
                      f"{(start_idx + target_samples)/config.model.sample_rate:.2f}s")

        given_wav = wav.unsqueeze(0)  # (1, 1, T)
        output = task_wrapper.partial_generation(
            given_wav=given_wav,
            text_conds_src=[args.text_prompt],
            return_full_output=True,
            **func_args,
        )

    elif args.task == "total_gen":
        output = task_wrapper.total_mixture_generation(
            text_conds_mix=[args.text_prompt],
            audio_dur=args.gen_audio_dur,
            return_submix_src=True,
            **func_args,
        )

    else:
        raise ValueError(f"不支持的任务类型: {args.task}")

    # 保存生成结果
    sample_rate = config.model.sample_rate
    for key, tensor in output.items():
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 3:
            wav_np = to_numpy(tensor[0])  # (1, T)
            filepath = os.path.join(args.output_dir, f"{key}.wav")
            sf.write(filepath, wav_np[0], sample_rate)
            print(f"已保存: {filepath}")

    if given_wav is not None:
        wav_np = to_numpy(given_wav[0])  # (1, T)
        sf.write(os.path.join(args.output_dir, "given_wav.wav"), wav_np[0], sample_rate)

    with open(os.path.join(args.output_dir, "generation_info.txt"), 'w', encoding='utf-8') as f:
        f.write("生成信息 (无质量控制)\n")
        f.write(f"任务类型: {args.task}\n")
        f.write(f"给定音频: {args.given_wav_path}\n")
        f.write(f"文本提示: {args.text_prompt}\n")
        f.write(f"CFG缩放: {args.cfg_scale}\n")
        f.write(f"扩散步数: {args.num_steps}\n")
        f.write(f"重叠时长: {args.overlap_dur}\n")
        f.write(f"重绘次数: {args.repaint_n}\n")
        f.write(f"质量控制: 禁用\n")
        f.write(f"模型检查点: {args.ckpt_path}\n")

    print(f"\n推理完成。输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
