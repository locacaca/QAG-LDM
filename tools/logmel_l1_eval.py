"""
单条 Mel MSE 计算工具，供 batch_src_extract_quality.sh 每生成一条后立即调用。

论文参数：SR=16kHz, n_fft=1024, hop=160, n_mels=64, 固定 10.24s 片段，
直接用 power mel spectrogram（不取 log）计算 MSE，输出维度 64×1024。
"""
import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf

# ---------- 论文对齐参数 ----------
SR = 16000
N_FFT = 1024
HOP_LENGTH = 160
N_MELS = 64
SEG_SAMPLES = int(10.24 * SR)   # 163840 samples → 1024 frames


def load_mono_tensor(path: str, target_sr: int = SR) -> torch.Tensor:
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    t = torch.from_numpy(x.astype("float32"))
    if sr != target_sr:
        t = torchaudio.functional.resample(t.unsqueeze(0), sr, target_sr).squeeze(0)
    return t


def mix_sources_tensor(paths: List[str], target_sr: int = SR) -> torch.Tensor:
    """Sum stems then peak-normalise."""
    if not paths:
        return torch.zeros(SEG_SAMPLES)
    tensors = [load_mono_tensor(p, target_sr) for p in paths]
    max_len = max(t.shape[0] for t in tensors)
    mix = torch.zeros(max_len)
    for t in tensors:
        mix[:t.shape[0]] += t
    peak = mix.abs().max()
    if peak > 1e-8:
        mix = mix / peak
    return mix


def peak_normalize(wav: torch.Tensor) -> torch.Tensor:
    peak = wav.abs().max()
    if peak > 1e-8:
        return wav / peak
    return wav


def fix_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    """截取或补零到固定长度。"""
    if wav.shape[0] >= length:
        return wav[:length]
    pad = torch.zeros(length)
    pad[:wav.shape[0]] = wav
    return pad


def best_align_start(ref: torch.Tensor, query: torch.Tensor) -> int:
    """互相关对齐：找 ref 中与 query 最匹配的起始位置。"""
    n = query.shape[0]
    if ref.shape[0] <= n:
        return 0
    q = query - query.mean()
    r = ref - ref.mean()
    corr = np.correlate(r.numpy(), q.numpy(), mode='valid')
    return int(np.argmax(corr))


_mel_transform = None

def get_mel_transform(device: str) -> T.MelSpectrogram:
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = T.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        ).to(device)
    return _mel_transform


def extract_mel(wav: torch.Tensor, device: str) -> torch.Tensor:
    """(T,) → (64, 1024) power mel spectrogram on device."""
    w = fix_length(wav, SEG_SAMPLES).unsqueeze(0).to(device)
    mel = get_mel_transform(device)(w)
    return mel.squeeze(0)   # (64, frames)


def mel_mse(gen_mel: torch.Tensor, ref_mel: torch.Tensor) -> float:
    t = min(gen_mel.shape[1], ref_mel.shape[1])
    return float(((gen_mel[:, :t] - ref_mel[:, :t]) ** 2).mean().item())


def read_stems_for_uid(manifest_path: str, uid: str, instrument: str) -> List[str]:
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get('uid') == uid:
                ref_sources = d.get('ref_sources', {})
                for key in ref_sources:
                    if key.lower() == instrument.lower():
                        return ref_sources[key]
    return []


def append_csv(csv_path: str, uid: str, quality: str, instrument: str,
               out_dir: str, dist: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    header_needed = not os.path.exists(csv_path)
    with open(csv_path, 'a', encoding='utf-8') as fw:
        if header_needed:
            fw.write('uid,quality,instrument,output_dir,mel_mse\n')
        fw.write(f"{uid},{quality},{instrument},{out_dir},{dist}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compute Mel MSE between generated source and reference stem (paper-aligned).'
    )
    parser.add_argument('--uid', required=True)
    parser.add_argument('--quality', required=True)
    parser.add_argument('--instrument', required=True)
    parser.add_argument('--out_dir', required=True,
                        help='Path to output_XXXX directory containing gen_src.wav')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--results_csv', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    gen_src_path = os.path.join(args.out_dir, 'gen_src.wav')
    if not os.path.isfile(gen_src_path):
        print(f'Missing gen_src.wav in {args.out_dir}, skip.')
        return

    stems = read_stems_for_uid(args.manifest, args.uid, args.instrument)
    if not stems:
        print(f'No stems found for uid={args.uid} instrument={args.instrument}, skip.')
        return

    device = args.device if torch.cuda.is_available() else 'cpu'
    # 若传入 cuda:N 形式，转换为 cuda（CUDA_VISIBLE_DEVICES 已限制可见设备）
    if device.startswith('cuda:'):
        device = 'cuda'

    gen_wav = peak_normalize(load_mono_tensor(gen_src_path, SR))
    ref_full = mix_sources_tensor(stems, SR)  # 内部已 peak normalize

    # 互相关对齐后截取固定 10.24s
    align_start = best_align_start(ref_full, gen_wav)
    ref_seg = fix_length(ref_full[align_start:], SEG_SAMPLES)
    gen_seg = fix_length(gen_wav, SEG_SAMPLES)

    gen_mel = extract_mel(gen_seg, device)   # (64, 1024)
    ref_mel = extract_mel(ref_seg, device)   # (64, 1024)

    dist = mel_mse(gen_mel, ref_mel)
    append_csv(args.results_csv, args.uid, args.quality, args.instrument,
               args.out_dir, dist)
    print(f'Mel MSE [{args.uid}] [{args.instrument}] q={args.quality} '
          f'align_start={align_start/SR:.2f}s dist={dist:.6f}')


if __name__ == '__main__':
    main()
