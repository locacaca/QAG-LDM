import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import soundfile as sf
import librosa


SR = 16000


def load_and_resample(path: str, target_sr: int = SR) -> np.ndarray:
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    return x.astype(np.float32)


def mix_sources(paths: List[str], target_sr: int = SR) -> np.ndarray:
    if not paths:
        return np.zeros(1, dtype=np.float32)
    audios = [load_and_resample(p, target_sr) for p in paths]
    max_len = max(len(a) for a in audios)
    mix = np.zeros(max_len, dtype=np.float32)
    for a in audios:
        mix[:len(a)] += a
    mix /= max(len(audios), 1)
    return mix


def best_align_segment(ref_full: np.ndarray, query_seg: np.ndarray) -> Tuple[np.ndarray, int]:
    n = len(query_seg)
    if len(ref_full) <= n:
        return ref_full[:n], 0
    q = query_seg - query_seg.mean()
    r = ref_full - ref_full.mean()
    corr = np.correlate(r, q, mode='valid')
    start = int(np.argmax(corr))
    return ref_full[start:start + n], start


def logmel(x: np.ndarray, sr: int = SR, n_mels: int = 64, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    return np.log(1e-6 + S)


def l1(a: np.ndarray, b: np.ndarray) -> float:
    m = min(a.shape[1], b.shape[1])
    if m <= 0:
        return float('nan')
    return float(np.mean(np.abs(a[:, :m] - b[:, :m])))


def read_stems_for_uid(manifest_path: str, uid: str, instrument: str) -> List[str]:
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get('uid') == uid:
                return d.get('ref_sources', {}).get(instrument, [])
    return []


def append_csv(csv_path: str, uid: str, quality: str, instrument: str, out_dir: str, dist: float) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header_needed = not os.path.exists(csv_path)
    with open(csv_path, 'a', encoding='utf-8') as fw:
        if header_needed:
            fw.write('uid,quality,instrument,output_dir,logmel_l1\n')
        fw.write(f"{uid},{quality},{instrument},{out_dir},{dist}\n")


def main():
    parser = argparse.ArgumentParser(description='Compute Log-Mel L1 distance for a single output directory.')
    parser.add_argument('--uid', required=True)
    parser.add_argument('--quality', required=True)
    parser.add_argument('--instrument', required=True)
    parser.add_argument('--out_dir', required=True, help='Path to output_XXXX directory containing gen_src.wav and given_mix.wav')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--results_csv', required=True)
    args = parser.parse_args()

    gen_src_path = os.path.join(args.out_dir, 'gen_src.wav')
    given_mix_path = os.path.join(args.out_dir, 'given_mix.wav')
    if not (os.path.isfile(gen_src_path) and os.path.isfile(given_mix_path)):
        print('Missing generated files, skip.')
        return

    stems = read_stems_for_uid(args.manifest, args.uid, args.instrument)
    if not stems:
        print('No stems found for given uid/instrument, skip.')
        return

    gen_src = load_and_resample(gen_src_path, SR)
    given_mix = load_and_resample(given_mix_path, SR)
    ref_full = mix_sources(stems, SR)
    ref_seg, _ = best_align_segment(ref_full, given_mix)

    dist = l1(logmel(gen_src, SR), logmel(ref_seg, SR))
    append_csv(args.results_csv, args.uid, args.quality, args.instrument, args.out_dir, dist)
    print(f'Log-Mel L1 for {args.uid} [{args.instrument}] @ quality={args.quality}: {dist:.6f}')


if __name__ == '__main__':
    main()


