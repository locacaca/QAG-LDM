import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np

from compute_fad_partial_gen import PANNsEmbedder, compute_fad, _load_mono


def _list_subdirs(root: str, prefix: Optional[str] = None) -> List[str]:
    if not os.path.isdir(root):
        raise NotADirectoryError(f"目录不存在: {root}")
    dirs = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if prefix and not name.startswith(prefix):
            continue
        dirs.append(full)
    if not dirs:
        raise RuntimeError(f"在 {root} 下没有找到符合前缀 {prefix!r} 的目录")
    return dirs


def _mix_stems(stem_dir: str, stems: List[str], sample_rate: int) -> Tuple[np.ndarray, List[str]]:
    wavs = []
    loaded = []
    for stem in stems:
        path = os.path.join(stem_dir, f"{stem}.wav")
        if not os.path.isfile(path):
            print(f"[WARN] {stem_dir} 缺少 {stem}.wav，已跳过该stem", flush=True)
            continue
        audio = _load_mono(path, sample_rate)
        wavs.append(audio)
        loaded.append(path)
    if not wavs:
        raise RuntimeError(f"{stem_dir} 下未加载到任何stem")
    max_len = max(len(w) for w in wavs)
    mix = np.zeros(max_len, dtype=np.float32)
    for w in wavs:
        if len(w) < max_len:
            w = np.pad(w, (0, max_len - len(w)))
        mix += w
    mix /= len(wavs)
    return mix, loaded


def _build_embeddings(root: str, prefix: Optional[str], stems: List[str], sample_rate: int, embedder: PANNsEmbedder) -> Tuple[np.ndarray, List[dict]]:
    embeddings = []
    metadata = []
    for subdir in _list_subdirs(root, prefix):
        try:
            mix_audio, used_paths = _mix_stems(subdir, stems, sample_rate)
            emb = embedder(mix_audio)
            embeddings.append(emb)
            metadata.append({"dir": subdir, "stems": used_paths})
        except Exception as exc:
            print(f"[WARN] 处理目录 {subdir} 失败: {exc}，已跳过", flush=True)
            continue
    if not embeddings:
        raise RuntimeError(f"{root} 下没有成功生成任何 embedding")
    return np.stack(embeddings, axis=0), metadata


def parse_args():
    parser = argparse.ArgumentParser(description="计算生成样本 mix 与 slakh2100 测试集 mix 之间的 FAD")
    parser.add_argument("--sample_root", type=str, required=True, help="包含 sample_XXXX 目录的根路径")
    parser.add_argument("--dataset_root", type=str, required=True, help="包含 Track0XXXX 目录的根路径")
    parser.add_argument("--sample_prefix", type=str, default="sample_", help="筛选生成目录的前缀")
    parser.add_argument("--dataset_prefix", type=str, default="Track0", help="筛选测试集目录的前缀")
    parser.add_argument("--stems", type=str, nargs="+", default=["bass", "drums", "guitar", "piano"], help="需要混合的stem名称")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--panns_checkpoint", type=str, default="/root/panns_data/Cnn14_mAP=0.431.pth")
    parser.add_argument("--output_json", type=str, default=None, help="保存结果的json路径")
    return parser.parse_args()


def main():
    args = parse_args()
    embedder = PANNsEmbedder(sr=args.sample_rate, device=args.device, checkpoint=args.panns_checkpoint)

    sample_embs, sample_meta = _build_embeddings(
        root=args.sample_root,
        prefix=args.sample_prefix,
        stems=args.stems,
        sample_rate=args.sample_rate,
        embedder=embedder,
    )
    dataset_embs, dataset_meta = _build_embeddings(
        root=args.dataset_root,
        prefix=args.dataset_prefix,
        stems=args.stems,
        sample_rate=args.sample_rate,
        embedder=embedder,
    )

    fad_value = compute_fad(sample_embs, dataset_embs)
    result = {
        "sample_root": args.sample_root,
        "dataset_root": args.dataset_root,
        "sample_count": len(sample_meta),
        "dataset_count": len(dataset_meta),
        "stems": args.stems,
        "fad": float(fad_value),
        "sample_entries": sample_meta,
        "dataset_entries": dataset_meta,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

