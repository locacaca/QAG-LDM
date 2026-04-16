import argparse
import json
import os
from collections import defaultdict
from statistics import mean, median
from typing import Dict, List, Tuple

import numpy as np

from compute_fad_partial_gen import (
    PANNsEmbedder,
    _load_mono,
    compute_fad,
    _get_device,
)


def _list_track_dirs(exp_root: str) -> List[str]:
    if not os.path.isdir(exp_root):
        raise FileNotFoundError(f"实验根目录不存在: {exp_root}")
    return sorted(
        [
            d
            for d in os.listdir(exp_root)
            if d.startswith("Track") and os.path.isdir(os.path.join(exp_root, d))
        ]
    )


def _collect_audio_pairs(
    exp_root: str, track_id: str, instrument: str
) -> Tuple[str, str]:
    inst_root = os.path.join(exp_root, track_id, instrument)
    gen_path = os.path.join(inst_root, "inpaint_000", f"{instrument}.wav")
    ref_path = os.path.join(inst_root, "original", f"{instrument}.wav")
    if not os.path.isfile(gen_path):
        raise FileNotFoundError(f"生成音频缺失: {gen_path}")
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"原始音频缺失: {ref_path}")
    return gen_path, ref_path


def _embed_audio(path: str, sample_rate: int, embedder: PANNsEmbedder) -> np.ndarray:
    audio = _load_mono(path, sample_rate)
    return embedder(audio)


def compute_fad_for_instrument(
    exp_root: str,
    instrument: str,
    sample_rate: int,
    embedder: PANNsEmbedder,
    allow_missing: bool,
) -> Dict:
    track_dirs = _list_track_dirs(exp_root)
    per_track = []
    errors: Dict[str, str] = {}
    for track in track_dirs:
        try:
            gen_path, ref_path = _collect_audio_pairs(exp_root, track, instrument)
            gen_emb = _embed_audio(gen_path, sample_rate, embedder)
            ref_emb = _embed_audio(ref_path, sample_rate, embedder)
            fad_val = float(compute_fad(gen_emb, ref_emb))
            per_track.append(
                {
                    "track_id": track,
                    "fad": fad_val,
                    "gen_path": gen_path,
                    "ref_path": ref_path,
                }
            )
        except FileNotFoundError as fnf:
            if allow_missing:
                errors[track] = str(fnf)
                continue
            raise
        except Exception as exc:
            errors[track] = str(exc)
    stats = _summarize_fad([p["fad"] for p in per_track])
    return {
        "instrument": instrument,
        "num_tracks": len(per_track),
        "per_track": per_track,
        "stats": stats,
        "errors": errors,
    }


def _summarize_fad(values: List[float]) -> Dict:
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    avg = mean(values)
    med = median(values)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "mean": avg,
        "median": med,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def save_json(output_dir: str, instrument: str, payload: Dict):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"fad_{instrument}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量计算部分生成音轨的 FAD，并输出统计结果"
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default="/app/data/code/MSDM/multi-source-diffusion-models-main/exp/inpaint_all",
        help="TrackXXXX 的根目录",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        nargs="+",
        default=["bass", "drums", "guitar", "piano"],
        help="需要计算的部分生成音轨类型",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fad_partial_results",
        help="保存 JSON 的目录",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=48000, help="重采样采样率"
    )
    parser.add_argument(
        "--panns_checkpoint",
        type=str,
        default="/root/panns_data/Cnn14_mAP=0.431.pth",
        help="PANNs 权重路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备，默认自动检测",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="忽略缺失音频，记录错误信息",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or _get_device()
    embedder = PANNsEmbedder(
        sr=args.sample_rate, device=device, checkpoint=args.panns_checkpoint
    )
    summary = defaultdict(dict)
    for inst in args.instruments:
        result = compute_fad_for_instrument(
            exp_root=args.exp_root,
            instrument=inst,
            sample_rate=args.sample_rate,
            embedder=embedder,
            allow_missing=args.allow_missing,
        )
        output_path = save_json(args.output_dir, inst, result)
        summary[inst]["stats"] = result["stats"]
        summary[inst]["num_tracks"] = result["num_tracks"]
        summary[inst]["json_path"] = output_path
        summary[inst]["num_errors"] = len(result["errors"])

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


