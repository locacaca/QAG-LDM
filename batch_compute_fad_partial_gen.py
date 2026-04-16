import argparse
import json
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List

from compute_fad_partial_gen import PANNsEmbedder, compute_fad_for_pair_eval_style


def load_tasks(jsonl_path: str) -> List[Dict]:
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def find_gen_mix_path(task_output_dir: str) -> str:
    # 支持嵌套子目录结构: 查找 gen_src.wav（用于 partial gen）
    import glob
    # 查找所有可能的 gen_src.wav 位置
    patterns = [
        os.path.join(task_output_dir, "gen_src.wav"),
        os.path.join(task_output_dir, "**", "gen_src.wav"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return ""


def main():
    parser = argparse.ArgumentParser(description="Batch compute FAD for partial generation outputs")
    parser.add_argument("--tasks_jsonl", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True, help="如 ./outputs_batch_partial_gen_quality")
    parser.add_argument("--quality_scores", type=float, nargs="+", default=[0.1, 0.9])
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--panns_checkpoint", type=str, default="/root/panns_data/Cnn14_mAP=0.431.pth")
    parser.add_argument("--device", type=str, default="auto", help="设备，可以是 'auto', 'cuda', 'cpu', 或 'cuda:N'")
    parser.add_argument("--output_summary", type=str, default="./fad_summary.json")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks_jsonl)
    panns_sr = 32000
    if args.sample_rate != panns_sr:
        print(f"[WARN] 忽略 --sample_rate={args.sample_rate}，PANNs 评估固定使用 {panns_sr} Hz 以匹配权重", flush=True)

    # 解析设备参数，支持 cuda:N 格式
    device = args.device
    if device == "auto":
        device = "cuda"
    if device.startswith("cuda:") and "," not in device:
        # cuda:N 格式，直接使用
        pass
    elif device == "cuda":
        # 检查可用的 CUDA 设备数量
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = f"cuda:0"  # 默认使用 cuda:0

    embedder = PANNsEmbedder(sr=panns_sr, device=device, checkpoint=args.panns_checkpoint)

    results = []
    success = 0
    failed = 0

    for q in args.quality_scores:
        q_dir = os.path.join(args.output_root, f"quality_{q}")
        for t in tasks:
            uid = t.get("uid", "")
            # 清理 UID 中的异常字符
            import re
            uid_clean = re.sub(r'[{}"]', '', uid).strip()
            given_wav_path = t.get("given_wav_path")
            unknown_files = t.get("unknown_audio_files", [])
            task_dir = os.path.join(q_dir, uid_clean)
            if not os.path.isdir(task_dir):
                failed += 1
                results.append({
                    "uid": uid_clean,
                    "quality_score": q,
                    "error": f"Task dir not found: {task_dir}"
                })
                continue

            out_json = os.path.join(task_dir, "fad_result.json")
            if args.skip_existing and os.path.isfile(out_json):
                try:
                    with open(out_json, "r", encoding="utf-8") as f:
                        prev = json.load(f)
                    prev["quality_score"] = q
                    results.append(prev)
                    success += 1
                except Exception as e:
                    # 如果读取失败，继续处理
                    pass
                continue

            gen_mix = find_gen_mix_path(task_dir)
            if not gen_mix:
                failed += 1
                results.append({
                    "uid": uid_clean,
                    "quality_score": q,
                    "error": f"gen_src.wav not found under {task_dir}"
                })
                continue

            try:
                fad = compute_fad_for_pair_eval_style(
                    gen_audio_path=gen_mix,
                    ref_audio_paths=unknown_files,
                    embedder=embedder,
                )
                rec = {
                    "uid": uid_clean,
                    "quality_score": q,
                    "gen_mix_path": gen_mix,
                    "given_wav_path": given_wav_path,
                    "unknown_audio_files": unknown_files,
                    "results": {"FAD": float(fad)},
                }
                os.makedirs(task_dir, exist_ok=True)
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
                results.append(rec)
                success += 1
            except Exception as e:
                failed += 1
                results.append({
                    "uid": uid_clean,
                    "quality_score": q,
                    "error": str(e)
                })

    # 汇总统计
    by_quality: Dict[float, List[float]] = defaultdict(list)
    for r in results:
        if "results" in r and "FAD" in r["results"]:
            by_quality[r["quality_score"]].append(r["results"]["FAD"])

    stats_by_quality = {}
    for q, vals in by_quality.items():
        if len(vals) == 0:
            continue
        stats_by_quality[str(q)] = {
            "count": len(vals),
            "mean": mean(vals),
            "min": min(vals),
            "max": max(vals),
            "std": pstdev(vals) if len(vals) > 1 else 0.0,
        }

    summary = {
        "total_tasks": len(tasks),
        "total_quality_scores": len(args.quality_scores),
        "successful": success,
        "failed": failed,
        "statistics_by_quality": stats_by_quality,
        "items": results,
    }

    out_dir = os.path.dirname(args.output_summary)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


