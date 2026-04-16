import argparse
import csv
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot attention heatmaps from all_layers_attn*.pt."
    )
    parser.add_argument(
        "--pt",
        type=str,
        default=None,
        help="Path to an all_layers_attn*.pt file. Defaults to all matching files in the current directory.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save figures. Defaults to <pt_stem>_heatmaps.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional upper bound of the color scale. Defaults to max(attn_map) + 0.01.",
    )
    parser.add_argument(
        "--quality-tokens",
        type=int,
        default=1,
        help="Number of prepended quality tokens. Default matches current model design.",
    )
    parser.add_argument(
        "--timestep-tokens",
        type=int,
        default=1,
        help="Number of prepended timestep tokens. Default matches current model design.",
    )
    parser.add_argument(
        "--sink-topk",
        type=int,
        default=8,
        help="How many sink candidate key positions to print.",
    )
    parser.add_argument(
        "--prefix-order",
        type=str,
        default="timestep_first",
        choices=["timestep_first", "quality_first"],
        help="Interpretation of prepended token order.",
    )
    parser.add_argument(
        "--plot-loss",
        action="store_true",
        help="Also plot loss curves from loss_curve*.csv in the current directory or beside --pt.",
    )
    return parser.parse_args()


def resolve_pt_paths(pt_arg: Optional[str]) -> List[Path]:
    if pt_arg is not None:
        path = Path(pt_arg)
        if not path.exists():
            raise FileNotFoundError(f"PT file not found: {path}")
        return [path]

    candidates = sorted(Path(".").glob("all_layers_attn*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = sorted(Path(".").glob("last_layers_attn*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            "No all_layers_attn*.pt or last_layers_attn*.pt file found in the current directory. "
            "Use --pt to specify one."
        )
    return candidates


def resolve_loss_csv_paths(pt_arg: Optional[str]) -> List[Path]:
    search_dir = Path(pt_arg).parent if pt_arg is not None else Path(".")
    candidates = sorted(search_dir.glob("loss_curve*.csv"), key=lambda p: p.stat().st_mtime)
    return candidates


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().float().tolist(), dtype=np.float32)


def normalize_attn_map(attn_tensor: torch.Tensor) -> np.ndarray:
    if not isinstance(attn_tensor, torch.Tensor):
        raise ValueError(f"Expected attn_map to be a torch.Tensor, got {type(attn_tensor)}")

    if attn_tensor.dim() == 2:
        return tensor_to_numpy(attn_tensor)

    if attn_tensor.dim() == 3:
        # Backward compatibility for old files storing [head, query, key].
        return tensor_to_numpy(attn_tensor.mean(dim=0))

    raise ValueError(
        f"Expected attn_map to be a 2D [query, key] tensor or 3D [head, query, key] tensor, "
        f"got shape={tuple(attn_tensor.shape)}"
    )


def plot_heatmap(attn_map: np.ndarray, title: str, save_path: Path, dpi: int, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_map, aspect="auto", cmap="viridis", origin="lower", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("key")
    ax.set_ylabel("query")
    fig.colorbar(im, ax=ax, pad=0.015)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(csv_path: Path, outdir: Path, dpi: int) -> None:
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    steps = [int(float(row["step"])) for row in rows if row.get("step") not in (None, "")]
    value_keys = [
        key for key in rows[0].keys()
        if key != "step" and any(row.get(key) not in (None, "") for row in rows)
    ]

    if not steps or not value_keys:
        return

    for key in value_keys:
        values = []
        key_steps = []
        for row in rows:
            raw_step = row.get("step")
            raw_value = row.get(key)
            if raw_step in (None, "") or raw_value in (None, ""):
                continue
            key_steps.append(int(float(raw_step)))
            values.append(float(raw_value))

        if not values:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(key_steps, values, linewidth=1.5)
        ax.set_title(key)
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(outdir / f"{csv_path.stem}_{key}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def print_row_stats(attn_map: np.ndarray, layer: int, branch: str, step: int, top_k: int = 10) -> None:
    num_queries = attn_map.shape[0]
    top_k = min(top_k, num_queries)
    print(f"[RowStats] layer={layer}, branch={branch}, step={step}, showing first {top_k} queries")
    for query_idx in range(top_k):
        row = attn_map[query_idx]
        row_sum = float(row.sum())
        row_max = float(row.max())
        row_min = float(row.min())
        print(
            f"  query={query_idx:02d} "
            f"row_sum={row_sum:.6f} "
            f"row_max={row_max:.6f} "
            f"row_min={row_min:.6f}"
        )


def describe_token_position(
    token_idx: int,
    num_quality_tokens: int,
    num_timestep_tokens: int,
    prefix_order: str,
) -> str:
    if prefix_order == "quality_first":
        if token_idx < num_quality_tokens:
            return f"quality[{token_idx}]"

        timestep_start = num_quality_tokens
        timestep_end = timestep_start + num_timestep_tokens
        if token_idx < timestep_end:
            return f"timestep[{token_idx - timestep_start}]"

        audio_idx = token_idx - timestep_end
        return f"audio[{audio_idx}]"

    if token_idx < num_timestep_tokens:
        return f"timestep[{token_idx}]"

    quality_start = num_timestep_tokens
    quality_end = quality_start + num_quality_tokens
    if token_idx < quality_end:
        return f"quality[{token_idx - quality_start}]"

    audio_idx = token_idx - quality_end
    return f"audio[{audio_idx}]"


def print_sink_candidates(
    attn_map: np.ndarray,
    layer: int,
    step: int,
    num_quality_tokens: int,
    num_timestep_tokens: int,
    prefix_order: str,
    top_k: int,
) -> None:
    key_scores = attn_map.mean(axis=0)
    sorted_indices = np.argsort(key_scores)[::-1]
    top_k = min(top_k, key_scores.shape[0])

    print(
        f"[SinkCandidates] layer={layer}, step={step}, "
        f"quality_tokens={num_quality_tokens}, timestep_tokens={num_timestep_tokens}"
    )
    for rank, token_idx in enumerate(sorted_indices[:top_k], start=1):
        score = float(key_scores[token_idx])
        token_desc = describe_token_position(
            int(token_idx),
            num_quality_tokens=num_quality_tokens,
            num_timestep_tokens=num_timestep_tokens,
            prefix_order=prefix_order,
        )
        print(
            f"  top{rank:02d} "
            f"key_idx={int(token_idx):03d} "
            f"type={token_desc} "
            f"mean_col_attn={score:.6f}"
        )

    inspect_prefix = min(num_quality_tokens + num_timestep_tokens + 8, key_scores.shape[0])
    print(f"[PrefixTokens] layer={layer}, step={step}, first {inspect_prefix} key positions")
    for token_idx in range(inspect_prefix):
        token_desc = describe_token_position(
            token_idx,
            num_quality_tokens=num_quality_tokens,
            num_timestep_tokens=num_timestep_tokens,
            prefix_order=prefix_order,
        )
        print(
            f"  key_idx={token_idx:03d} "
            f"type={token_desc} "
            f"mean_col_attn={float(key_scores[token_idx]):.6f}"
        )


def main():
    args = parse_args()
    pt_paths = resolve_pt_paths(args.pt)
    root_outdir = Path(args.outdir) if args.outdir else None

    for pt_path in pt_paths:
        payload = torch.load(pt_path, map_location="cpu")

        if "data" not in payload or not isinstance(payload["data"], list):
            raise ValueError(f"Unexpected PT structure in {pt_path}")

        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        prefix_order = args.prefix_order
        payload_prefix_order = config.get("prefix_order", None)
        if payload_prefix_order == ["quality", "timestep", "audio"]:
            prefix_order = "quality_first"
        elif payload_prefix_order == ["timestep", "quality", "audio"]:
            prefix_order = "timestep_first"

        outdir = root_outdir if root_outdir else pt_path.with_suffix("")
        outdir = outdir.parent / f"{outdir.name}_heatmaps"
        if len(pt_paths) > 1:
            outdir = outdir / pt_path.stem
        outdir.mkdir(parents=True, exist_ok=True)

        records = sorted(payload["data"], key=lambda record: (record["step"], record["layer"]))
        for record in records:
            layer = record["layer"]
            step = record["step"]
            branch = record["branch_name"]
            attn_tensor = record["attn_map"]

            attn_map = normalize_attn_map(attn_tensor)
            print_row_stats(attn_map, layer=layer, branch=branch, step=step, top_k=10)
            print_sink_candidates(
                attn_map,
                layer=layer,
                step=step,
                num_quality_tokens=args.quality_tokens,
                num_timestep_tokens=args.timestep_tokens,
                prefix_order=prefix_order,
                top_k=args.sink_topk,
            )

            layer_dir = outdir / f"layer_{layer:02d}_{branch}_step_{step}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            vmax = args.vmax if args.vmax is not None else float(attn_map.max()) + 0.001
            heatmap_path = layer_dir / f"layer_{layer:02d}_{branch}_heatmap.png"
            plot_heatmap(
                attn_map,
                title=f"Layer {layer}",
                save_path=heatmap_path,
                dpi=args.dpi,
                vmax=vmax,
            )

        print(f"Saved heatmaps to: {outdir}")

    if args.plot_loss:
        loss_csv_paths = resolve_loss_csv_paths(args.pt)
        for csv_path in loss_csv_paths:
            outdir = root_outdir if root_outdir else csv_path.with_suffix("")
            outdir = outdir.parent / f"{outdir.name}_plots"
            if len(loss_csv_paths) > 1:
                outdir = outdir / csv_path.stem
            outdir.mkdir(parents=True, exist_ok=True)
            plot_loss_curve(csv_path, outdir=outdir, dpi=args.dpi)
            print(f"Saved loss plots to: {outdir}")


if __name__ == "__main__":
    main()
