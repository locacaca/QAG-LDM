import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot timestep token attention score vs layer from attention_sink_curve*.csv or all_layers_attn*.pt."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Primary CSV path. Defaults to the latest matching CSV in the current directory unless --pt is used.",
    )
    parser.add_argument(
        "--csv2",
        type=str,
        default=None,
        help="Optional second CSV path. If provided, both curves are drawn on the same figure.",
    )
    parser.add_argument(
        "--csv3",
        type=str,
        default=None,
        help="Optional third CSV path.",
    )
    parser.add_argument(
        "--csv4",
        type=str,
        default=None,
        help="Optional fourth CSV path.",
    )
    parser.add_argument(
        "--csv5",
        type=str,
        default=None,
        help="Optional fifth CSV path.",
    )
    parser.add_argument(
        "--pt",
        type=str,
        default=None,
        help="Primary PT path. If provided, reconstructs sink scores from all_layers_attn*.pt instead of CSV.",
    )
    parser.add_argument(
        "--pt2",
        type=str,
        default=None,
        help="Optional second PT path. If provided, both PT-derived curves are drawn on the same figure.",
    )
    parser.add_argument(
        "--pt3",
        type=str,
        default=None,
        help="Optional third PT path.",
    )
    parser.add_argument(
        "--pt4",
        type=str,
        default=None,
        help="Optional fourth PT path.",
    )
    parser.add_argument(
        "--pt5",
        type=str,
        default=None,
        help="Optional fifth PT path.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path. Defaults to a name derived from the input CSV file(s).",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Optional branch filter, e.g. self or fused.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Optional exact step filter. If omitted, averages all matched records per layer.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional display name for the first curve.",
    )
    parser.add_argument(
        "--name2",
        type=str,
        default=None,
        help="Optional display name for the second curve.",
    )
    parser.add_argument(
        "--name3",
        type=str,
        default=None,
        help="Optional display name for the third curve.",
    )
    parser.add_argument(
        "--name4",
        type=str,
        default=None,
        help="Optional display name for the fourth curve.",
    )
    parser.add_argument(
        "--name5",
        type=str,
        default=None,
        help="Optional display name for the fifth curve.",
    )
    return parser.parse_args()


def resolve_csv_path(csv_arg: Optional[str]) -> Path:
    if csv_arg is not None:
        path = Path(csv_arg)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return path

    candidates = sorted(Path(".").glob("attention_sink_curve*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            "No attention_sink_curve*.csv found in the current directory. Use --csv to specify one."
        )
    return candidates[-1]


def resolve_pt_path(pt_arg: Optional[str]) -> Path:
    if pt_arg is not None:
        path = Path(pt_arg)
        if not path.exists():
            raise FileNotFoundError(f"PT file not found: {path}")
        return path

    candidates = sorted(Path(".").glob("all_layers_attn*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            "No all_layers_attn*.pt found in the current directory. Use --pt to specify one."
        )
    return candidates[-1]


def load_rows(csv_path: Path, branch: Optional[str], step: Optional[int]) -> List[Dict]:
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    filtered = []
    for row in rows:
        if branch is not None and row.get("branch_name") != branch:
            continue
        if step is not None and int(float(row["step"])) != step:
            continue
        filtered.append(row)

    if not filtered:
        raise ValueError("No rows matched the requested filters.")
    return filtered


def load_rows_from_pt(pt_path: Path, branch: Optional[str], step: Optional[int]) -> List[Dict]:
    payload = torch.load(pt_path, map_location="cpu")
    records = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        raise ValueError(f"Unexpected PT structure in {pt_path}")

    filtered = []
    for record in records:
        if branch is not None and record.get("branch_name") != branch:
            continue
        if step is not None and int(record["step"]) != step:
            continue

        attn_map = record.get("attn_map")
        if not isinstance(attn_map, torch.Tensor):
            raise ValueError(f"Expected tensor attention map in {pt_path}, got {type(attn_map)}")

        if attn_map.dim() == 3:
            attn_map = attn_map.mean(dim=0)
        if attn_map.dim() != 2:
            raise ValueError(f"Expected 2D or 3D attention map, got shape={tuple(attn_map.shape)}")

        # The logger stores mean-over-heads attention maps [query, key].
        # The original sink score is mean attention received by key 0 across all queries.
        sink_score = float(attn_map[:, 0].float().mean().item())
        filtered.append(
            {
                "step": int(record["step"]),
                "layer": int(record["layer"]),
                "branch_name": record.get("branch_name", "unknown"),
                "sink_score": sink_score,
            }
        )

    if not filtered:
        raise ValueError("No PT records matched the requested filters.")
    return filtered


def aggregate_by_layer(rows: List[Dict]) -> Dict[int, float]:
    values: Dict[int, List[float]] = {}
    for row in rows:
        layer = int(float(row["layer"]))
        sink_score = float(row["sink_score"])
        values.setdefault(layer, []).append(sink_score)
    return {layer: sum(scores) / len(scores) for layer, scores in values.items()}


def plot_curves(curves: List[Dict], title: str, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    all_layers = set()

    for curve in curves:
        layer_scores = curve["layer_scores"]
        layers = sorted(layer_scores.keys())
        scores = [layer_scores[layer] for layer in layers]
        all_layers.update(layers)
        mean_score = sum(scores) / len(scores)
        (line,) = ax.plot(layers, scores, linewidth=1.8, label=curve["label"])
        ax.axhline(
            mean_score,
            linestyle="--",
            linewidth=1.2,
            color=line.get_color(),
            alpha=0.9,
            label=f"{curve['label']} mean={mean_score:.4f}",
        )

    ax.set_xlabel("layer")
    ax.set_ylabel("timestep token attention score")
    ax.set_title(title)
    min_layer = min(all_layers)
    max_layer = max(all_layers)
    xticks = list(range(min_layer, max_layer + 1, 5))
    if max_layer not in xticks:
        xticks.append(max_layer)
    ax.set_xticks(sorted(set(xticks)))
    ax.set_ylim(0.0, 1.0)
    if len(curves) > 1 or any("mean" not in curve["label"].lower() for curve in curves):
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    csv_args = [args.csv, args.csv2, args.csv3, args.csv4, args.csv5]
    pt_args = [args.pt, args.pt2, args.pt3, args.pt4, args.pt5]
    using_pt = any(value is not None for value in pt_args)
    using_csv = any(value is not None for value in csv_args)
    if using_pt and using_csv:
        raise ValueError("Use either CSV inputs or PT inputs, not both.")

    if using_pt:
        provided_pt_args = [value for value in pt_args if value is not None]
        input_paths = [resolve_pt_path(path) for path in provided_pt_args]
    else:
        provided_csv_args = [value for value in csv_args if value is not None]
        if provided_csv_args:
            input_paths = [resolve_csv_path(path) for path in provided_csv_args]
        else:
            input_paths = [resolve_csv_path(None)]

    curves = []
    curve_names = [args.name, args.name2, args.name3, args.name4, args.name5]
    for idx, input_path in enumerate(input_paths):
        rows = load_rows_from_pt(input_path, branch=args.branch, step=args.step) if using_pt else load_rows(input_path, branch=args.branch, step=args.step)
        layer_scores = aggregate_by_layer(rows)
        curves.append(
            {
                "label": curve_names[idx] if idx < len(curve_names) and curve_names[idx] else input_path.stem,
                "layer_scores": layer_scores,
            }
        )

    if args.out:
        output_path = Path(args.out)
    elif len(input_paths) == 1:
        output_path = input_paths[0].with_name(f"{input_paths[0].stem}_lineplot.png")
    else:
        output_path = input_paths[0].with_name(f"{input_paths[0].stem}_vs_{input_paths[1].stem}_lineplot.png")

    title = "Timestep Token Attention Score"
    if args.branch is not None:
        title += f" ({args.branch})"
    if args.step is not None:
        title += f" step={args.step}"
    else:
        title += " (avg over matched steps)"

    plot_curves(curves, title=title, output_path=output_path, dpi=args.dpi)
    print(f"Saved line plot to: {output_path}")


if __name__ == "__main__":
    main()
