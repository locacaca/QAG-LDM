import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot loss curves from loss_curve*.csv files up to a target epoch."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=".",
        help="Directory containing loss_curve*.csv files.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Plot losses using files up to and including this epoch.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["gated", "ungated"],
        help="Optional mode filter.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory. Defaults to <log-dir>/loss_plots_epoch<epoch>.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric whitelist, e.g. train/loss train/mse_loss.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=0.0,
        help="Lower bound of y-axis. Defaults to 0.0.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=1.0,
        help="Upper bound of y-axis. Defaults to 1.0.",
    )
    parser.add_argument(
        "--drop-above",
        type=float,
        default=None,
        help="Optional threshold; points with value above it are excluded from plotting.",
    )
    return parser.parse_args()


def parse_epoch_and_mode(path: Path) -> Optional[Tuple[int, Optional[str]]]:
    match = re.match(r"^loss_curve(?:_(gated|ungated))?_epoch(\d+)(?:_.*)?\.csv$", path.name)
    if match is None:
        return None
    mode = match.group(1)
    epoch = int(match.group(2))
    return epoch, mode


def resolve_csv_paths(log_dir: Path, max_epoch: int, mode: Optional[str]) -> List[Tuple[int, Path]]:
    matched = []
    for path in sorted(log_dir.glob("loss_curve*.csv")):
        parsed = parse_epoch_and_mode(path)
        if parsed is None:
            continue
        epoch, path_mode = parsed
        if epoch > max_epoch:
            continue
        if mode is not None and path_mode != mode:
            continue
        matched.append((epoch, path))

    if not matched:
        mode_msg = f" with mode={mode}" if mode is not None else ""
        raise FileNotFoundError(
            f"No loss_curve*.csv files found in {log_dir} up to epoch {max_epoch}{mode_msg}."
        )
    return matched


def load_metric_points(csv_paths: List[Tuple[int, Path]], metrics: Optional[List[str]]) -> Dict[str, List[Tuple[int, float, int]]]:
    metric_points: Dict[str, List[Tuple[int, float, int]]] = {}

    for epoch, csv_path in csv_paths:
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        metric_keys = [
            key for key in rows[0].keys()
            if key != "step" and any(row.get(key) not in (None, "") for row in rows)
        ]
        if metrics is not None:
            metric_keys = [key for key in metric_keys if key in metrics]

        for row in rows:
            raw_step = row.get("step")
            if raw_step in (None, ""):
                continue
            step = int(float(raw_step))

            for key in metric_keys:
                raw_value = row.get(key)
                if raw_value in (None, ""):
                    continue
                metric_points.setdefault(key, []).append((step, float(raw_value), epoch))

    for key in metric_points:
        metric_points[key].sort(key=lambda item: item[0])

    return metric_points


def sanitize_metric_name(metric: str) -> str:
    return metric.replace("/", "_").replace("\\", "_")


def plot_metric(
    metric: str,
    points: List[Tuple[int, float, int]],
    outdir: Path,
    dpi: int,
    ymin: float,
    ymax: float,
    drop_above: Optional[float],
) -> None:
    if drop_above is not None:
        points = [item for item in points if item[1] <= drop_above]
    if not points:
        return

    steps = [item[0] for item in points]
    values = [item[1] for item in points]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(steps, values, linewidth=1.5)
    ax.set_title(metric)
    ax.set_xlabel("step")
    ax.set_ylabel("value")

    # Mark epoch boundaries using the first step seen for each epoch.
    epoch_first_step: Dict[int, int] = {}
    for step, _, epoch in points:
        epoch_first_step.setdefault(epoch, step)

    for epoch in sorted(epoch_first_step):
        ax.axvline(epoch_first_step[epoch], linestyle="--", linewidth=0.8, alpha=0.25, color="gray")

    if steps:
        ax.set_xlim(min(steps), max(steps))
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(outdir / f"{sanitize_metric_name(metric)}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    csv_paths = resolve_csv_paths(log_dir=log_dir, max_epoch=args.epoch, mode=args.mode)
    metric_points = load_metric_points(csv_paths=csv_paths, metrics=args.metrics)
    if not metric_points:
        raise ValueError("No metric data found in the selected loss CSV files.")

    outdir = Path(args.outdir) if args.outdir else log_dir / f"loss_plots_epoch{args.epoch}"
    outdir.mkdir(parents=True, exist_ok=True)

    for metric, points in metric_points.items():
        if not points:
            continue
        plot_metric(
            metric=metric,
            points=points,
            outdir=outdir,
            dpi=args.dpi,
            ymin=args.ymin,
            ymax=args.ymax,
            drop_above=args.drop_above,
        )

    print(f"Saved loss plots to: {outdir}")


if __name__ == "__main__":
    main()
