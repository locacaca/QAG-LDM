import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


TARGET_FILES = (
    "mix_clap.npy",
    "src_clap.npy",
    "submix_clap.npy",
)


def describe_array(path: Path) -> dict:
    arr = np.load(path, mmap_mode="r")
    return {
        "path": path,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "ndim": arr.ndim,
    }


def summarize(records: list[dict]) -> dict[str, set[tuple[int, ...]]]:
    grouped: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for rec in records:
        grouped[rec["path"].name].add(rec["shape"])
    return grouped


def find_target_files(root: Path) -> list[Path]:
    matches: list[Path] = []
    for name in TARGET_FILES:
        matches.extend(root.rglob(name))
    return sorted(set(matches))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect saved local CLAP numpy files and print their actual dimensions."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan recursively. Default: current directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of example files to print. Default: 20.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = find_target_files(root)

    print(f"scan_root: {root}")
    print(f"matched_files: {len(files)}")

    if not files:
        print("No *_clap.npy files found.")
        return

    records = [describe_array(path) for path in files]
    grouped = summarize(records)

    print("\nSummary by filename")
    for name in TARGET_FILES:
        shapes = sorted(grouped.get(name, set()))
        if not shapes:
            continue
        shape_text = ", ".join(str(shape) for shape in shapes)
        count = sum(1 for rec in records if rec["path"].name == name)
        print(f"- {name}: count={count}, shapes={shape_text}")

    print("\nExample files")
    for rec in records[: args.limit]:
        print(f"- {rec['path']}: shape={rec['shape']}, ndim={rec['ndim']}, dtype={rec['dtype']}")

    clap_like = [rec for rec in records if rec["ndim"] == 2 and rec["shape"][0] == 512]
    if clap_like:
        min_frames = min(rec["shape"][1] for rec in clap_like)
        max_frames = max(rec["shape"][1] for rec in clap_like)
        print("\nInterpretation")
        print("- First dimension is consistently 512, matching CLAP embedding width.")
        print(f"- Second dimension varies from {min_frames} to {max_frames}, indicating saved time-window frames.")
    else:
        print("\nInterpretation")
        print("- Files were found, but shapes do not match the expected [512, num_frames] layout.")


if __name__ == "__main__":
    main()
