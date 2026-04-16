"""
Utility script to compute AudioLDM evaluation metrics (FAD, FID, KID, etc.)
for a pair of audio tracks by reusing the same EvaluationHelper logic that
`MusicLDM` employs during validation.

Example:
    python run_eval_for_tracks.py
        --generated /path/to/gen.wav
        --reference /path/to/ref.wav
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import torch  # type: ignore[import]

from audioldm_eval import EvaluationHelper  # type: ignore[import]



DEFAULT_REFERENCE = (
    "/app/data/code/MSDM/multi-source-diffusion-models-main/"
    "exp/inpaint_all/Track01883/bass/original/bass.wav"
)
DEFAULT_GENERATED = (
    "/app/data/code/MSDM/multi-source-diffusion-models-main/"
    "exp/inpaint_all/Track01883/bass/inpaint_000/bass.wav"
)

MIN_FILES_PER_DIR = 2

def _normalize_input_path(
    candidate: Path, tmp_root: Path, label: str
) -> Tuple[Path, bool]:
    """
    Return a directory that contains the audio files to evaluate.

    If `candidate` is already a directory, it is returned as-is
    and the boolean flag is False. Otherwise the file is copied
    into a temporary directory under `tmp_root/label`, which is
    returned together with True to indicate that it should be
    cleaned up by the caller.
    """
    candidate = candidate.expanduser()
    if candidate.is_dir():
        return candidate, False
    if candidate.is_file():
        target_dir = tmp_root / label
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, target_dir / candidate.name)
        return target_dir, True
    raise FileNotFoundError(f"Input path does not exist: {candidate}")


def _ensure_min_files(directory: Path, min_files: int = MIN_FILES_PER_DIR) -> None:
    files = [p for p in directory.iterdir() if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No audio files found in {directory}")
    source = files[0]
    duplicate_index = 0
    while len(files) < min_files:
        duplicate_index += 1
        dup_name = f"{source.stem}_dup{duplicate_index}{source.suffix}"
        dup_path = directory / dup_name
        shutil.copy2(source, dup_path)
        files.append(dup_path)

def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(
    generated: Path,
    reference: Path,
    sample_rate: int = 16000,
    device: torch.device | None = None,
) -> dict:
    device = device or _resolve_device(None)
    helper = EvaluationHelper(sample_rate, device)

    tmp_root = Path(tempfile.mkdtemp())
    try:
        gen_dir, gen_tmp = _normalize_input_path(Path(generated), tmp_root, "generated")
        ref_dir, ref_tmp = _normalize_input_path(Path(reference), tmp_root, "reference")
        _ensure_min_files(gen_dir)
        _ensure_min_files(ref_dir)
        metrics = helper.main(str(gen_dir), str(ref_dir))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute FAD/FID/KID and related metrics for two audio tracks "
            "using audioldm_eval.EvaluationHelper."
        )
    )
    parser.add_argument(
        "--generated",
        type=Path,
        default=Path(DEFAULT_GENERATED),
        help="Path to the generated/inpainted audio (file or directory).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(DEFAULT_REFERENCE),
        help="Path to the reference/ground-truth audio (file or directory).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate expected by EvaluationHelper.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (default: auto-detect CUDA, fallback to CPU).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("generated_metrics.json"),
        help="Path to save the resulting metrics as JSON (default: ./generated_metrics.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    metrics = compute_metrics(
        generated=args.generated,
        reference=args.reference,
        sample_rate=args.sample_rate,
        device=device,
    )

    print("Evaluation metrics (matching audioldm_eval outputs):")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    output_path = args.output_json.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()

