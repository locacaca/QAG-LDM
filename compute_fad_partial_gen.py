import argparse
import json
import os
from typing import List

import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy.linalg import sqrtm


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_mono(path: str, target_sr: int) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio not found: {path}")
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    x = x.astype(np.float32)
    if sr != target_sr:
        x_t = torch.from_numpy(x).unsqueeze(0)
        x_t = torchaudio.functional.resample(x_t, sr, target_sr)
        x = x_t.squeeze(0).numpy()
    return x


# ---------------------------------------------------------------------------
# PANNs embedder — per-frame embeddings (standard FAD practice)
# ---------------------------------------------------------------------------

class PANNsEmbedder:
    """
    Extracts per-clip embeddings using PANNs CNN14.

    Standard FAD (Kilgour et al. 2019) fits a multivariate Gaussian over a
    *set* of embeddings, one per audio clip.  We therefore return one
    embedding vector per __call__, but we do NOT average over time frames
    inside the model — instead we use the clip-level embedding that
    AudioTagging already exposes (the global average-pooled feature before
    the classifier head), which is the standard choice in the community.
    """

    PANNS_SR = 32000  # CNN14 was trained at 32 kHz

    def __init__(self, sr: int = 32000, device: str = "cuda",
                 checkpoint: str = "/root/panns_data/Cnn14_mAP=0.431.pth"):
        from panns_inference import AudioTagging
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"PANNs checkpoint not found: {checkpoint}")
        # Always run the model at its native 32 kHz
        self.sr = self.PANNS_SR
        self.device = device
        self.wrapper = AudioTagging(checkpoint_path=checkpoint, device=device)
        self.model = self.wrapper.model
        self.model.eval()

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        audio : 1-D float32 numpy array at self.sr (32 kHz).
        returns: 1-D float32 numpy array, shape (embedding_dim,)
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected 1-D audio, got shape {audio.shape}")
        # Minimum length guard
        if len(audio) < 1024:
            audio = np.pad(audio, (0, 1024 - len(audio)))

        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)  # (1, T)
        with torch.no_grad():
            output = self.wrapper.inference(audio_t)

        # AudioTagging.inference returns (clipwise_output, embedding)
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            emb = output[1]          # embedding is the second element
        elif isinstance(output, dict):
            emb = output.get("embedding", output.get("clipwise_output"))
        else:
            emb = output

        if torch.is_tensor(emb):
            emb = emb.detach().cpu().numpy()
        emb = np.asarray(emb, dtype=np.float32)
        # Shape may be (1, D) or (D,)
        if emb.ndim == 2:
            emb = emb.mean(axis=0)
        return emb.ravel()


# ---------------------------------------------------------------------------
# FAD computation — standard Fréchet Distance formula
# ---------------------------------------------------------------------------

def _fit_gaussian(embeddings: np.ndarray):
    """Return (mu, sigma) for a set of embeddings (N, D)."""
    mu = embeddings.mean(axis=0)
    # rowvar=False: each row is an observation
    sigma = np.cov(embeddings, rowvar=False)
    if sigma.ndim == 0:                  # single feature edge-case
        sigma = np.array([[float(sigma)]])
    return mu, sigma


def compute_fad(emb_gen: np.ndarray, emb_ref: np.ndarray) -> float:
    """
    Fréchet Audio Distance between two sets of clip-level embeddings.

    emb_gen, emb_ref : (N, D) float32 arrays
    """
    if emb_gen.ndim == 1:
        emb_gen = emb_gen[np.newaxis, :]
    if emb_ref.ndim == 1:
        emb_ref = emb_ref[np.newaxis, :]

    # Duplicate single-sample sets so np.cov doesn't crash
    if emb_gen.shape[0] == 1:
        emb_gen = np.vstack([emb_gen, emb_gen])
    if emb_ref.shape[0] == 1:
        emb_ref = np.vstack([emb_ref, emb_ref])

    mu1, sigma1 = _fit_gaussian(emb_gen)
    mu2, sigma2 = _fit_gaussian(emb_ref)

    diff = mu1 - mu2
    # Matrix square root of sigma1 @ sigma2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        if np.abs(covmean.imag).max() > 1e-3:
            raise RuntimeError("sqrtm produced large imaginary component")
        covmean = covmean.real

    fad = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fad


# ---------------------------------------------------------------------------
# Per-task FAD helper (used by batch_partial_gen_quality_with_fad.sh)
# ---------------------------------------------------------------------------

def compute_fad_for_pair_eval_style(
    gen_audio_path: str,
    ref_audio_paths: List[str],
    embedder: PANNsEmbedder,
) -> float:
    """
    Compute FAD between a single generated file and a list of reference files.

    Note: with very few samples the Gaussian fit is unreliable; treat the
    result as an approximate per-task score rather than a population metric.
    """
    sr = embedder.sr
    gen_audio = _load_mono(gen_audio_path, sr)
    emb_gen = embedder(gen_audio)[np.newaxis, :]          # (1, D)

    ref_embs = []
    for p in ref_audio_paths:
        ref_embs.append(embedder(_load_mono(p, sr)))
    if not ref_embs:
        raise ValueError("No reference audio files provided")
    emb_ref = np.stack(ref_embs, axis=0)                  # (N_ref, D)

    return compute_fad(emb_gen, emb_ref)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute FAD between a generated mix and reference audio files (PANNs CNN14)"
    )
    parser.add_argument("--gen_mix_path", type=str, required=True)
    parser.add_argument("--given_wav_path", type=str, default=None,
                        help="Optional, not used for FAD — recorded in output only")
    parser.add_argument("--unknown_audio_files", type=str, nargs="+", required=True)
    parser.add_argument("--sample_rate", type=int, default=32000,
                        help="Ignored — PANNs CNN14 always runs at 32 kHz")
    parser.add_argument("--panns_checkpoint", type=str,
                        default="/root/panns_data/Cnn14_mAP=0.431.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    embedder = PANNsEmbedder(device=args.device, checkpoint=args.panns_checkpoint)
    fad = compute_fad_for_pair_eval_style(
        gen_audio_path=args.gen_mix_path,
        ref_audio_paths=args.unknown_audio_files,
        embedder=embedder,
    )

    result = {
        "gen_mix_path": args.gen_mix_path,
        "given_wav_path": args.given_wav_path,
        "unknown_audio_files": args.unknown_audio_files,
        "results": {"FAD": float(fad)},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
