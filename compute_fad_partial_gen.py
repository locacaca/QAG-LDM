import argparse
import json
import os
from typing import List, Dict, Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy.linalg import sqrtm


def _get_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PANNsEmbedder:
    def __init__(self, sr=32000, device="cuda", checkpoint="/root/panns_data/Cnn14_mAP=0.431.pth"):
        from panns_inference import AudioTagging
        self.sr = sr
        self.device = device
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"PANNs checkpoint not found: {checkpoint}")
        self.wrapper = AudioTagging(checkpoint_path=checkpoint, device=device)
        self.model = self.wrapper.model
        self.model.eval()

    def __call__(self, audio):
        import numpy as np
        if audio is None:
            raise ValueError("输入音频为空")
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        elif isinstance(audio, list):
            audio = torch.tensor(audio, dtype=torch.float32)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        min_len = 1024
        if audio.size(-1) < min_len:
            pad_width = min_len - audio.size(-1)
            audio = torch.nn.functional.pad(audio, (0, pad_width))
        audio = audio.to(self.device)
        with torch.no_grad():
            output = self.wrapper.inference(audio)
            if isinstance(output, dict):
                emb = output.get("embedding", None)
            elif isinstance(output, (tuple, list)):
                emb = output[0]
            elif torch.is_tensor(output):
                emb = output
            elif isinstance(output, np.ndarray):
                emb = output
            else:
                raise TypeError(f"未知的输出类型: {type(output)}")
        if emb is None:
            raise RuntimeError("PANNs inference 没有返回 embedding")
        if torch.is_tensor(emb):
            emb = emb.detach().cpu().mean(dim=0).numpy()
        elif isinstance(emb, np.ndarray):
            emb = emb.mean(axis=0)
        else:
            raise TypeError(f"未知的 embedding 类型: {type(emb)}")
        return emb


def compute_fad(emb_gen: np.ndarray, emb_ref: np.ndarray) -> float:
    def _norm(x: np.ndarray) -> np.ndarray:
        if x is None:
            return None
        if x.ndim == 1:
            return x[np.newaxis, :]
        if x.ndim == 3:
            b, t, d = x.shape
            return x.reshape(b * t, d)
        if x.ndim == 2:
            return x
        raise RuntimeError("不支持的 embedding 维度: %d" % x.ndim)

    emb_gen = _norm(emb_gen)
    emb_ref = _norm(emb_ref)
    if emb_gen.shape[0] == 1:
        emb_gen = np.vstack([emb_gen, emb_gen])
    if emb_ref.shape[0] == 1:
        emb_ref = np.vstack([emb_ref, emb_ref])
    mu1, mu2 = emb_gen.mean(0), emb_ref.mean(0)
    cov1, cov2 = np.cov(emb_gen, rowvar=False), np.cov(emb_ref, rowvar=False)
    if cov1.ndim == 0:
        cov1 = np.array([[float(cov1)]])
    if cov2.ndim == 0:
        cov2 = np.array([[float(cov2)]])
    diff = mu1 - mu2
    covmean = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    try:
        val = float(diff @ diff + np.trace(cov1 + cov2 - 2 * covmean))
    except Exception:
        cov1_diag = np.diag(np.diag(cov1))
        cov2_diag = np.diag(np.diag(cov2))
        covmean = sqrtm(cov1_diag @ cov2_diag)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        val = float(diff @ diff + np.trace(cov1_diag + cov2_diag - 2 * covmean))
    return val


def _load_mono(path: str, target_sr: int) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio not found: {path}")
    x, sr = sf.read(path)
    if sr != target_sr:
        x_t = torch.from_numpy(x).float()
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        else:
            if x_t.shape[0] < x_t.shape[1]:
                x_t = x_t.T
        x_t = torchaudio.functional.resample(x_t, sr, target_sr)
        x = x_t.numpy()
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x.mean(axis=0)
    return x.astype(np.float32)


def compute_fad_for_pair_eval_style(gen_audio_path: str, ref_audio_paths: List[str], embedder: PANNsEmbedder) -> float:
    target_sr = embedder.sr
    g = _load_mono(gen_audio_path, target_sr)
    eg = embedder(g)
    ref_embeds = []
    for p in ref_audio_paths:
        r = _load_mono(p, target_sr)
        er = embedder(r)
        ref_embeds.append(er)
    if len(ref_embeds) == 0:
        raise ValueError("参考音频为空")
    er_all = np.stack(ref_embeds, axis=0)
    fad = compute_fad(eg, er_all)
    return fad


def main():
    parser = argparse.ArgumentParser(description="Compute FAD between generated mix and unknown audio files (eval_total_gen style)")
    parser.add_argument("--gen_mix_path", type=str, required=True)
    parser.add_argument("--given_wav_path", type=str, required=False, help="可选，不用于FAD，只用于记录")
    parser.add_argument("--unknown_audio_files", type=str, nargs="+", required=True)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--panns_checkpoint", type=str, default="/root/panns_data/Cnn14_mAP=0.431.pth")
    parser.add_argument("--device", type=str, default="cuda", help="设备，可以是 'cuda', 'cpu', 或 'cuda:N'")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    # 解析设备参数，支持 cuda:N 格式
    device = args.device
    if device.startswith("cuda:") and "," not in device:
        # cuda:N 格式，直接使用
        pass
    elif device == "cuda":
        # 检查可用的 CUDA 设备数量
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = f"cuda:0"  # 默认使用 cuda:0
    # 其他情况保持不变 (cpu 或其他)

    panns_sr = 32000
    if args.sample_rate != panns_sr:
        print(f"[WARN] 忽略 --sample_rate={args.sample_rate}，PANNs 评估固定使用 {panns_sr} Hz 以匹配权重", flush=True)
    embedder = PANNsEmbedder(sr=panns_sr, device=device, checkpoint=args.panns_checkpoint)
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
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


