
import json
import soundfile as sf

from omegaconf import OmegaConf
from hydra import main as hydra_main

# ==== Import infer.py entry points ====
import infer  # 我们直接 import infer.py 并调用其中的生成逻辑
from infer import _ensure_dir, _proc_audio_mean_mono
import os
import torch
import torchaudio
import numpy as np
from torch import nn
from torch.hub import load_state_dict_from_url
# ==== 评估指标 & 工具 ====
try:
    import librosa
except Exception:
    librosa = None

from typing import List, Dict, Optional
from dataclasses import dataclass

from scipy.linalg import sqrtm

# ==== PANNs 导入兼容层 ====
# 兼容不同版本的 panns-inference：优先尝试 Cnn14_Embedding，再尝试 PANNs / Wavegram 等
Cnn14_Embedding = None
PANNs_model = None
try:
    from panns_inference import Cnn14_Embedding
except Exception:
    Cnn14_Embedding = None
    try:
        # 有些发行版使用不同的名字或导出不同的接口
        from panns_inference import PANNs as PANNs_model
    except Exception:
        PANNs_model = None


def _get_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


from panns_inference import AudioTagging, SoundEventDetection, labels

class PANNsEmbedder:
    def __init__(self, sr=32000, device="cuda", checkpoint="/root/panns_data/Cnn14_mAP=0.431.pth"):
        import torchaudio
        import os
        from panns_inference import AudioTagging

        self.sr = sr
        self.device = device

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"PANNs checkpoint not found: {checkpoint}")

        # AudioTagging 是推理类，不是 nn.Module
        self.wrapper = AudioTagging(checkpoint_path=checkpoint, device=device)
        self.model = self.wrapper.model  # 这里才是 nn.Module
        self.model.eval()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=320,
            n_mels=64
        ).to(device)

    def __call__(self, audio):
        """
        输入 audio 可以是 numpy / torch / list
        返回 PANNs embedding 向量 (numpy.ndarray, shape=(D,))
        """
        import torch
        import numpy as np

        # ========== 1. 空输入检查 ==========
        if audio is None:
            raise ValueError("输入音频为空")

        # ========== 2. 转 torch tensor ==========
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        elif isinstance(audio, list):
            audio = torch.tensor(audio, dtype=torch.float32)

        # ========== 3. 单声道处理 ==========
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1, T]

        # ========== 4. 长度检查 ==========
        min_len = 1024
        if audio.size(-1) < min_len:
            pad_width = min_len - audio.size(-1)
            audio = torch.nn.functional.pad(audio, (0, pad_width))

        audio = audio.to(self.device)

        # ========== 5. 前向推理 ==========
        with torch.no_grad():
            output = self.wrapper.inference(audio)

            # 兼容 dict / tuple / list
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

        # ========== 6. mean pooling & 返回 numpy ==========
        if torch.is_tensor(emb):
            emb = emb.detach().cpu().mean(dim=0).numpy()
        elif isinstance(emb, np.ndarray):
            emb = emb.mean(axis=0)
        else:
            raise TypeError(f"未知的 embedding 类型: {type(emb)}")

        return emb



# ========== FAD 计算工具（稳健版本） ==========

def compute_fad(emb_gen: np.ndarray, emb_ref: np.ndarray) -> float:
    """计算 Fréchet Audio Distance，接受任意形状的 embedding（会做必要的 reshape）。

    emb_*: numpy arrays, shape 可为 (D,), (N, D), (B, T, D) 等。
    """
    # --- 规范化输入为 (N, D)
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
        raise RuntimeError('不支持的 embedding 维度: %d' % x.ndim)

    emb_gen = _norm(emb_gen)
    emb_ref = _norm(emb_ref)

    # 如果样本数 < 2，则复制以确保 np.cov 返回矩阵
    if emb_gen.shape[0] == 1:
        emb_gen = np.vstack([emb_gen, emb_gen])
    if emb_ref.shape[0] == 1:
        emb_ref = np.vstack([emb_ref, emb_ref])

    mu1, mu2 = emb_gen.mean(0), emb_ref.mean(0)
    cov1, cov2 = np.cov(emb_gen, rowvar=False), np.cov(emb_ref, rowvar=False)

    # 防护：cov 可能退化为标量（极端情况），将其扩展为对角矩阵
    if cov1.ndim == 0:
        cov1 = np.array([[float(cov1)]])
    if cov2.ndim == 0:
        cov2 = np.array([[float(cov2)]])

    # 防护：检查并处理 NaN/Inf
    if not np.isfinite(cov1).all() or not np.isfinite(cov2).all():
        # 替换 NaN/Inf 为小值
        cov1 = np.nan_to_num(cov1, nan=1e-8, posinf=1e-8, neginf=1e-8)
        cov2 = np.nan_to_num(cov2, nan=1e-8, posinf=1e-8, neginf=1e-8)
        # 添加正则化确保正定
        eps = 1e-6 * np.eye(cov1.shape[0])
        cov1 = cov1 + eps
        cov2 = cov2 + eps

    diff = mu1 - mu2
    covmean = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # 如果 covmean 仍包含 NaN/Inf，尝试更保守的方法
    if not np.isfinite(covmean).all():
        # 使用对角近似
        cov1_diag = np.diag(np.diag(cov1))
        cov2_diag = np.diag(np.diag(cov2))
        covmean = sqrtm(cov1_diag @ cov2_diag)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        val = float(diff @ diff + np.trace(cov1_diag + cov2_diag - 2 * covmean))
    else:
        val = float(diff @ diff + np.trace(cov1 + cov2 - 2 * covmean))

    return val


# ========== 数据结构 ==========
@dataclass
class EvalItem:
    uid: str
    given_wav: Optional[str]
    ref_mix: Optional[str]
    ref_sources: Dict[str, str]


@dataclass
class EvalConfig:
    ckpt_path: str
    task: str
    text_prompt: Optional[str]
    manifest_path: str
    output_root: str
    sample_rate: int = 48000
    quality_score: float = 0.75  # 添加质量控制参数，默认为中等质量(0.5)
    enable_quality_control: bool = True  # 是否启用质量控制，默认为True


# ========== Evaluator ==========
class Evaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        try:
            self.embedder = PANNsEmbedder(sr=cfg.sample_rate, device=_get_device())
        except Exception as e:
            raise RuntimeError('无法初始化 PANNs 嵌入器: %s' % str(e))

    def _load_mono(self, path: str) -> np.ndarray:
        x, sr = sf.read(path)
        print(f"[Debug] Loaded {path}: shape={x.shape}, sr={sr}")  # 打印原始 shape

        if sr != self.cfg.sample_rate:
            x_t = torch.from_numpy(x).float()
            if x_t.ndim == 1:
                x_t = x_t.unsqueeze(0)
            else:
                # 保证通道在 axis 0
                if x_t.shape[0] < x_t.shape[1]:
                    x_t = x_t.T
            x_t = torchaudio.functional.resample(x_t, sr, self.cfg.sample_rate)
            x = x_t.numpy()

        # 最终保证 x 是一维 numpy 数组
        if isinstance(x, np.ndarray) and x.ndim > 1:
            x = x.mean(axis=0)

        print(f"[Debug] Final mono shape: {x.shape}")  # 打印最终 shape
        return x.astype(np.float32)

    def evaluate_item(self, gen_mix_path: str, ref_mix_path: str) -> Dict[str, float]:
        if ref_mix_path is None:
            return {}

        # 读取生成音频和参考音频
        g, r = self._load_mono(gen_mix_path), self._load_mono(ref_mix_path)

        # ==== 输出音频长度确认 ====
        print(f"[Debug] Generated audio length: {len(g)} samples, Reference audio length: {len(r)} samples")

        # 调用 PANNs 嵌入器
        eg, er = self.embedder(g), self.embedder(r)

        # 计算 FAD
        fad = compute_fad(eg, er)
        return {"FAD": fad}


# ========== Runner ==========
@hydra_main(version_base=None, config_path="configs", config_name="default_dit")
def run(hydra_cfg):
    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
    ecfg = EvalConfig(
        ckpt_path=cfg_dict["ckpt_path"],
        task=cfg_dict["task"],
        text_prompt=cfg_dict.get("text_prompt"),
        manifest_path=cfg_dict["eval"]["manifest_path"],
        output_root=cfg_dict.get("output_root", "runs/eval"),
        sample_rate=cfg_dict["model"]["sample_rate"],
        quality_score=cfg_dict.get("quality_score", 0.75),  # 添加质量分数，默认为0.75
        enable_quality_control=cfg_dict.get("enable_quality_control", True),  # 添加质量控制开关，默认为True
    )

    # 读取 manifests
    items: List[EvalItem] = []
    with open(ecfg.manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            items.append(EvalItem(uid=d["uid"], given_wav=d.get("given_wav"), ref_mix=d.get("ref_mix"), ref_sources=d.get("ref_sources", {})))

    evaluator = Evaluator(ecfg)
    all_results = []

    for it in items:
        out_dir = os.path.join(ecfg.output_root, ecfg.task, it.uid)
        _ensure_dir(out_dir)
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # ==== 调用 infer.py 的生成逻辑 ====
        gen_paths = infer.run_inference(
            ckpt_path=ecfg.ckpt_path,
            task=ecfg.task,
            text_prompt=ecfg.text_prompt,
            given_wav=it.given_wav,
            out_dir=out_dir,
            quality_score=ecfg.quality_score,  # 添加质量控制参数
            enable_quality_control=ecfg.enable_quality_control  # 添加质量控制开关
        )
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # ==== 评估 ====
        metrics = evaluator.evaluate_item(gen_paths.get("mix"), it.ref_mix)
        res = {"uid": it.uid, "metrics": metrics}
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(res, f, indent=2)
        all_results.append(res)
        print(f"[Eval] {it.uid}: {metrics}")
        
        # 强制垃圾回收
        import gc
        gc.collect()

    # 聚合
    agg = {}
    for k in {m for r in all_results for m in r["metrics"]}:
        vals = [r["metrics"][k] for r in all_results if k in r["metrics"]]
        agg[k] = float(np.mean(vals))
    with open(os.path.join(ecfg.output_root, f"summary_{ecfg.task}.json"), "w") as f:
        json.dump({"aggregate": agg, "items": all_results}, f, indent=2)
    print("==== Aggregate ====")
    for k, v in agg.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    run()
