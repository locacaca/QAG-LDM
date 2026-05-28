import os; opj = os.path.join
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from multi_track_stable_audio.models.factory import create_mgeldm_from_config
from multi_track_stable_audio.utils import load_ckpt_state_dict, to_numpy

from multi_track_stable_audio.inference.task_wrapper import InferenceTaskWrapper

import warnings
import torch
import torchaudio
import soundfile as sf
import re
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
warnings.filterwarnings("ignore")

# from sweetdebug import sweetdebug; sweetdebug()


# def generate_filename(output_dir):
#     """
#     Generate a unique filename for the output audio file.
#     """
#     # if not os.path.exists(output_dir):
#     #     os.makedirs(output_dir)
#     # return opj(output_dir, f"output_{len(os.listdir(output_dir)) + 1}.wav")
#     return f"output_{len(os.listdir(output_dir)) + 1}"
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _proc_audio_mean_mono(wav: torch.Tensor, downsample_ratio: int) -> torch.Tensor:
    wav = wav.mean(dim=0, keepdim=True)
    wav_len = wav.shape[-1]
    if wav_len % downsample_ratio != 0:
        new_len = (wav_len // downsample_ratio) * downsample_ratio
        wav = wav[:, :new_len]
    return wav

def save_spectrogram(wav: np.ndarray, 
                     sample_rate: int,
                     output_path):
    if wav.ndim > 1:
        sig = wav.mean(axis=0)  # Average across channels if multi-channel
    else:
        sig = wav
    # 清理非有限值，避免 librosa 抛错
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if not np.any(np.isfinite(sig)) or np.all(sig == 0):
        # 若数据仍不合法或全零，则跳过频谱保存
        return
    D = librosa.stft(sig, n_fft=1024, hop_length=256, win_length=1024)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # plt.figure(figsize=(10, 4))
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(
        S_db, sr=sample_rate, x_axis='time', y_axis='linear', cmap='magma', ax=ax,
        hop_length=256,
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    


def proc_audio(wav, downsample_ratio):
    assert wav.dim() == 2, "Expected wav to be a 2D tensor (channels, time)."
    wav = wav.mean(dim=0, keepdim=True)  # Average across channels
    ## wav len must be divisible by downsample_ratio
    wav_len = wav.shape[-1]
    if wav_len % downsample_ratio != 0:
        new_len = (wav_len // downsample_ratio) * downsample_ratio
        wav = wav[:, :new_len]
    assert wav.shape[-1] % downsample_ratio == 0, "Wav length must be divisible by downsample_ratio."
    return wav


@hydra.main(version_base=None, config_path="configs", config_name="default_dit")
def inference(config):
    """
    We need:
    config.ckpt_path: unwrapped DiT checkpoint path
    config.task: Task to run (e.g., 'generate', 'extract', 'partial_generation', 'accompaniment_generation')
    
    config.given_wav_path: Path to the input audio file for inference, can be None for total generation
    config.text_prompt: Text prompt for the model, can be None for unconditional generation
    config.output_dir: Directory to save the output files
    
    ## For generaiton,
        config.gen_audio_dur: Duration of the audio to generate (in seconds)
    
    config.num_steps: Number of diffusion steps
    config.cfg_scale: CFG scale for inference
    config.overlap_dur: Overlap duration for outpainting (in seconds)
    config.repaint_n: RePaint steps for continuous generation (i.e., for outpainting given overlapped region.)
    
    ## 质量控制参数
    config.quality_score: 质量分数，范围为0-1，默认为0.5（中等质量）
    
    ## Attention Gating 参数
    config.enable_attention_gating: 是否启用注意力门控，默认从配置读取
    """
    print(f"Running task: {config.task}")
    
    # 检查 checkpoint 中的 enable_attention_gating 设置
    # 这对于正确加载模型至关重要
    # 优先级: 命令行参数 > MGELDM 配置 > trainer 配置 > 默认值 True
    enable_attention_gating = getattr(config, 'enable_attention_gating', None)
    
    if enable_attention_gating is None:
        # 从 MGELDM 配置中获取
        if "MGELDM" in config.model:
            enable_attention_gating = config.model["MGELDM"].get("enable_attention_gating", None)
        else:
            enable_attention_gating = config.model.get("enable_attention_gating", None)
    
    if enable_attention_gating is None:
        # 从 trainer 配置中获取
        enable_attention_gating = config.trainer.get("enable_attention_gating", True)
    
    # 强制将 enable_attention_gating 传递给模型配置
    if "MGELDM" in config.model:
        config.model["MGELDM"]["enable_attention_gating"] = enable_attention_gating
    else:
        config.model["enable_attention_gating"] = enable_attention_gating
    
    print(f"enable_attention_gating: {enable_attention_gating}")
    
    model = create_mgeldm_from_config(config.model)
    # ckpt = load_ckpt_state_dict(config.ckpt_path, adjust_key_names=True)
    ckpt = load_ckpt_state_dict(config.ckpt_path)
    model.load_state_dict(ckpt, strict=True)
    del ckpt  # 释放 checkpoint 内存
    torch.cuda.empty_cache()
    print(f"Loaded model from {config.ckpt_path}")
    
    # 将模型移至指定设备并初始化所有组件
    device = torch.device(f"cuda:0")
    model.prepare_for_inference(device)
    print("模型已准备好进行推理，设备:", device)
    
    task_wrapper = InferenceTaskWrapper(
        model=model,
        segment_length_trained=config.model.segment_length,
        timestep_eps=config.model.timestep_eps,
        clap_ckpt_path=None, ## MGE-LDM already has CLAP text conditioner.
        load_clap_audio=False, ## Audio conditioning is not implemented yet
        device=device,
    )
    # os.makedirs(config.output_dir, exist_ok=True)
    
    # 检查是否有质量分数参数
    quality_score = getattr(config, 'quality_score', 0.5)  # 默认为0.5（中等质量）
    print(f"使用质量分数: {quality_score}")
    
    # 预先测试质量控制器是否可用
    func_args = {
        "overlap_dur": config.overlap_dur,
        "cfg_scale": config.cfg_scale,
        "num_timesteps": config.num_steps,
        "repaint_n": config.repaint_n,
        "quality_score": quality_score,
        "verbose": True,
    }
    
    if config.task == "total_gen":
        output_dir = opj(config.output_dir, "total_gen")
        
        """
        output: dict(
            "gen_mix": torch.Tensor, (B, 1, T)
            "gen_submix": torch.Tensor, (B, 1, T)
            "gen_src": torch.Tensor, (B, 1, T)
        )
        """
        assert config.given_wav_path is None, "For total generation, given_wav_path should be None."
        text_conds_mix = [config.text_prompt]
        
        output = \
            task_wrapper.total_mixture_generation(
                text_conds_mix=text_conds_mix, # only generated single output if length of list = 1
                audio_dur= config.gen_audio_dur,
                return_submix_src=True,
                **func_args,  # overlap_dur, cfg_scale, num_steps, repaint_n, quality_score, verbose
            )
        
        
        num_samples = len(text_conds_mix)
        given_wav = None

                
    elif config.task == "partial_gen":
        """
        task_wrapper.partial_generation supports batch processing,
        but this code is designed for single input generation.
        """
        output_dir = opj(config.output_dir, "partial_gen_single")
        
        assert config.given_wav_path is not None, "For partial generation, given_wav_path should be provided."
        given_wav, sr_ori = torchaudio.load(config.given_wav_path)
        given_wav = torchaudio.transforms.Resample(sr_ori, config.model.sample_rate)(given_wav)
        given_wav = proc_audio(given_wav, downsample_ratio=2048) ## Downsample ratio of the autoencoder
        # given_wav: (1, T) 
        # if given_wav.shape[0] > 1:
        #     print("Warning: Multichannel audio detected. Only the first channel will be used.")
        #     given_wav = torch.mean(given_wav, dim=0, keepdim=True)  # Average across channels
        assert given_wav.shape[0] == 1, "Expected single channel audio. Multichannel audio is not implemented yet."
        
        # 检查是否需要随机截取片段
        segment_duration = getattr(config, 'segment_duration', None)
        random_segment = getattr(config, 'random_segment', False)
        
        if segment_duration is not None and random_segment:
            # 使用模型训练时的segment_length来确保兼容性
            segment_length = config.model.segment_length  # 通常是100个潜在帧
            downsample_ratio = 2048  # 压缩比
            target_samples = segment_length * downsample_ratio  # 100 * 2048 = 204800个样本
            current_samples = given_wav.shape[1]
            
            if current_samples > target_samples:
                # 随机选择起始位置
                max_start = current_samples - target_samples
                start_idx = np.random.randint(0, max_start + 1)
                end_idx = start_idx + target_samples
                
                # 截取片段
                given_wav = given_wav[:, start_idx:end_idx]
                actual_duration = target_samples / config.model.sample_rate
                print(f"随机截取音频片段: {actual_duration:.2f}秒 (从 {start_idx/config.model.sample_rate:.2f}s 到 {end_idx/config.model.sample_rate:.2f}s)")
                print(f"片段长度: {target_samples}个样本 = {segment_length}个潜在帧")
            else:
                print(f"音频长度 ({current_samples/config.model.sample_rate:.2f}s) 小于模型期望长度 ({actual_duration:.2f}s)，使用完整音频")
        
        given_wav = given_wav.unsqueeze(0)  # (1, 1, T) # Temporary code. 
        
        text_conds_src = [config.text_prompt]
        assert given_wav.shape[0] == len(text_conds_src), "The number of given_wav and text_conds_src must be the same."
        
        output = \
            task_wrapper.partial_generation(
                given_wav=given_wav,
                text_conds_src=text_conds_src,  # only generated single output if length of list = 1
                overlap_dur=config.overlap_dur,
                cfg_scale=config.cfg_scale,
                num_timesteps=config.num_steps,
                repaint_n=config.repaint_n,  # RePaint steps for continuous generation.
                quality_score=quality_score,
                verbose=True,
                return_full_output=True,
            )
        num_samples = given_wav.shape[0]

    elif config.task == "source_extract":
        assert config.given_wav_path is not None, "For source extraction, given_wav_path should be provided."
        output_dir = opj(config.output_dir, "src_extract")
        mix_wav, sr_ori = torchaudio.load(config.given_wav_path)
        mix_wav = torchaudio.transforms.Resample(sr_ori, config.model.sample_rate)(mix_wav)
        mix_wav = proc_audio(mix_wav, downsample_ratio=2048)  # (1, T)

        # 可选：按训练长度随机截取片段
        segment_duration = getattr(config, 'segment_duration', None)
        random_segment = getattr(config, 'random_segment', False)
        if segment_duration is not None and random_segment:
            segment_length = config.model.segment_length  # 潜在帧数（如100）
            downsample_ratio = 2048
            target_samples = segment_length * downsample_ratio
            current_samples = mix_wav.shape[1]
            if current_samples > target_samples:
                max_start = current_samples - target_samples
                start_idx = np.random.randint(0, max_start + 1)
                end_idx = start_idx + target_samples
                mix_wav = mix_wav[:, start_idx:end_idx]
                actual_duration = target_samples / config.model.sample_rate
                print(f"随机截取混音片段: {actual_duration:.2f}秒 (从 {start_idx/config.model.sample_rate:.2f}s 到 {end_idx/config.model.sample_rate:.2f}s)")
                print(f"片段长度: {target_samples}个样本 = {segment_length}个潜在帧")
            else:
                print(f"音频长度 ({current_samples/config.model.sample_rate:.2f}s) 小于期望长度，使用完整音频")

        assert mix_wav.shape[0] == 1, "Expected single channel audio. Multichannel audio is not implemented yet."
        mix_wav = mix_wav.unsqueeze(0)  # (1, 1, T)
        text_conds_src = [config.text_prompt]
        assert mix_wav.shape[0] == len(text_conds_src), "The number of mix_wav and text_conds_src must be the same."
        output = \
            task_wrapper.source_extraction(
                mix_wav=mix_wav,
                text_conds_src=text_conds_src,  # only generated single output if length of list = 1
                return_full_output=True,
                **func_args,  # overlap_dur, cfg_scale, num_steps, repaint_n, quality_score, verbose
            )
        num_samples = mix_wav.shape[0]
        given_wav = mix_wav  # Use mix_wav as given_wav for saving
        
    elif config.task == "accomp_gen":
        """
        given_wav must be single-source audio files.
        """
        assert config.given_wav_path is not None, "For accompaniment generation, given_wav_path should be provided."
        output_dir = opj(config.output_dir, "accomp_gen")
        given_src, sr_ori = torchaudio.load(config.given_wav_path)
        given_src = torchaudio.transforms.Resample(sr_ori, config.model.sample_rate)(given_src)
        given_src = proc_audio(given_src, downsample_ratio=2048)  # Downsample ratio of the autoencoder
        # given_src: (1, T)
        assert given_src.shape[0] == 1, "Expected single channel audio. Multichannel audio is not implemented yet."
        given_src = given_src.unsqueeze(0)  # (1, 1, T) 
        text_conds_submix = [config.text_prompt]
        assert given_src.shape[0] == len(text_conds_submix), "The number of given_src and text_conds_submix must be the same."
        output = \
            task_wrapper.accompaniments_generation(
                given_src=given_src,
                text_conds_submix=text_conds_submix,  # only generated single output if length of list = 1
                return_full_output=True,
                **func_args,  # overlap_dur, cfg_scale, num_steps, repaint_n, quality_score, verbose
            )
        num_samples = len(text_conds_submix)
        given_wav = given_src  # Use given_src as given_wav for saving
        
    elif config.task == "partial_gen_iter":
        """
        task_wrapper.partial_generation_iter does not support batch processing.
        """
        assert config.given_wav_path is not None, "For partial generation iteration, given_wav_path should be provided."
        output_dir = opj(config.output_dir, "partial_gen_iter")
        ordered_text_cond = config.text_prompt.split(";")  # e.g., "text1; text2; text3"
        assert len(ordered_text_cond) >= 2, "For partial generation iteration, at least two text prompts are required."
        
        given_wav, sr_ori = torchaudio.load(config.given_wav_path)
        given_wav = torchaudio.transforms.Resample(sr_ori, config.model.sample_rate)(given_wav)
        given_wav = proc_audio(given_wav, downsample_ratio=2048)  # Downsample ratio of the autoencoder
        assert given_wav.shape[0] == 1, "Expected single channel audio. Multichannel audio is not implemented yet."
        # given_wav = given_wav.unsqueeze(0)  # (1, 1, T)
        ordered_text_cond_src = [text.strip() for text in ordered_text_cond]
        output = \
            task_wrapper.partial_generation_iterative(
                given_wav=given_wav,
                ordered_text_cond_src=ordered_text_cond_src,  # e.g., ["text1", "text2", "text3"]
                **func_args,  # overlap_dur, cfg_scale, num_steps, repaint_n, quality_score, verbose
            )
        ## output: "generated_mixture", "given_wav", "generated_sources"(list)
        given_wav = output["given_wav"]  # Use given_wav from output for saving
        output_new = {
            "gen_mix": output["generated_mixture"],
        }
        for ii, text_cond in enumerate(ordered_text_cond_src):
            output_new[f"gen_src_{ii}_{text_cond}"] = output["generated_sources"][ii]
        output = output_new
        num_samples = 1
        
    else:
        raise ValueError(f"Unknown task: {config.task}. Supported tasks are 'total_gen', 'partial_gen', 'source_extract', 'accomp_gen', 'partial_gen_iter'.")


    os.makedirs(output_dir, exist_ok=True)
    ## Save the output audio files
    for bb in range(num_samples):
        ## Temporary directory to save the output.
        # os.makedirs(output_dir, exist_ok=True)
        # output_subdir = f"output_{len(os.listdir(output_dir)) + 1:04d}"
        # os.makedirs(opj(output_dir, output_subdir), exist_ok=True)
        pattern = re.compile(r"^output_(\d{4})$")
        existing = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(opj(output_dir, d)) and pattern.match(d)
        ]
        numbers = sorted(int(pattern.match(d).group(1)) for d in existing)
        next_idx = numbers[-1] + 1 if numbers else 1
        output_subdir = f"output_{next_idx:04d}"
        os.makedirs(opj(output_dir, output_subdir), exist_ok=True)
        
        # Save the generated audio files
        for key, tensor in output.items():
            filename = f"{key}.wav"
            filepath = opj(output_dir, output_subdir, filename)
            # Save the tensor as a wav file
            wav = to_numpy(tensor[bb])
            assert wav.shape[0] == 1, "Expected single channel audio. Multichannel audio is not implemented yet."
            sf.write(filepath, wav[0], config.model.sample_rate)
            print(f"Saved {key} to {filepath}")
            save_spectrogram(wav[0], config.model.sample_rate, filepath.replace('.wav', '.png'))
            
            
        if given_wav is not None:
            if config.task == "source_extract":
                save_path = opj(output_dir, output_subdir, "given_mix.wav")
            else:
                save_path = opj(output_dir, output_subdir, "given_wav.wav")
            # Save the given wav file
            given_wav = to_numpy(given_wav[bb])
            sf.write(save_path, given_wav[0], config.model.sample_rate)
            print(f"Saved given wav to {save_path}")
            save_spectrogram(given_wav[0], config.model.sample_rate, save_path.replace('.wav', '.png'))

        with open(opj(output_dir, output_subdir, "prompt.txt"), "w") as f:
            if config.task == "partial_gen_iter":
                f.write(f"Task: {config.task}\n")
                f.write(f"Given wav: {config.given_wav_path}\n")
                f.write(f"Ordered text conditions: {config.text_prompt}\n")
            else:
                f.write(f"Task: {config.task}\n")
                f.write(f"Given wav: {config.given_wav_path}\n")
                f.write(f"Text prompt: {config.text_prompt}\n")
            f.write(f"Quality score: {quality_score}\n")
            f.write(f"Overlap duration: {config.overlap_dur} seconds\n")
            f.write(f"CFG scale: {config.cfg_scale}\n")
            f.write(f"Number of diffusion steps: {config.num_steps}\n")
            f.write(f"RePaint steps for time-axis inpainting: {config.repaint_n}\n")

def run_inference(
    ckpt_path: str,
    task: str,
    text_prompt: str = None,
    given_wav: str = None,
    out_dir: str = "./outputs",
    gen_audio_dur: float = 10.0,
    num_steps: int = 50,
    cfg_scale: float = 3.0,
    overlap_dur: float = 1.0,
    repaint_n: int = 0,
    quality_score: float = 0.5,
    device: str = "cuda:0",
    segment_duration: float = None,
    random_segment: bool = False,
    enable_attention_gating: bool = True,
    **kwargs,
):
    """
    统一推理接口：支持 total_gen, partial_gen, source_extract
    
    Args:
        quality_score: 质量分数，范围为0-1，默认为0.5（中等质量）
        enable_quality_control: 是否启用质量控制，默认为True
        segment_duration: 片段时长（秒），用于随机截取音频片段，默认为None（不截取）
        random_segment: 是否启用随机片段截取，默认为False
        enable_attention_gating: 是否启用注意力门控，必须与 checkpoint 匹配
    """
    _ensure_dir(out_dir)

    cfg = hydra.compose(config_name="default_dit")
    
    # 强制将 enable_attention_gating 传递给模型配置
    if "MGELDM" in cfg.model:
        cfg.model["MGELDM"]["enable_attention_gating"] = enable_attention_gating
    else:
        cfg.model["enable_attention_gating"] = enable_attention_gating
    
    print(f"enable_attention_gating: {enable_attention_gating}")
    
    model = create_mgeldm_from_config(cfg.model)
    ckpt = load_ckpt_state_dict(ckpt_path)
    model.load_state_dict(ckpt, strict=True)

    wrapper = InferenceTaskWrapper(
        model=model,
        segment_length_trained=cfg.model["segment_length"],
        timestep_eps=cfg.model["timestep_eps"],
        clap_ckpt_path=None,
        load_clap_audio=False,
        device=torch.device(device if torch.cuda.is_available() else "cpu"),
    )

    func_args = dict(
        overlap_dur=overlap_dur,
        cfg_scale=cfg_scale,
        num_timesteps=num_steps,
        repaint_n=repaint_n,
        verbose=True,
        quality_score=quality_score,
    )
    
    # 根据enable_quality_control参数决定是否传递quality_score

    gen_paths = {}

    def _load_and_mix_audio(paths, target_sr):
        """
        paths: str 或 list[str]
        返回: torch.Tensor, shape=(1, T)
        """
        if isinstance(paths, str):
            paths = [paths]
        audios = []
        for p in paths:
            x, sr = sf.read(p)
            if sr != target_sr:
                x = librosa.resample(y=x, orig_sr=sr, target_sr=target_sr)
            if x.ndim > 1:
                x = x.mean(axis=1)  # librosa: shape=(n_samples, n_channels)
            audios.append(x.astype(np.float32))
        max_len = max([len(a) for a in audios])
        mix = np.zeros(max_len, dtype=np.float32)
        for a in audios:
            mix[:len(a)] += a
        mix = mix / max(len(audios), 1)
        return torch.from_numpy(mix).unsqueeze(0)  # (1, T)


    if task == "total_gen":
        out = wrapper.total_mixture_generation(
            text_conds_mix=[text_prompt],
            audio_dur=gen_audio_dur,
            return_submix_src=True,
            **func_args,
        )
        wav_mix = to_numpy(out["gen_mix"][0])[0]
        mix_path = os.path.join(out_dir, "gen_mix.wav")
        sf.write(mix_path, wav_mix, cfg.model["sample_rate"])
        gen_paths["mix"] = mix_path
        for k in out:
            if k.startswith("gen_src") or k.startswith("gen_submix"):
                w = to_numpy(out[k][0])[0]
                p = os.path.join(out_dir, f"{k}.wav")
                sf.write(p, w, cfg.model["sample_rate"])
                gen_paths[k] = p

    elif task == "partial_gen":
        assert given_wav, "partial_gen 需要 given_wav 输入"
        wav = _load_and_mix_audio(given_wav, cfg.model["sample_rate"])
        wav = _proc_audio_mean_mono(wav, downsample_ratio=2048)  # (1, T)
        
        # 检查是否需要随机截取片段
        if segment_duration is not None and random_segment:
            # 使用模型训练时的segment_length来确保兼容性
            segment_length = cfg.model["segment_length"]  # 通常是100个潜在帧
            downsample_ratio = 2048  # 压缩比
            target_samples = segment_length * downsample_ratio  # 100 * 2048 = 204800个样本
            current_samples = wav.shape[1]
            
            if current_samples > target_samples:
                # 随机选择起始位置
                max_start = current_samples - target_samples
                start_idx = np.random.randint(0, max_start + 1)
                end_idx = start_idx + target_samples
                
                # 截取片段
                wav = wav[:, start_idx:end_idx]
                actual_duration = target_samples / cfg.model["sample_rate"]
                print(f"随机截取音频片段: {actual_duration:.2f}秒 (从 {start_idx/cfg.model['sample_rate']:.2f}s 到 {end_idx/cfg.model['sample_rate']:.2f}s)")
                print(f"片段长度: {target_samples}个样本 = {segment_length}个潜在帧")
            else:
                print(f"音频长度 ({current_samples/cfg.model['sample_rate']:.2f}s) 小于模型期望长度 ({actual_duration:.2f}s)，使用完整音频")
        
        wav = wav.unsqueeze(0)  # (1,1,T)
        out = wrapper.partial_generation(
            given_wav=wav,
            text_conds_src=[text_prompt],
            return_full_output=True,
            **func_args,
        )

        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.dim() == 3:
                w = to_numpy(v[0])[0]
                p = os.path.join(out_dir, f"{k}.wav")
                sf.write(p, w, cfg.model["sample_rate"])
                gen_paths[k] = p
        given_path = os.path.join(out_dir, "given_wav.wav")
        sf.write(given_path, to_numpy(wav[0])[0], cfg.model["sample_rate"])
        gen_paths["given"] = given_path

    elif task == "source_extract":
        assert given_wav, "source_extract 需要 given_wav 输入"
        mix, sr0 = torchaudio.load(given_wav)
        mix = torchaudio.transforms.Resample(sr0, cfg.model["sample_rate"])(mix)
        mix = _proc_audio_mean_mono(mix, downsample_ratio=2048).unsqueeze(0)

        out = wrapper.source_extraction(
            mix_wav=mix,
            text_conds_src=[text_prompt],
            return_full_output=True,
            **func_args,
        )

        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.dim() == 3:
                w = to_numpy(v[0])[0]
                p = os.path.join(out_dir, f"{k}.wav")
                sf.write(p, w, cfg.model["sample_rate"])
                gen_paths[k] = p
        given_path = os.path.join(out_dir, "given_mix.wav")
        sf.write(given_path, to_numpy(mix[0])[0], cfg.model["sample_rate"])
        gen_paths["given"] = given_path

    else:
        raise ValueError(f"未知任务: {task}")

    return gen_paths

if __name__=="__main__":
    # 优化 CUDA 内存分配，减少碎片
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    inference()
