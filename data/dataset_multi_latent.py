import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate


import numpy as np
import os; opj = os.path.join
from glob import glob
from tqdm import tqdm
from time import time
import yaml
import json

import typing as tp

NUM_TRIES=15

from .dataset_single import ConcatDataset


def get_salient_latents(
    *,
    latent: tp.Union[np.ndarray, str], ## path or latent (64, length)
    segment_len: int, # 80
    num_tries,
    latent_zero, #(64)
    threshold, # (0.07)
    state: np.random.RandomState,
):
    """
    Load non-silent latent as possible
    """
    if isinstance(latent, str):
        latent = np.load(latent)
    
    if latent_zero is not None:
        latent_zero = latent_zero.reshape(-1, 1) # (64, 1)
    else:
        assert threshold is None, "If latent_zero is None, threshold must be None"
    
    total_len = latent.shape[-1]
    lower_bound = 0
    upper_bound = max(total_len - segment_len, 0)
    
    if threshold is None:
        offset_st = state.randint(lower_bound, upper_bound)
        offset_end = offset_st + segment_len
        latent_excerpt = latent[..., offset_st:offset_end]
    else:
        diff_abs = -np.inf
        num_try = 0
        while diff_abs <= threshold:
            offset_st = state.randint(lower_bound, upper_bound)
            offset_end = offset_st + segment_len
            latent_excerpt = latent[..., offset_st:offset_end] # (64, 80)
            diff_abs = np.abs(np.mean(latent_excerpt-latent_zero)).item()
            num_try += 1
            if num_tries is not None and num_try >= num_tries:
                break
    
    return latent_excerpt, offset_st, offset_end


def get_prompt_from_labels(label_list, state=None):
    """
    Convert a list of labels into a single natural-language prompt.
    - Replace '_' with space, lowercase everything.
    - Join multiple labels with commas and 'and'.
    - Pick one of several templates at random, using the provided RNG state.

    Args:
        label_list (list of str): e.g. ["vocals", "drums", "electric_guitar"]
        state (np.random.RandomState or None): if None, a new RandomState is created.
    Returns:
        str: a prompt string.
    """
    # 1) RNG setup
    rng = state if state is not None else np.random.RandomState()

    # 2) Normalize labels
    labels = [lab.replace('_', ' ').lower() for lab in label_list]

    # 3) Join into a natural list string
    if len(labels) == 0:
        raise ValueError("Label list is empty, cannot generate prompt.")
    elif len(labels) == 1:
        joined = labels[0]
    elif len(labels) == 2:
        joined = f"{labels[0]} and {labels[1]}"
    else:
        joined = ", ".join(labels[:-1]) + f", and {labels[-1]}"

    # 4) Prompt templates (instrument/stem only, no verbs)
    
    templates = [
        "The sound of {labels}.",
        "The instrumental sound of {labels}.",
    ]
    
    templates += [
        "Only {labels}.",
        "{labels} sounds.",
        "Instrumental {labels}.",
        "Stem of {labels}.",
        "Pure {labels}.",
        "Audio of {labels}.",
        "{labels} part.",
        "{labels} stem.",
        "{labels} segment."
    ]
    
    templates += [
        "{labels}",
        "Only {labels}",
        "{labels} stem",
        "{labels} track",
        "{labels} audio",
        "Stem: {labels}",
        "Track: {labels}",
        "Audio: {labels}",
    ]
    templates += [
        "The music of {labels}.",
        "The performance of {labels}.",
        "The track of {labels}.",
        "The audio track of {labels}.",
        "The {labels}",
        "Audio elements: {labels}.",
    ]

    # 5) Select one template at random
    template = rng.choice(templates)
    return template.format(labels=joined)

class MixSubmixSourceLatentDataset(Dataset):
    def __init__(
        self,
        *,
        data_dir,
        segment_length,
        split="train",
        num_examples=10000,
        zero_diff_threshold=0.1,
        zero_latent_path=None,
        num_tracks_debug=None,
        state_id=None,
        # return_src_txt=True,
        return_mix_only=False, ## For mixture only pre-training
        ## For duration calculation
        ae_comp_ratio=2048,
        ae_sample_rate=16000,
        global_cocola_min=None,
        global_cocola_max=None,
    ):
        super().__init__()
        assert zero_latent_path is not None, "zero_latent_path must be provided"

        self.dataset_name = os.path.basename(data_dir)
        self.global_cocola_min = global_cocola_min
        self.global_cocola_max = global_cocola_max
        
        if split is not None:
            self.data_dir = opj(data_dir, split)
        else:
            self.data_dir = data_dir
        self.segment_length = segment_length
        self.num_examples = num_examples
        self.zero_diff_threshold = zero_diff_threshold
        self.zero_latent_path = zero_latent_path
        self.num_tracks_debug = num_tracks_debug
        self.state_id = state_id
        # self.return_src_txt = return_src_txt
        self.return_mix_only = return_mix_only
        self.ae_comp_ratio = ae_comp_ratio
        self.ae_sample_rate = ae_sample_rate
        
        if self.state_id is not None:
            self.state = np.random.RandomState(self.state_id)
        else:
            self.state = None
            
        self.latent_zero = np.load(self.zero_latent_path) if self.zero_latent_path else None

        self.tracks = os.listdir(self.data_dir)
        self.tracks = sorted(self.tracks)
        if self.num_tracks_debug is not None:
            self.tracks = self.tracks[:self.num_tracks_debug]
        
        self._preload_dataset()
        
        if len(self.data_dict) == 0:
            raise ValueError(
                f"No data for dataset dir: {data_dir}. "
            )
            
        self.latent_to_dur_ratio = ae_comp_ratio / ae_sample_rate
        
        

        

    def _preload_dataset(self):
        """
        track00000
        - mix_clap.npy: (512, num_sec)
        - mix.npy: (64, T)
        + comb0
            - src.npy: (64, T)
            - src_clap.npy: (512, num_sec)
            - submix.npy: (64, T)
            - submix_clap.npy: (512, num_sec)
            - comb_info.json:
                {
                    "src_label": "drums",
                    "submix_label_list": [
                        "bass",
                        "piano",
                        "electric_guitar"
                    ],
                    "wav_sr_ori": 44100,
                    "sample_rate_ae": 16000,
                    "sample_rate_clap": 48000
                }
        + comb1
            - ...
        """
        """
        We load all the data into memory.
        """
        # 检查是否提供了全局 cocola 范围
        if self.global_cocola_min is not None and self.global_cocola_max is not None:
            min_val = self.global_cocola_min
            max_val = self.global_cocola_max
            print(f"使用全局 cocola 归一化范围: [{min_val}, {max_val}]")
        else:
            # 回退到原来的逻辑：扫描当前数据集的 cocola 分数范围
            print("未提供全局 cocola 范围，使用数据集内部范围...")
            all_cocola_scores = []

            # 先扫描一遍，收集所有cocola分数
            print("第一阶段：扫描cocola分数范围...")
            for track in self.tracks:
                track_dir = opj(self.data_dir, track)
                if not os.path.exists(track_dir):
                    continue

                if self.return_mix_only is False:
                    comb_dir_list = glob(opj(track_dir, "comb*"))

                    for comb_dir in comb_dir_list:
                        if not os.path.exists(opj(comb_dir, "comb_info.json")):
                            continue

                        with open(opj(comb_dir, "comb_info.json"), "r") as f:
                            try:
                                comb_info = json.load(f)
                            except:
                                continue

                        # 如果有cocola_score，收集它
                        if "cocola_score" in comb_info and "other" not in comb_info.get("src_label", "").lower():
                            cocola_val = comb_info["cocola_score"]
                            all_cocola_scores.append(cocola_val)

            # 计算最大值和最小值
            if all_cocola_scores:
                min_cocola = min(all_cocola_scores)
                max_cocola = max(all_cocola_scores)
                avg_cocola = sum(all_cocola_scores) / len(all_cocola_scores)

                # 计算中位数和百分位数
                sorted_scores = sorted(all_cocola_scores)
                median_cocola = sorted_scores[len(sorted_scores) // 2]
                p25_cocola = sorted_scores[len(sorted_scores) // 4]
                p75_cocola = sorted_scores[3 * len(sorted_scores) // 4]

                # 向下取整最小值，向上取整最大值
                min_val = np.floor(min_cocola)
                max_val = np.ceil(max_cocola)

                print("\n" + "="*50)
                print(f"Cocola分数统计信息（共{len(all_cocola_scores)}个样本）：")
                print(f"- 范围：最小值={min_val} (原始{min_cocola:.4f})，最大值={max_val} (原始{max_cocola:.4f})")
                print(f"- 平均值：{avg_cocola:.4f}")
                print(f"- 中位数：{median_cocola:.4f}")
                print(f"- 25%分位数：{p25_cocola:.4f}，75%分位数：{p75_cocola:.4f}")
                print(f"- 归一化范围：[{min_val}, {max_val}]")

                # 输出分布直方图
                bins = 10
                hist, bin_edges = np.histogram(all_cocola_scores, bins=bins)
                print("\nCocola分数分布直方图：")
                for i in range(bins):
                    bar_len = int(hist[i] / len(all_cocola_scores) * 50)
                    print(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}: {'#' * bar_len} ({hist[i]})")
                print("="*50 + "\n")
            else:
                min_val, max_val = 0.0, 100.0  # 默认范围
                print("未找到cocola分数，使用默认范围[0, 100]")
        
        # 第二阶段：加载数据并使用找到的范围进行归一化
        self.data_dict = {}
        pbar = tqdm(self.tracks, desc=f"Preloading {self.dataset_name} dataset", unit="track")
        
        for track in pbar:
            self.data_dict[track] = {}
            track_dir = opj(self.data_dir, track)
            zmix = np.load(opj(track_dir, "mix.npy"))  
            cmix = np.load(opj(track_dir, "mix_clap.npy"))
            if np.isnan(zmix).any() or np.isnan(cmix).any():
                print(f"NaN found in {track}. Skipping this track.")
                continue
            
            self.data_dict[track]["mix"] = zmix
            self.data_dict[track]["mix_clap"] = cmix
            
            if self.return_mix_only is False:
                comb_dir_list = glob(opj(track_dir, "comb*"))
                valid_combs_found = False  # 跟踪是否找到有效组合
                
                for comb_dir in comb_dir_list:
                    comb_name = os.path.basename(comb_dir) 
                    
                    with open(opj(comb_dir, "comb_info.json"), "r") as f:
                        comb_info = json.load(f)
                    
                    # 如果'other'在src标签中或没有cocola_score，跳过此组合
                    if "other" in comb_info["src_label"].lower() or "cocola_score" not in comb_info:
                        continue
                    
                    # 读取cocola_score
                    cocola_val = comb_info["cocola_score"]
                    
                    # 使用找到的范围进行归一化
                    normalized_cocola = (cocola_val - min_val) / (max_val - min_val)
                    normalized_cocola = max(0.0, min(1.0, normalized_cocola))  # 确保在[0,1]范围内
                    
                    # 每100个样本输出一次归一化结果
                    sample_count = sum(len(self.data_dict[t]) for t in self.data_dict)
                    if sample_count % 100 == 0:
                        print(f"样本 #{sample_count}: 原始值={cocola_val:.4f}, 归一化值={normalized_cocola:.4f}")
                    
                    # 归一化完成
                    
                    zsrc = np.load(opj(comb_dir, "src.npy"))
                    csrc = np.load(opj(comb_dir, "src_clap.npy"))
                    zsubmix = np.load(opj(comb_dir, "submix.npy"))
                    csubmix = np.load(opj(comb_dir, "submix_clap.npy"))
                    
                    if np.isnan(zsrc).any() or np.isnan(csrc).any() or \
                       np.isnan(zsubmix).any() or np.isnan(csubmix).any():
                        print(f"NaN found in {comb_name} of {track}. Skipping this combination.")
                        continue
                    
                    src_inst = comb_info["src_label"]
                    submix_inst = comb_info["submix_label_list"]
                    
                    # 创建组合字典并存储数据
                    self.data_dict[track][comb_name] = {}
                    self.data_dict[track][comb_name]["cocola_score"] = normalized_cocola
                    self.data_dict[track][comb_name]["cocola_score_raw"] = cocola_val  # 保存原始值，便于调试
                    self.data_dict[track][comb_name]["src"] = zsrc
                    self.data_dict[track][comb_name]["src_clap"] = csrc
                    self.data_dict[track][comb_name]["submix"] = zsubmix
                    self.data_dict[track][comb_name]["submix_clap"] = csubmix
                    self.data_dict[track][comb_name]["src_inst"] = src_inst
                    self.data_dict[track][comb_name]["submix_inst"] = submix_inst
                    
                    valid_combs_found = True
                
                # 如果没有找到有效组合，删除整个轨道
                if not valid_combs_found and not self.return_mix_only:
                    del self.data_dict[track]
        
        # 检查是否有足够的数据
        if len(self.data_dict) == 0:
            raise ValueError(f"No valid data for dataset dir: {self.data_dir}. All combinations either lack cocola_score or contain 'other' in src_label.")
        
        # 输出最终统计信息
        all_normalized_scores = []
        for track in self.data_dict:
            for comb in self.data_dict[track]:
                if comb.startswith("comb") and "cocola_score" in self.data_dict[track][comb]:
                    all_normalized_scores.append(self.data_dict[track][comb]["cocola_score"])
        
        if all_normalized_scores:
            print("\n" + "="*50)
            print(f"归一化后的Cocola分数统计（共{len(all_normalized_scores)}个样本）：")
            print(f"- 最小值：{min(all_normalized_scores):.4f}")
            print(f"- 最大值：{max(all_normalized_scores):.4f}")
            print(f"- 平均值：{sum(all_normalized_scores)/len(all_normalized_scores):.4f}")
            
            # 输出归一化后的分布直方图
            bins = 10
            hist, bin_edges = np.histogram(all_normalized_scores, bins=bins, range=(0, 1))
            print("\n归一化后的分布直方图：")
            for i in range(bins):
                bar_len = int(hist[i] / len(all_normalized_scores) * 50)
                print(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}: {'#' * bar_len} ({hist[i]})")
            print("="*50 + "\n")
        
        return self.data_dict
        
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        if self.state:
            state = self.state
        else:
            state = np.random.RandomState((int(time()) % (2 ** 32)-1)+idx)
        
        # 从已加载的轨道中选择，而不是从所有轨道中选择
        available_tracks = list(self.data_dict.keys())
        if not available_tracks:
            raise ValueError(f"No available tracks in data_dict. This might indicate that no tracks have valid combinations with cocola_score.")
        
        track = state.choice(available_tracks)
        track_data = self.data_dict[track]
        
        
        ## Load mix
        zmix = track_data["mix"]
        cmix = track_data["mix_clap"]
        
        ## Load src, submix
        if self.return_mix_only:
            ## Random segment from mix
            zmix, offset_st, offset_end = get_salient_latents(
                latent=zmix,
                segment_len=self.segment_length,
                num_tries=NUM_TRIES,
                latent_zero=self.latent_zero,
                threshold=self.zero_diff_threshold,
                state=state
            )
            clap_len = cmix.shape[-1]
            offset_clap = int(offset_st * self.latent_to_dur_ratio)
            offset_clap = min(offset_clap, clap_len - 1)
            cmix = cmix[:, offset_clap]
        
            zsrc = np.zeros_like(zmix)
            csrc = np.zeros_like(cmix)
            src_prompt = ""
            
            zsubmix = zmix
            csubmix = cmix
            submix_prompt = ""
        else:
            # 获取可用的组合名称列表
            comb_name_list = [c for c in track_data.keys() if c.startswith("comb")]
            if not comb_name_list:
                # 这种情况不应该发生，因为我们已经在_preload_dataset中过滤了没有有效组合的轨道
                # 但为了安全起见，我们还是添加一个检查
                raise ValueError(f"Track {track} has no valid combinations. This should not happen after filtering in _preload_dataset.")
            
            comb_name = state.choice(comb_name_list)
            comb_data = track_data[comb_name] 
            
            zsrc = comb_data["src"]
            csrc = comb_data["src_clap"]
            src_inst = comb_data["src_inst"]
            
            zsubmix = comb_data["submix"]
            csubmix = comb_data["submix_clap"]
            submix_inst = comb_data["submix_inst"]
            
            ## sanity check
            assert zsrc.shape == zsubmix.shape == zmix.shape
            assert csrc.shape == csubmix.shape == cmix.shape
            
            ## Salient latent for src
            zsrc, offset_st, offset_end = get_salient_latents(
                latent=zsrc,
                segment_len=self.segment_length,
                num_tries=NUM_TRIES,
                latent_zero=self.latent_zero,
                threshold=self.zero_diff_threshold,
                state=state
            )
            
            clap_len = csrc.shape[-1]
            offset_clap = int(offset_st * self.latent_to_dur_ratio) ## => "seconds"
            offset_clap = min(offset_clap, clap_len - 1)
            csrc = csrc[:, offset_clap]
            
            zsubmix = zsubmix[..., offset_st:offset_end]
            csubmix = csubmix[:, offset_clap]
            
            zmix = zmix[..., offset_st:offset_end]
            cmix = cmix[:, offset_clap]
            
            ## Get prompts
            src_prompt = get_prompt_from_labels([src_inst], state=state)
            submix_prompt = get_prompt_from_labels(submix_inst, state=state)
            
        items = {
            "mix_latent": zmix,
            "mix_clap": cmix,
            
            "src_latent": zsrc,
            "src_clap": csrc,
            "src_prompt": src_prompt,
            
            "submix_latent": zsubmix,
            "submix_clap": csubmix,
            "submix_prompt": submix_prompt,
        }
        
        # 添加cocola_score到返回项
        if not self.return_mix_only:
            cocola_score = comb_data.get("cocola_score", None)
            cocola_score_raw = comb_data.get("cocola_score_raw", None)
            items["cocola_score"] = cocola_score
            items["cocola_score_raw"] = cocola_score_raw  # 添加原始值，便于调试
            
        return items
    

class MTGJamendoLatentDataset(Dataset):
    def __init__(
        self,
        *,
        data_dir,
        segment_length,
        split="train",
        num_examples=10000,
        zero_diff_threshold=0.1,
        zero_latent_path=None,
        num_tracks_debug=None,
        state_id=None,
        # return_src_txt=True,
        return_mix_only=True, ## For mixture only pre-training
        ## For duration calculation
        ae_comp_ratio=2048,
        ae_sample_rate=16000,
    ):
        super().__init__()
        assert zero_latent_path is not None, "zero_latent_path must be provided"
        assert return_mix_only is  True
        
        self.dataset_name = os.path.basename(data_dir)
        
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.num_examples = num_examples
        self.zero_diff_threshold = zero_diff_threshold
        self.zero_latent_path = zero_latent_path
        self.num_tracks_debug = num_tracks_debug
        self.state_id = state_id
        # self.return_src_txt = return_src_txt
        self.return_mix_only = return_mix_only
        self.ae_comp_ratio = ae_comp_ratio
        self.ae_sample_rate = ae_sample_rate
        
        if self.state_id is not None:
            self.state = np.random.RandomState(self.state_id)
        else:
            self.state = None
            
        self.latent_zero = np.load(self.zero_latent_path) if self.zero_latent_path else None
        
        self.idx_dirs = sorted(os.listdir(data_dir)) ## 00~99
        assert len(self.idx_dirs) == 100
        if split == "train":
            self.idx_dirs = self.idx_dirs[:98]
        elif split in ["valid", "validation"]:
            self.idx_dirs = self.idx_dirs[98:]
        else: 
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'valid'.")

        if num_tracks_debug is not None:
            self.idx_dirs = self.idx_dirs[:num_tracks_debug]

        # self.tracks = os.listdir(self.data_dir)
        # self.tracks = sorted(self.tracks)
        # if self.num_tracks_debug is not None:
        #     self.tracks = self.tracks[:self.num_tracks_debug]
        
        self._preload_dataset()
        
        if len(self.data_dict) == 0:
            raise ValueError(
                f"No data for dataset dir: {data_dir}. "
            )
            
        self.latent_to_dur_ratio = ae_comp_ratio / ae_sample_rate
        
        
    def _preload_dataset(self):
        """
        track00000
        - mix_clap.npy: (512, num_sec)
        - mix.npy: (64, T)
        """
        """
        We load all the data into memory.
        """
        self.data_dict = {}
        
        pbar = tqdm(self.idx_dirs, desc=f"Preloading {self.dataset_name} dataset", unit="directory")
        # pbar = tqdm(self.tracks, desc=f"Preloading {self.dataset_name} dataset", unit="track")
        data_count = 0
        for idx_dir in pbar:
            trackdir_list = os.listdir(opj(self.data_dir, idx_dir))
            for trackdir in trackdir_list:
                zmix = np.load(opj(self.data_dir, idx_dir, trackdir, "mix.npy"))
                cmix = np.load(opj(self.data_dir, idx_dir, trackdir, "mix_clap.npy"))
                
                ## IF there is NaN in the mix, we skip this track
                if np.isnan(zmix).any() or np.isnan(cmix).any():
                    print(f"NaN found in {idx_dir}/{trackdir}. Skipping this track.")
                    continue
                
                self.data_dict[f"{idx_dir}_{trackdir}"] = {}
                self.data_dict[f"{idx_dir}_{trackdir}"]["mix"] = zmix
                self.data_dict[f"{idx_dir}_{trackdir}"]["mix_clap"] = cmix
                data_count += 1
        return self.data_dict            
        
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        if self.state:
            state = self.state
        else:
            state = np.random.RandomState((int(time()) % (2 ** 32)-1)+idx)
        
        data_key = state.choice(list(self.data_dict.keys()))
        data = self.data_dict[data_key]
        
        ## Load mix
        zmix = data["mix"]
        cmix = data["mix_clap"]
        
        # print(""f"Mix shape: {zmix.shape}, Clap shape: {cmix.shape}")
        
        ## Random segment from mix
        zmix, offset_st, offset_end = get_salient_latents(
            latent=zmix,
            segment_len=self.segment_length,
            num_tries=NUM_TRIES,
            latent_zero=self.latent_zero,
            threshold=self.zero_diff_threshold,
            state=state
        )
        clap_len = cmix.shape[-1]
        offset_clap = int(offset_st * self.latent_to_dur_ratio)
        offset_clap = min(offset_clap, clap_len - 1)
        cmix = cmix[:, offset_clap]
    
        zsrc = np.zeros_like(zmix)
        csrc = np.zeros_like(cmix)
        src_prompt = ""
        
        zsubmix = zmix
        csubmix = cmix
        submix_prompt = ""
        
        ## Get mix prompt
        items = {
            "mix_latent": zmix,
            "mix_clap": cmix,
            
            "src_latent": zsrc,
            "src_clap": csrc,
            "src_prompt": src_prompt,
            
            "submix_latent": zsubmix,
            "submix_clap": csubmix,
            "submix_prompt": submix_prompt,
        }
        
        return items

def collate_fn_multi_latent(batch):
    collated = {}
    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch]

        collated[key] = default_collate(values)

    return collated
    
            

def compute_global_cocola_range(dataset_configs_split):
    """
    计算所有数据集的全局 cocola 分数范围
    dataset_configs_split: 已经按split分割后的数据集配置
    """
    print("计算全局 cocola 分数范围...")

    all_cocola_scores = []

    for dataset_name, dataset_cfg in dataset_configs_split.items():
        if dataset_name == "mtg_jamendo":
            # MTG Jamendo 不使用 cocola 分数，跳过
            continue

        data_dir = dataset_cfg["data_dir"]
        split_name = dataset_cfg.get("split", None)

        print(f"扫描数据集 {dataset_name} ({split_name})...")

        # 构建数据集目录路径
        if split_name:
            dataset_dir = opj(data_dir, split_name)
        else:
            dataset_dir = data_dir

        if not os.path.exists(dataset_dir):
            print(f"警告：数据集目录不存在 {dataset_dir}")
            continue

        # 扫描所有轨道
        tracks = os.listdir(dataset_dir)
        for track in tracks:
            track_dir = opj(dataset_dir, track)
            if not os.path.exists(track_dir) or not track.startswith("track"):
                continue

            # 查找所有 comb 目录
            comb_dirs = [d for d in os.listdir(track_dir) if d.startswith("comb")]
            for comb_dir in comb_dirs:
                comb_path = opj(track_dir, comb_dir)
                comb_info_path = opj(comb_path, "comb_info.json")

                if not os.path.exists(comb_info_path):
                    continue

                try:
                    with open(comb_info_path, "r") as f:
                        comb_info = json.load(f)

                    # 收集 cocola 分数（排除包含 'other' 的样本）
                    if "cocola_score" in comb_info and "other" not in comb_info.get("src_label", "").lower():
                        cocola_val = comb_info["cocola_score"]
                        all_cocola_scores.append(cocola_val)
                except:
                    continue

    # 计算全局范围
    if all_cocola_scores:
        min_cocola = min(all_cocola_scores)
        max_cocola = max(all_cocola_scores)
        avg_cocola = sum(all_cocola_scores) / len(all_cocola_scores)

        # 计算中位数和百分位数
        sorted_scores = sorted(all_cocola_scores)
        median_cocola = sorted_scores[len(sorted_scores) // 2]
        p25_cocola = sorted_scores[len(sorted_scores) // 4]
        p75_cocola = sorted_scores[3 * len(sorted_scores) // 4]

        # 向下取整最小值，向上取整最大值
        min_val = np.floor(min_cocola)
        max_val = np.ceil(max_cocola)

        print("\n" + "="*60)
        print(f"全局 Cocola 分数统计信息（共{len(all_cocola_scores)}个样本）：")
        print(f"- 范围：最小值={min_val} (原始{min_cocola:.4f})，最大值={max_val} (原始{max_cocola:.4f})")
        print(f"- 平均值：{avg_cocola:.4f}")
        print(f"- 中位数：{median_cocola:.4f}")
        print(f"- 25%分位数：{p25_cocola:.4f}，75%分位数：{p75_cocola:.4f}")
        print(f"- 全局归一化范围：[{min_val}, {max_val}]")
        print("="*60 + "\n")

        return min_val, max_val
    else:
        print("未找到 cocola 分数，使用默认范围 [0, 100]")
        return 0.0, 100.0


def create_multi_latent_dataloader_from_config(
    dataset_configs,
    split: str,
    batch_size: int,
    sample_rate: int,
    segment_length: int,
    mixture_only_pretraining: bool = False,

    ae_comp_ratio: int = None,
    ## loader params
    num_workers: int = 8,
    shuffle: bool = True,
):
    dataset_list = []
    dataset_configs = dataset_configs[split]

    # 只有在非 mixture_only_pretraining 模式下才需要计算全局 cocola 范围
    global_cocola_min = None
    global_cocola_max = None

    if not mixture_only_pretraining:
        global_cocola_min, global_cocola_max = compute_global_cocola_range(dataset_configs)

    for dataset_name, dataset_cfg in dataset_configs.items():
        if dataset_name == "mtg_jamendo":
            assert mixture_only_pretraining, "MTG Jamendo dataset is only for mixture-only pretraining"
            dset_cls = MTGJamendoLatentDataset
        else:
            dset_cls = MixSubmixSourceLatentDataset

        dataset = dset_cls(
            data_dir=dataset_cfg["data_dir"],
            segment_length=segment_length,
            split=dataset_cfg.get("split", None),
            num_examples=dataset_cfg["num_examples"],
            zero_diff_threshold=dataset_cfg["zero_diff_threshold"],
            zero_latent_path=dataset_cfg["zero_latent_path"],
            num_tracks_debug=dataset_cfg.get("num_tracks_debug", None),
            state_id=dataset_cfg.get("state_id", None),
            return_mix_only=mixture_only_pretraining,
            ae_comp_ratio=ae_comp_ratio,
            ae_sample_rate=sample_rate,
            # 传递全局 cocola 范围
            global_cocola_min=global_cocola_min,
            global_cocola_max=global_cocola_max,
        )

        dataset_list.append(dataset)
    
    dataset_cat = ConcatDataset(dataset_list, shuffle=shuffle)
    dataloader = DataLoader(
        dataset_cat,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_multi_latent,
        pin_memory=True,
        drop_last=True,  # Ensure consistent batch size
    )
    return dataloader
