import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch


class AttentionSinkLogger:
    def __init__(
        self,
        log_dir: str = "./logs/attention_sink",
        record_interval: int = 100,
        max_samples_per_epoch: int = 100,
        num_layers: int = 16,
    ):
        self.log_dir = log_dir
        self.record_interval = record_interval
        self.max_samples_per_epoch = max_samples_per_epoch
        self.num_layers = num_layers

        self.first_token_attention: List[Dict] = []
        self.all_layers_attn_maps: List[Dict] = []
        self.loss_history: List[Dict] = []

        self.gated_mode = True
        self.step_count = 0
        self.epoch_count = 0
        self.force_record_next_step = False

        os.makedirs(log_dir, exist_ok=True)

    def set_gated_mode(self, gated: bool):
        self.gated_mode = gated

    def record_first_token_attention(
        self,
        layer_idx: int,
        sink_score: Union[torch.Tensor, float],
        gate_factor: float = 1.0,
        context_len: Optional[int] = None,
        branch_name: str = "unknown",
        is_cross_attention: bool = False,
    ):
        if isinstance(sink_score, torch.Tensor):
            sink_score = float(sink_score.detach().cpu().item())
        else:
            sink_score = float(sink_score)

        self.first_token_attention.append(
            {
                "layer": layer_idx,
                "is_cross_attention": is_cross_attention,
                "branch_name": branch_name,
                "sink_score": sink_score,
                "gate_factor": float(gate_factor),
                "context_len": context_len,
                "step": self.step_count,
            }
        )

        print(
            f"[FirstTokenAttn] layer={layer_idx}, branch={branch_name}, gate={gate_factor:.4f}, "
            f"context_len={context_len}, sink_score={sink_score:.6f}"
        )

    def record_attention_map(
        self,
        layer_idx: int,
        attention_map: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
        branch_name: str = "unknown",
        is_cross_attention: bool = False,
    ):
        attn_map = attention_map.detach().to(torch.float16).cpu()

        gate_info = None
        if gate is not None:
            if gate.dim() == 4:
                gate_info = gate[0].mean(dim=-1).detach().to(torch.float16).cpu()
            elif gate.dim() == 3:
                gate_info = gate[0].mean(dim=-1).detach().to(torch.float16).cpu()
            else:
                gate_info = gate.detach().to(torch.float16).cpu()

        self.all_layers_attn_maps.append(
            {
                "layer": layer_idx,
                "is_cross_attention": is_cross_attention,
                "branch_name": branch_name,
                "attn_map": attn_map,
                "gate": gate_info,
                "step": self.step_count,
            }
        )

    def record_loss(
        self,
        loss: Union[torch.Tensor, float],
        losses: Optional[Dict[str, Union[torch.Tensor, float]]] = None,
    ):
        if isinstance(loss, torch.Tensor):
            loss_value = float(loss.detach().cpu().item())
        else:
            loss_value = float(loss)

        record = {
            "step": self.step_count,
            "train_loss": loss_value,
        }

        if losses is not None:
            for name, value in losses.items():
                if isinstance(value, torch.Tensor):
                    record[name] = float(value.detach().cpu().item())
                else:
                    record[name] = float(value)

        self.loss_history.append(record)

    def should_record(self) -> bool:
        return self.force_record_next_step or self.step_count % self.record_interval == 0

    def step(self):
        self.step_count += 1

    def request_record_next_step(self):
        self.force_record_next_step = True

    def clear_forced_record(self):
        self.force_record_next_step = False

    def new_epoch(self):
        self.epoch_count += 1
        if (
            len(self.first_token_attention) > 0
            or len(self.all_layers_attn_maps) > 0
            or len(self.loss_history) > 0
        ):
            self._save_data()
        self.first_token_attention = []
        self.all_layers_attn_maps = []
        self.loss_history = []

    def flush_pending(self, tag: Optional[str] = None):
        if (
            len(self.first_token_attention) == 0
            and len(self.all_layers_attn_maps) == 0
            and len(self.loss_history) == 0
        ):
            return
        self._save_data(tag=tag)

    def _save_data(self, tag: Optional[str] = None):
        suffix = "_gated" if self.gated_mode else "_ungated"
        tag_suffix = f"_{tag}" if tag else ""

        curve_path = os.path.join(
            self.log_dir,
            f"attention_sink_curve{suffix}_epoch{self.epoch_count}{tag_suffix}.csv",
        )
        with open(curve_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "layer",
                    "branch_name",
                    "is_cross_attention",
                    "sink_score",
                    "gate_factor",
                    "context_len",
                ],
            )
            writer.writeheader()
            writer.writerows(self.first_token_attention)

        attn_path = os.path.join(
            self.log_dir,
            f"all_layers_attn{suffix}_epoch{self.epoch_count}{tag_suffix}.pt",
        )
        torch.save(
            {
                "data": self.all_layers_attn_maps,
                "config": {
                    "gated_mode": self.gated_mode,
                    "record_interval": self.record_interval,
                    "num_layers": self.num_layers,
                    "prefix_order": ["timestep", "quality", "audio"],
                },
            },
            attn_path,
        )

        if self.loss_history:
            loss_keys = sorted(
                {
                    key
                    for record in self.loss_history
                    for key in record.keys()
                    if key != "step"
                }
            )
            loss_path = os.path.join(
                self.log_dir,
                f"loss_curve{suffix}_epoch{self.epoch_count}{tag_suffix}.csv",
            )
            with open(loss_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["step"] + loss_keys)
                writer.writeheader()
                writer.writerows(self.loss_history)

        print(f"[AttentionSinkLogger] Saved data to {self.log_dir}")

    def get_summary_stats(self) -> Dict:
        if len(self.first_token_attention) == 0:
            return {}

        layer_branch_stats = defaultdict(lambda: defaultdict(list))
        for record in self.first_token_attention:
            key = (record["layer"], record["branch_name"])
            layer_branch_stats[key]["attn"].append(record["sink_score"])
            layer_branch_stats[key]["gate"].append(record.get("gate_factor", 1.0))

        return {
            "gated_mode": self.gated_mode,
            "total_samples": len(self.first_token_attention),
            "layer_branch_stats": {
                f"layer_{layer}_{branch}": {
                    "attn_mean": float(np.mean(values["attn"])),
                    "attn_std": float(np.std(values["attn"])),
                    "gate_mean": float(np.mean(values["gate"])),
                    "count": len(values["attn"]),
                }
                for (layer, branch), values in layer_branch_stats.items()
            },
        }


_global_logger: Optional[AttentionSinkLogger] = None


def get_attention_logger() -> Optional[AttentionSinkLogger]:
    return _global_logger


def set_attention_logger(logger: AttentionSinkLogger):
    global _global_logger
    _global_logger = logger


def create_attention_logger(
    log_dir: str = "./logs/attention_sink",
    record_interval: int = 100,
    num_layers: int = 16,
) -> AttentionSinkLogger:
    global _global_logger
    _global_logger = AttentionSinkLogger(
        log_dir=log_dir,
        record_interval=record_interval,
        num_layers=num_layers,
    )
    return _global_logger
