import typing as tp

import torch
import torch.nn as nn

from .conditioners import MultiConditioner
from .dit import DiffusionTransformer
from .pretransforms import Pretransform


class MGELDM(nn.Module):
    def __init__(
        self,
        io_channels: int,
        num_tracks: int,
        embed_dim: int,
        global_cond_dim: int,
        global_cond_grouped: bool,
        timestep_features_dim: int,
        depth: int,
        num_heads: int,
        norm_type: tp.Literal["layernorm", "groupnorm"],
        use_t_emb_trackwise: bool,
        conditioner: MultiConditioner,
        sample_rate: int,
        diffusion_objective: tp.Literal["v", "rectified_flow"],
        pretransform: tp.Optional[Pretransform],
        global_cond_keys: tp.List[str],
        fusion_divergence_type: str = "squared_euclidean",
        fusion_temperature: float = 1.0,
        enable_quality_control: bool = True,
        enable_attention_gating: bool = True,
        num_quality_tokens: int = 1,
        quality_token_dim: int = 512,
        quality_mask_prob: float = 0.15,
        **kwargs,
    ):
        super().__init__()

        effective_num_tracks = num_tracks
        cross_cond_dim = 0

        self.model_dit = DiffusionTransformer(
            io_channels=io_channels,
            num_tracks=effective_num_tracks,
            patch_size=1,
            embed_dim=embed_dim,
            cross_cond_token_dim=cross_cond_dim,
            project_cross_cond_tokens=False,
            global_cond_dim=global_cond_dim,
            global_cond_grouped=global_cond_grouped,
            input_concat_dim=0,
            timestep_features_dim=timestep_features_dim,
            prepend_cond_dim=0,
            depth=depth,
            num_heads=num_heads,
            transformer_type="continuous_transformer",
            norm_type=norm_type,
            use_t_emb_trackwise=use_t_emb_trackwise,
            enable_quality_control=enable_quality_control,
            enable_attention_gating=enable_attention_gating,
            num_quality_tokens=num_quality_tokens,
            quality_token_dim=quality_token_dim,
            quality_mask_prob=quality_mask_prob,
            **kwargs,
        )

        self.enable_quality_control = enable_quality_control
        self.enable_attention_gating = enable_attention_gating
        self.num_quality_tokens = num_quality_tokens
        self.quality_token_dim = quality_token_dim
        self.quality_mask_prob = quality_mask_prob

        with torch.no_grad():
            for param in self.model_dit.parameters():
                param *= 0.5

        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.global_cond_keys = global_cond_keys
        self.num_tracks = num_tracks
        self.enable_quality_control = enable_quality_control
        self.enable_attention_gating = enable_attention_gating
        self.num_quality_tokens = num_quality_tokens
        self.quality_token_dim = quality_token_dim
        self.quality_mask_prob = quality_mask_prob

    def get_conditioning_inputs(
        self,
        conditioning_tensors: tp.Dict[str, tp.Any],
        specify_global_key: str = None,
        mask_tracks: tp.List[int] = [],
    ):
        if len(self.global_cond_keys) <= 0:
            raise NotImplementedError("Global conditioning keys are not defined.")

        global_conds_dict = {}
        if specify_global_key:
            mix_emb, submix_emb, src_emb = conditioning_tensors[specify_global_key]
            if mix_emb is None:
                mix_emb = torch.zeros_like(src_emb)
            if submix_emb is None:
                submix_emb = torch.zeros_like(src_emb)
            global_conds_dict[specify_global_key] = (mix_emb, submix_emb, src_emb)
        else:
            for key in self.global_cond_keys:
                mix_emb, submix_emb, src_emb = conditioning_tensors[key]
                global_conds_dict[key] = (mix_emb, submix_emb, src_emb)

        num_cond = len(global_conds_dict)
        assert num_cond > 0, "No global conditioning inputs found."

        if num_cond == 1:
            global_cond_tup = list(global_conds_dict.values())[0]
            global_cond = torch.cat(global_cond_tup, dim=1)
        elif num_cond == 2:
            mix_audemb, submix_audemb, src_audemb = global_conds_dict["audio_cond"]
            mix_textemb, submix_textemb, src_textemb = global_conds_dict["prompt_cond"]
            if src_textemb is not None:
                comb_weights = torch.rand(num_cond, device=src_audemb.device) + 1e-6
                comb_weights /= comb_weights.sum()
                src_comb_emb = comb_weights[0] * src_audemb + comb_weights[1] * src_textemb
            else:
                src_comb_emb = src_audemb
            global_cond = torch.cat([mix_audemb, submix_audemb, src_comb_emb], dim=1)
        else:
            raise ValueError(
                f"Unexpected number of global conditioning inputs: {num_cond}. Expected 1 or 2."
            )

        if len(mask_tracks) > 0:
            global_cond[:, mask_tracks] = 0.0

        quality_scores = None

        if self.enable_quality_control:
            quality_scores_raw = conditioning_tensors.get("quality_scores_raw", None)
            if quality_scores_raw is not None:
                if isinstance(quality_scores_raw, list):
                    quality_scores = torch.tensor(
                        quality_scores_raw, dtype=torch.float32, device=global_cond.device
                    )
                else:
                    quality_scores = quality_scores_raw.to(dtype=torch.float32)

        return {
            "cross_attn_cond": None,
            "cross_attn_cond_mask": None,
            "global_embed": global_cond,
            "input_concat_cond": None,
            "prepend_cond": None,
            "prepend_cond_mask": None,
            "quality_scores": quality_scores,
        }

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: tp.Dict[str, tp.Any],
        **kwargs,
    ):
        return self.model_dit(
            x=x,
            t=t,
            **self.get_conditioning_inputs(cond),
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        raise NotImplementedError("Use generate_diffusion_cond instead.")

    def prepare_for_inference(self, device):
        self.to(device)
        if hasattr(self, "conditioner"):
            self.conditioner.set_device(device)
