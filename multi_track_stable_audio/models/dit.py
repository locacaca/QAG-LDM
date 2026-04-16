import typing as tp

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F

from .layers import FourierFeatures
from .transformer import ContinuousTransformer, QualityTokenGenerator
from ..utils import exists


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        io_channels: int,
        num_tracks: int = 3,
        patch_size: int = 1,
        embed_dim: int,
        cross_cond_token_dim: int = 0,
        project_cross_cond_tokens: bool = True,
        global_cond_dim: int,
        global_cond_grouped: bool = True,
        input_concat_dim: int = 0,
        timestep_features_dim: int = 256,
        prepend_cond_dim: int = 0,
        depth: int,
        num_heads: int,
        transformer_type: tp.Literal["x-transformers", "continuous_transformer"] = "continuous_transformer",
        norm_type: tp.Literal["layernorm", "groupnorm"] = "groupnorm",
        use_t_emb_trackwise: bool = True,
        enable_crossatten: bool = False,
        num_quality_tokens: int = 1,
        quality_token_dim: int = 512,
        quality_mask_prob: float = 0.15,
        **kwargs,
    ):
        super().__init__()
        assert transformer_type == "continuous_transformer", "Only continuous_transformer is supported now."
        assert patch_size == 1, "Only patch_size=1 is supported now."

        self.patch_size = patch_size
        self.cross_cond_token_dim = cross_cond_token_dim
        self.input_concat_dim = input_concat_dim
        self.num_tracks = num_tracks
        self.io_channels = io_channels
        self.use_t_emb_trackwise = use_t_emb_trackwise
        self.quality_mask_prob = quality_mask_prob

        dim_in = num_tracks * io_channels + input_concat_dim
        dim_out = num_tracks * io_channels
        embed_dim_mult = embed_dim * num_tracks
        t_embed_dim = embed_dim if use_t_emb_trackwise else embed_dim_mult

        self.global_cond_grouped = global_cond_grouped
        self.timestep_features = FourierFeatures(1, timestep_features_dim)
        conv_cls = nn.Conv1d

        if use_t_emb_trackwise:
            self.to_timestep_embed = nn.Sequential(
                Rearrange("b n e -> b (n e) 1"),
                conv_cls(
                    timestep_features_dim * num_tracks,
                    t_embed_dim * num_tracks,
                    kernel_size=1,
                    groups=num_tracks,
                    bias=True,
                ),
                nn.SiLU(),
                conv_cls(
                    t_embed_dim * num_tracks,
                    t_embed_dim * num_tracks,
                    kernel_size=1,
                    groups=num_tracks,
                    bias=True,
                ),
                Rearrange("b ne 1 -> b ne"),
            )
        else:
            self.to_timestep_embed = nn.Sequential(
                nn.Linear(timestep_features_dim, t_embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(t_embed_dim, t_embed_dim, bias=True),
            )

        if cross_cond_token_dim > 0:
            # Keep cross-attention context in the original 512-dim CLAP/quality space.
            cross_cond_embed_dim = cross_cond_token_dim
            self.to_cross_cond_embed = nn.Identity()
        else:
            cross_cond_embed_dim = 0
            self.to_cross_cond_embed = None

        if global_cond_dim > 0:
            if global_cond_grouped is False:
                self.to_global_embed = nn.Sequential(
                    Rearrange("b n e -> b (n e)"),
                    nn.Linear(global_cond_dim * num_tracks, global_cond_dim * num_tracks, bias=False),
                    nn.SiLU(),
                    nn.Linear(global_cond_dim * num_tracks, embed_dim_mult, bias=False),
                )
            else:
                self.to_global_embed = nn.Sequential(
                    Rearrange("b n e -> b (n e) 1"),
                    conv_cls(
                        global_cond_dim * num_tracks,
                        global_cond_dim * num_tracks,
                        kernel_size=1,
                        groups=num_tracks,
                        bias=False,
                    ),
                    nn.SiLU(),
                    conv_cls(
                        global_cond_dim * num_tracks,
                        embed_dim_mult,
                        kernel_size=1,
                        groups=num_tracks,
                        bias=False,
                    ),
                    Rearrange("b ne 1 -> b ne"),
                )

        if prepend_cond_dim > 0:
            raise NotImplementedError("Prepend condition is not supported in MGE-LDM yet.")

        norm_kwargs = {"num_groups": num_tracks} if norm_type == "groupnorm" else {}

        self.transformer = ContinuousTransformer(
            dim=embed_dim_mult,
            depth=depth,
            dim_heads=embed_dim_mult // num_heads,
            dim_in=dim_in * patch_size,
            dim_out=dim_out * patch_size,
            cross_attend=cross_cond_token_dim > 0,
            cond_token_dim=cross_cond_embed_dim,
            global_cond_dim=embed_dim_mult,
            norm_type=norm_type,
            global_cond_group=num_tracks if global_cond_grouped else 1,
            norm_kwargs=norm_kwargs,
            use_crossatten=enable_crossatten,
            dim_quality=quality_token_dim,
            num_quality_tokens=num_quality_tokens,
            **kwargs,
        )

        self.enable_crossatten = enable_crossatten
        if enable_crossatten:
            self.quality_token_generator = QualityTokenGenerator(
                quality_dim=1,
                num_tokens=num_quality_tokens,
                token_dim=embed_dim_mult,
                hidden_dim=quality_token_dim,
            )
            self.quality_null_token = nn.Parameter(torch.zeros(1, num_quality_tokens, embed_dim_mult))
            nn.init.normal_(self.quality_null_token, std=0.02)
            self.quality_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim_mult))
            nn.init.normal_(self.quality_mask_token, std=0.02)
            self.quality_score_predictor = nn.Sequential(
                nn.LayerNorm(embed_dim_mult),
                nn.Linear(embed_dim_mult, 1),
            )

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(dim_out, dim_out, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def _forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: tp.Optional[torch.Tensor] = None,
        cross_attn_cond: tp.Optional[torch.Tensor] = None,
        cross_attn_cond_mask: tp.Optional[torch.Tensor] = None,
        input_concat_cond: tp.Optional[torch.Tensor] = None,
        global_embed: tp.Optional[torch.Tensor] = None,
        prepend_cond: tp.Optional[torch.Tensor] = None,
        prepend_cond_mask: tp.Optional[torch.Tensor] = None,
        return_info: bool = False,
        quality_scores: tp.Optional[torch.Tensor] = None,
        quality_null_mask: tp.Optional[torch.Tensor] = None,
        return_quality_aux: bool = False,
        quality_score_targets: tp.Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if exists(cross_attn_cond):
            if self.to_cross_cond_embed is not None:
                cross_attn_cond = self.to_cross_cond_embed(cross_attn_cond)
            else:
                raise ValueError(
                    "Cross-attention conditioning is not enabled. Set enable_crossatten=True in MGELDM config."
                )

        if exists(global_embed):
            global_embed = self.to_global_embed(global_embed)
        else:
            raise NotImplementedError("Global conditioning (CLAP) should be provided in MGE-LDM")

        quality_tokens = None
        quality_prepend = None
        quality_mask = None
        if self.enable_crossatten:
            null_quality_prepend = self.quality_null_token.to(
                dtype=x.dtype,
                device=x.device,
            ).expand(x.shape[0], -1, -1)
            if quality_scores is not None:
                quality_prepend = self.quality_token_generator(quality_scores)
                if quality_score_targets is None:
                    quality_score_targets = quality_scores
            else:
                quality_prepend = null_quality_prepend

            if quality_null_mask is not None:
                quality_prepend = torch.where(
                    quality_null_mask[:, None, None],
                    null_quality_prepend,
                    quality_prepend,
                )

            if return_quality_aux and self.training and quality_scores is not None:
                quality_mask = torch.rand(
                    quality_scores.shape[0], device=quality_scores.device
                ) < self.quality_mask_prob
                if quality_null_mask is not None:
                    quality_mask = quality_mask & (~quality_null_mask)
                if quality_mask.any():
                    expanded_mask_token = self.quality_mask_token.to(
                        dtype=quality_prepend.dtype,
                        device=quality_prepend.device,
                    ).expand(quality_prepend.shape[0], -1, -1)
                    quality_prepend = torch.where(
                        quality_mask[:, None, None],
                        expanded_mask_token,
                        quality_prepend,
                    )

        if exists(prepend_cond):
            raise NotImplementedError("Prepend condition is not supported in MGE-LDM yet.")

        if exists(input_concat_cond):
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2],), mode="nearest")
            x = torch.cat([x, input_concat_cond], dim=1)

        if t.dim() == 1:
            if self.use_t_emb_trackwise:
                t = t.unsqueeze(1).repeat(1, self.num_tracks)
                timestep_feat = self.timestep_features(t[:, :, None])
                timestep_embed = self.to_timestep_embed(timestep_feat)
            else:
                timestep_feat = self.timestep_features(t[:, None])
                timestep_embed = self.to_timestep_embed(timestep_feat)
        elif t.dim() == 2:
            assert self.use_t_emb_trackwise is True
            timestep_feat = self.timestep_features(t[:, :, None])
            timestep_embed = self.to_timestep_embed(timestep_feat)
        else:
            raise ValueError(f"Unsupported timestep shape: {t.shape}")

        prepend_chunks = [timestep_embed.unsqueeze(1)]
        if quality_prepend is not None:
            prepend_chunks.append(quality_prepend)
        prepend_inputs = torch.cat(prepend_chunks, dim=1)
        prepend_length = prepend_inputs.shape[1]
        prepend_mask = torch.ones((x.shape[0], prepend_length), device=x.device, dtype=torch.bool)

        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        info = None
        output = self.transformer(
            x,
            mask=mask,
            prepend_embeds=prepend_inputs,
            prepend_mask=prepend_mask,
            global_cond=global_embed,
            return_info=return_info,
            return_final_hidden=return_quality_aux,
            context=None,
            context_mask=None,
            quality_tokens=quality_tokens,
            **kwargs,
        )
        final_hidden = None
        if return_info and return_quality_aux:
            output, info, final_hidden = output
        elif return_info:
            output, info = output
        elif return_quality_aux:
            output, final_hidden = output

        quality_aux = None
        if return_quality_aux and quality_prepend is not None and quality_score_targets is not None and final_hidden is not None:
            quality_token_start = 1  # prepend order: timestep -> quality -> audio
            quality_token_end = quality_token_start + quality_prepend.shape[1]
            quality_hidden = final_hidden[:, quality_token_start:quality_token_end, :].mean(dim=1)
            quality_score_pred = torch.sigmoid(self.quality_score_predictor(quality_hidden)).squeeze(-1)
            if quality_mask is None:
                quality_mask = torch.zeros_like(quality_score_pred, dtype=torch.bool)
            quality_aux = {
                "quality_score_pred": quality_score_pred,
                "quality_score_target": quality_score_targets.to(dtype=quality_score_pred.dtype),
                "quality_score_mask": quality_mask,
            }

        output = rearrange(output, "b t c -> b c t")[:, :, prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if return_info and return_quality_aux:
            return output, info, quality_aux
        if return_info:
            return output, info
        if return_quality_aux:
            return output, quality_aux
        return output

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        causal=False,
        scale_phi=0.0,
        mask=None,
        return_info=False,
        quality_scores=None,
        quality_null_mask=None,
        return_quality_aux: bool = False,
        **kwargs,
    ):
        if cross_attn_cond is not None and self.to_cross_cond_embed is None:
            raise ValueError(
                f"Cross-attention conditioning is not enabled but got cross_attn_cond with shape {cross_attn_cond.shape}. "
                "Set enable_crossatten=True in MGELDM config."
            )

        assert negative_cross_attn_cond is None
        assert negative_cross_attn_mask is None
        assert input_concat_cond is None
        assert prepend_cond is None
        assert prepend_cond_mask is None
        assert causal is False
        assert mask is None
        assert exists(global_embed)

        train_quality_null_mask = quality_null_mask
        if self.training and cfg_dropout_prob > 0.0:
            if exists(prepend_cond):
                raise NotImplementedError

            if exists(global_embed):
                null_embed = torch.zeros_like(global_embed, device=global_embed.device)
                batch_size, num_tracks, _ = global_embed.shape
                dropout_mask = torch.bernoulli(
                    torch.full((batch_size, num_tracks, 1), cfg_dropout_prob, device=global_embed.device)
                ).to(torch.bool)
                global_embed = torch.where(dropout_mask, null_embed, global_embed)

            if exists(cross_attn_cond):
                null_cross_attn_cond = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                batch_size = cross_attn_cond.shape[0]
                dropout_mask = torch.bernoulli(
                    torch.full((batch_size, 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)
                ).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_cross_attn_cond, cross_attn_cond)

            if exists(quality_scores) and self.enable_crossatten and not return_quality_aux:
                dropout_mask = torch.bernoulli(
                    torch.full(quality_scores.shape, cfg_dropout_prob, device=quality_scores.device)
                ).to(torch.bool)
                train_quality_null_mask = (
                    dropout_mask if train_quality_null_mask is None
                    else (train_quality_null_mask | dropout_mask)
                )

        if cfg_scale != 1.0 and exists(global_embed):
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)

            if exists(input_concat_cond):
                raise NotImplementedError
            batch_input_concat_cond = None

            if exists(cross_attn_cond):
                null_cross_attn_cond = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                if exists(negative_cross_attn_cond):
                    if exists(negative_cross_attn_mask):
                        negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)
                        negative_cross_attn_cond = torch.where(
                            negative_cross_attn_mask, negative_cross_attn_cond, null_cross_attn_cond
                        )
                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)
                else:
                    batch_cond = torch.cat([cross_attn_cond, null_cross_attn_cond], dim=0)

                if exists(cross_attn_cond_mask):
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
                else:
                    batch_cond_masks = None
            else:
                batch_cond = None
                batch_cond_masks = None

            if exists(prepend_cond):
                raise NotImplementedError
            batch_prepend_cond = None
            batch_prepend_cond_mask = None

            null_embed = torch.zeros_like(global_embed, device=global_embed.device)
            batch_global_cond = torch.cat([global_embed, null_embed], dim=0)
            batch_masks = torch.cat([mask, mask], dim=0) if exists(mask) else None

            batch_quality_scores = None
            batch_quality_null_mask = None
            if self.enable_crossatten:
                if quality_scores is not None:
                    batch_quality_scores = torch.cat([quality_scores, quality_scores], dim=0)
                    unconditional_null_mask = torch.cat(
                        [
                            torch.zeros_like(quality_scores, dtype=torch.bool),
                            torch.ones_like(quality_scores, dtype=torch.bool),
                        ],
                        dim=0,
                    )
                    if train_quality_null_mask is not None:
                        batch_quality_null_mask = torch.cat(
                            [train_quality_null_mask, unconditional_null_mask],
                            dim=0,
                        )
                    else:
                        batch_quality_null_mask = unconditional_null_mask
                else:
                    batch_quality_null_mask = torch.ones(
                        batch_inputs.shape[0],
                        device=batch_inputs.device,
                        dtype=torch.bool,
                    )

            batch_output = self._forward(
                batch_inputs,
                batch_timestep,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_masks,
                input_concat_cond=batch_input_concat_cond,
                global_embed=batch_global_cond,
                prepend_cond=batch_prepend_cond,
                prepend_cond_mask=batch_prepend_cond_mask,
                mask=batch_masks,
                return_info=return_info,
                quality_scores=batch_quality_scores,
                quality_null_mask=batch_quality_null_mask,
                return_quality_aux=return_quality_aux,
                quality_score_targets=batch_quality_scores,
                **kwargs,
            )

            info = None
            quality_aux = None
            if return_info and return_quality_aux:
                batch_output, info, quality_aux = batch_output
            elif return_info:
                batch_output, info = batch_output
            elif return_quality_aux:
                batch_output, quality_aux = batch_output

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
            cfg_output = uncond_output + cfg_scale * (cond_output - uncond_output)

            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                output = scale_phi * (cfg_output * (cond_out_std / out_cfg_std)) + (1 - scale_phi) * cfg_output
            else:
                output = cfg_output

            if return_info and return_quality_aux:
                return output, info, quality_aux
            if return_info:
                return output, info
            if return_quality_aux:
                return output, quality_aux
            return output

        return self._forward(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_cond_mask,
            input_concat_cond=input_concat_cond,
            global_embed=global_embed,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            mask=mask,
            return_info=return_info,
            quality_scores=quality_scores,
            quality_null_mask=train_quality_null_mask,
            return_quality_aux=return_quality_aux,
            quality_score_targets=quality_scores,
            **kwargs,
        )
