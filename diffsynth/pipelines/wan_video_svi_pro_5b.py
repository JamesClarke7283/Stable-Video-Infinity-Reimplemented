# SPDX-License-Identifier: Apache-2.0
# Derived from DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio and
# https://github.com/vita-epfl/Stable-Video-Infinity, branch svi_wan22),
# Copyright ModelScope Team, licensed under the Apache License 2.0.
# SVI-Pro conditioning design from Stable-Video-Infinity (https://github.com/vita-epfl/Stable-Video-Infinity),
# Copyright VITA @ EPFL, licensed under the Apache License 2.0.
# Modifications for Wan2.2-TI2V-5B SVI-Pro (fused N-frame clean-hold conditioning,
# SVI-Pro clip chaining for the single-DiT 5B model) by the
# Stable-Video-Infinity-Reimplemented authors.
#
# What this module does:
#   Wan2.2-TI2V-5B has no conditioning channels (in_dim == z_dim == 48), so the
#   SVI 2.0 Pro trick for the 14B model (`y = concat([anchor_latent, motion_latent,
#   padding])` in extra input channels) cannot be used. Instead we generalize the
#   5B's native first-frame hold to an N-frame clean hold:
#     cond_latents = concat([anchor_latent?, motion_latent?], dim=2)  (N in {0,1,2})
#   clamped into `latents[:, :, 0:N]` after every sampling step, with a per-frame
#   timestep map marking those N frames as t=0.
#     - I2V first clip: cond = [anchor]           (N=1, == native I2V)
#     - I2V later clips: cond = [anchor, motion]  (N=2)
#     - T2V first clip: cond = []                 (N=0, == native T2V)
#     - T2V later clips: cond = [motion] or [generated_anchor, motion]
import torch
from PIL import Image
from einops import rearrange
from tqdm import tqdm
from typing import Optional, Union

from ..core import ModelConfig
from ..diffusion.base_pipeline import PipelineUnit
from ..diffusion import FlowMatchScheduler
from ..models.wan_video_dit import sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_dit import WanModel
from .wan_video import (
    WanVideoPipeline,
    WanVideoUnit_ImageEmbedderFused,
    WanVideoUnit_InputVideoEmbedder,
)


def model_fn_wan_video_svi(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    fuse_vae_embedding_in_latents: bool = False,
    num_cond_frames: int = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    """Fork of diffsynth.pipelines.wan_video.model_fn_wan_video, stripped to the
    Wan2.2-TI2V-5B path (no VACE/VAP/Animate/S2V/LongCat/TeaCache/USP), with the
    per-frame timestep map generalized from exactly one t=0 frame to
    `num_cond_frames` t=0 frames (0, 1 or 2). All other frames get `timestep`.
    """
    tokens_per_frame = latents.shape[3] * latents.shape[4] // 4
    total_frames = latents.shape[2]
    if num_cond_frames is None:
        # Legacy behavior of the stock model_fn: exactly one clean frame.
        num_cond_frames = 1 if fuse_vae_embedding_in_latents else 0

    if dit.seperated_timestep and num_cond_frames > 0:
        timestep = torch.concat([
            torch.zeros((num_cond_frames, tokens_per_frame), dtype=latents.dtype, device=latents.device),
            torch.ones((total_frames - num_cond_frames, tokens_per_frame), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)

    x = dit.patchify(x, None)
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block in dit.blocks:
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x


class WanVideoUnit_ImageEmbedderFusedSVI(PipelineUnit):
    """SVI-Pro conditioning builder for Wan2.2-TI2V-5B (inference).

    Builds `cond_latents = concat([anchor_latent?, motion_latent?], dim=2)` and
    clamps it into the initial noise; the pipeline's __call__ re-clamps it after
    every sampling step. Sources:
      - anchor: PIL `anchor` (defaults to `input_image`, the I2V task) or a raw
        `anchor_latent` tensor (e.g. the generated anchor for T2V chains).
      - motion: last `num_motion_latent` latents of the previous clip
        (`prev_last_latent`), pure latent hand-off, never decode/re-encode.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "anchor", "anchor_latent", "prev_last_latent",
                          "num_motion_latent", "latents", "height", "width",
                          "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "cond_latents", "num_cond_frames"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: "WanVideoSviPro5BPipeline", input_image, anchor, anchor_latent,
                prev_last_latent, num_motion_latent, latents, height, width,
                tiled, tile_size, tile_stride):
        if anchor is None and input_image is not None:
            anchor = input_image
        parts = []
        if anchor_latent is not None:
            parts.append(anchor_latent.to(dtype=pipe.torch_dtype, device=pipe.device))
        elif anchor is not None:
            pipe.load_models_to_device(self.onload_model_names)
            image = pipe.preprocess_image(anchor.resize((width, height))).transpose(0, 1)
            anchor_latent = pipe.vae.encode([image], device=pipe.device, tiled=tiled,
                                            tile_size=tile_size, tile_stride=tile_stride)
            parts.append(anchor_latent.to(dtype=pipe.torch_dtype, device=pipe.device))
        if prev_last_latent is not None:
            motion_latent = prev_last_latent.to(dtype=pipe.torch_dtype, device=pipe.device)[:, :, -num_motion_latent:]
            parts.append(motion_latent)
        if len(parts) == 0:
            return {}
        cond_latents = torch.concat(parts, dim=2)
        num_cond_frames = cond_latents.shape[2]
        latents[:, :, :num_cond_frames] = cond_latents
        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "cond_latents": cond_latents,
            "num_cond_frames": num_cond_frames,
        }


class WanVideoUnit_SVICondVAE(PipelineUnit):
    """Training-time VAE encoder for SVI conditioning (prev clip + anchor image).

    Emits the conditioning latents *separately* (`svi_anchor_latent`,
    `svi_motion_latent`) so the SVI error-recycling loss can sample the
    conditioning mode (I2V/T2V, first/later clip) and inject errors per part.
    """
    def __init__(self):
        super().__init__(
            input_params=("svi_prev_video", "svi_anchor_image", "num_motion_latent",
                          "tiled", "tile_size", "tile_stride"),
            output_params=("svi_motion_latent", "svi_anchor_latent"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: "WanVideoSviPro5BPipeline", svi_prev_video, svi_anchor_image,
                num_motion_latent, tiled, tile_size, tile_stride):
        outputs = {}
        if svi_prev_video is not None and len(svi_prev_video) > 0:
            pipe.load_models_to_device(self.onload_model_names)
            prev_video = pipe.preprocess_video(svi_prev_video)
            prev_latents = pipe.vae.encode(prev_video, device=pipe.device, tiled=tiled,
                                           tile_size=tile_size, tile_stride=tile_stride)
            outputs["svi_motion_latent"] = prev_latents.to(dtype=pipe.torch_dtype, device=pipe.device)[:, :, -num_motion_latent:]
        if svi_anchor_image is not None:
            pipe.load_models_to_device(self.onload_model_names)
            image = pipe.preprocess_image(svi_anchor_image).transpose(0, 1)
            anchor_latent = pipe.vae.encode([image], device=pipe.device, tiled=tiled,
                                            tile_size=tile_size, tile_stride=tile_stride)
            outputs["svi_anchor_latent"] = anchor_latent.to(dtype=pipe.torch_dtype, device=pipe.device)
        return outputs


class WanVideoSviPro5BPipeline(WanVideoPipeline):
    """Wan2.2-TI2V-5B pipeline with SVI-Pro fused conditioning and clip chaining."""

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        # Swap the stock fused embedder (single first-frame hold) for the SVI one
        # (N-frame clean hold), and add the training-time cond encoder right after
        # the input-video embedder.
        units = []
        for unit in self.units:
            if isinstance(unit, WanVideoUnit_ImageEmbedderFused):
                units.append(WanVideoUnit_ImageEmbedderFusedSVI())
            else:
                units.append(unit)
            if isinstance(unit, WanVideoUnit_InputVideoEmbedder):
                units.append(WanVideoUnit_SVICondVAE())
        self.units = units
        self.model_fn = model_fn_wan_video_svi

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        vram_limit: float = None,
    ):
        # Redirect model path (same convention as WanVideoPipeline.from_pretrained)
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]

        pipe = WanVideoSviPro5BPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        pipe.dit = model_pool.fetch_model("wan_video_dit")
        pipe.vae = model_pool.fetch_model("wan_video_vae")

        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # SVI-Pro conditioning
        anchor: Optional[Image.Image] = None,
        anchor_latent: Optional[torch.Tensor] = None,
        prev_last_latent: Optional[torch.Tensor] = None,
        num_motion_latent: Optional[int] = 1,
        # Video-to-video (unused by SVI chaining, kept for parity)
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 704,
        width: Optional[int] = 1280,
        num_frames: Optional[int] = 121,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Gradient checkpointing (training)
        use_gradient_checkpointing: Optional[bool] = False,
        use_gradient_checkpointing_offload: Optional[bool] = False,
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Inputs
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "input_image": input_image,
            "anchor": anchor,
            "anchor_latent": anchor_latent,
            "prev_last_latent": prev_last_latent,
            "num_motion_latent": num_motion_latent,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": use_gradient_checkpointing_offload,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(["dit"])
        models = {"dit": self.dit}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            # SVI-Pro clean-hold clamp: re-impose the conditioning latents.
            if inputs_shared.get("cond_latents") is not None:
                num_cond_frames = inputs_shared["cond_latents"].shape[2]
                inputs_shared["latents"][:, :, :num_cond_frames] = inputs_shared["cond_latents"]

        # Keep the final latents for chaining BEFORE decode.
        prev_last_latent = inputs_shared["latents"].detach().clone()

        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return {"video": video, "prev_last_latent": prev_last_latent}
