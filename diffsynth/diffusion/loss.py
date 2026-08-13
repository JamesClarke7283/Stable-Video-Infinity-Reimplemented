from .base_pipeline import BasePipeline
import torch


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss


# ---------------------------------------------------------------------------
# SVI Error-Recycling Fine-Tuning loss (added by the
# Stable-Video-Infinity-Reimplemented authors).
# Algorithm from "Stable Video Infinity: Infinite-Length Video Generation with
# Error Recycling" (arXiv 2510.09212, Copyright VITA @ EPFL, Apache License 2.0),
# re-implemented for Wan2.2-TI2V-5B fused conditioning (see
# diffsynth/pipelines/wan_video_svi_pro_5b.py).
# ---------------------------------------------------------------------------
import random
from dataclasses import dataclass, field


@dataclass
class SVIConfig:
    """Knobs for SVI Error-Recycling Fine-Tuning (defaults from the SVI paper
    Table 9 / shipped SVI 1.0 configs)."""
    # Error-injection probabilities
    p_vid: float = 0.9            # inject video-latent error E_vid
    p_img: float = 0.9            # inject conditioning error E_img (anchor/motion)
    p_noi: float = 0.01           # inject noise error E_noi
    p_clean: float = 0.5          # probability of a fully error-free input
    # Conditioning-mode probabilities (I2V chained / I2V first clip / T2V chained / T2V pure)
    mode_probs: dict = field(default_factory=lambda: {
        "i2v_chained": 0.45, "i2v_first": 0.15, "t2v_chained": 0.30, "t2v_pure": 0.10,
    })
    # Replay memory
    warmup_iter: int = 50                    # bank every step during warmup
    clean_bank_update_prob: float = 0.1      # on clean-input steps, bank w.p. this
    # Loss
    loss_mask_cond_frames: bool = True       # only score the generated frames


def _svi_sample_mode(mode_probs, has_anchor, has_motion, rng: random.Random):
    """Sample the conditioning mode, degrading gracefully to what the available
    conditioning latents allow."""
    modes = list(mode_probs.keys())
    probs = [mode_probs[m] for m in modes]
    mode = rng.choices(modes, weights=probs, k=1)[0]
    if mode == "i2v_chained" and not (has_anchor and has_motion):
        mode = "i2v_first" if has_anchor else ("t2v_chained" if has_motion else "t2v_pure")
    elif mode == "i2v_first" and not has_anchor:
        mode = "t2v_chained" if has_motion else "t2v_pure"
    elif mode == "t2v_chained" and not has_motion:
        mode = "t2v_pure"
    return mode


def SVIErrorRecyclingLoss(pipe: BasePipeline, error_bank, svi_config: SVIConfig, iteration: int, rng_seed: int = None, **inputs):
    """SVI Error-Recycling Fine-Tuning loss (paper Eq. 3-6).

    Corrupts the clean flow-matching input with the DiT's own banked errors,
    trains the model to predict the error-recycled velocity V_rcy pointing to the
    *clean* latent, and banks freshly computed errors back into the replay memory.

    `rng_seed` (default None = OS entropy, training behavior) makes the
    mode/injection draws reproducible — used by validation for comparable
    val-loss curves.
    """
    scheduler = pipe.scheduler  # training mode, 1000 shifted timesteps
    rng = random.Random(rng_seed)
    device = pipe.device

    input_latents = inputs["input_latents"]            # (1, 48, T, h, w), clean x0
    anchor_latent = inputs.get("svi_anchor_latent")    # (1, 48, 1, h, w) or None
    motion_latent = inputs.get("svi_motion_latent")    # (1, 48, m, h, w) or None

    # ---- Conditioning mode (I2V/T2V x first/later clip) ---------------------
    mode = _svi_sample_mode(
        svi_config.mode_probs,
        has_anchor=anchor_latent is not None,
        has_motion=motion_latent is not None,
        rng=rng,
    )
    cond_parts = []
    if mode.startswith("i2v"):
        cond_parts.append(anchor_latent)
    if mode.endswith("chained"):
        cond_parts.append(motion_latent)
    num_cond_frames = sum(p.shape[2] for p in cond_parts)
    cond_latents = torch.cat(cond_parts, dim=2) if num_cond_frames > 0 else None

    # ---- Timestep -----------------------------------------------------------
    timestep_id = torch.randint(0, len(scheduler.timesteps), (1,))
    timestep = scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=device)
    sigma = scheduler.sigmas[timestep_id].to(device=device)

    # ---- Error injection gates (paper Eq. 3) --------------------------------
    inject_vid = rng.random() < svi_config.p_vid
    inject_img = rng.random() < svi_config.p_img and num_cond_frames > 0
    inject_noi = rng.random() < svi_config.p_noi
    clean_step = rng.random() < svi_config.p_clean
    if clean_step:
        inject_vid = inject_img = inject_noi = False

    # ---- Build the corrupted input ------------------------------------------
    x0 = input_latents
    noise = torch.randn_like(x0)

    x0_corrupt = x0.clone()
    if inject_vid:
        e_vid = error_bank.sample("vid", timestep, mode="current", device=device, dtype=x0.dtype)
        if e_vid is not None:
            x0_corrupt[:, :, num_cond_frames:] = x0_corrupt[:, :, num_cond_frames:] + e_vid[:, :, num_cond_frames:]

    noise_corrupt = noise
    if inject_noi:
        e_noi = error_bank.sample("noi", timestep, mode="current", device=device, dtype=x0.dtype)
        if e_noi is not None:
            noise_corrupt = noise + e_noi

    # x_tilde = (1-sigma) * x0_corrupt + sigma * noise_corrupt
    x_tilde = scheduler.add_noise(x0_corrupt, noise_corrupt, timestep)
    if num_cond_frames > 0:
        cond_in = cond_latents
        if inject_img:
            e_img = error_bank.sample_cond_error(num_cond_frames, timestep, device=device, dtype=x0.dtype)
            if e_img is not None:
                cond_in = cond_latents + e_img
        # Conditioning frames are held clean (t=0): place them un-noised.
        x_tilde[:, :, :num_cond_frames] = cond_in

    # ---- Error-recycled target (paper Eq. 6): point to the CLEAN latent -----
    training_target = noise_corrupt - x0

    # ---- Forward ------------------------------------------------------------
    v_pred = pipe.model_fn(
        dit=pipe.dit,
        latents=x_tilde,
        timestep=timestep,
        context=inputs["context"],
        fuse_vae_embedding_in_latents=num_cond_frames > 0,
        num_cond_frames=num_cond_frames,
        use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
        use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
    )

    if svi_config.loss_mask_cond_frames and num_cond_frames > 0:
        loss = torch.nn.functional.mse_loss(
            v_pred[:, :, num_cond_frames:].float(), training_target[:, :, num_cond_frames:].float()
        )
    else:
        loss = torch.nn.functional.mse_loss(v_pred.float(), training_target.float())
    loss = loss * scheduler.training_weight(timestep)

    # ---- Bidirectional one-step error curation (paper Eq. 4, no grad) -------
    # E_clean = sigma * (V_rcy - v_hat)  (clean-endpoint residual -> bank "vid")
    # E_noise = (1-sigma) * (v_hat - V_rcy) (noise-endpoint residual -> bank "noi")
    bank_this_step = (iteration < svi_config.warmup_iter) or (not clean_step) or (
        rng.random() < svi_config.clean_bank_update_prob
    )
    if bank_this_step:
        with torch.no_grad():
            e_clean = (sigma * (training_target - v_pred.detach())).clone()
            e_noise = ((1 - sigma) * (v_pred.detach() - training_target)).clone()
            # Conditioning frames are clamped; their "errors" are meaningless.
            if num_cond_frames > 0:
                e_clean[:, :, :num_cond_frames] = 0
                e_noise[:, :, :num_cond_frames] = 0
            error_bank.update("vid", timestep, e_clean)
            error_bank.update("noi", timestep, e_noise)

    return loss
