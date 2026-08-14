# SPDX-License-Identifier: Apache-2.0
# Derived from DiffSynth-Studio examples/wanvideo/model_training/train.py
# (https://github.com/modelscope/DiffSynth-Studio; SVI branch:
# https://github.com/vita-epfl/Stable-Video-Infinity, branch svi_wan22),
# Copyright ModelScope Team, licensed under the Apache License 2.0.
# SVI Error-Recycling Fine-Tuning training entry for Wan2.2-TI2V-5B by the
# Stable-Video-Infinity-Reimplemented authors.
import os, argparse, random, warnings, time, re
import torch
import imageio
import accelerate
from PIL import Image

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import DataProcessingOperator, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video_svi_pro_5b import WanVideoSviPro5BPipeline, ModelConfig
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.diffusion.error_replay import ErrorReplayBank
from diffsynth.diffusion.loss import SVIConfig, SVIErrorRecyclingLoss
from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.parsers import add_general_config, add_video_size_config
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Silence the per-tile "VAE encoding/decoding" tqdm bars inside the training
# loop - the outer svi-train bar is the only progress display we want.
import diffsynth.models.wan_video_vae as _wan_vae
_wan_vae.tqdm = lambda iterable=None, *args, **kwargs: iterable


def save_training_state(state_path, model_logger, optimizer, scheduler, raw_model,
                        epoch_id, ema, last_val, best_val, bad_val_events):
    """Full training-state snapshot for exact resume: optimizer/scheduler state,
    step/epoch counters, EMA + early-stopping state, error replay banks, and RNG
    states. Written atomically (tmp + rename) next to the step-*.safetensors LoRA
    checkpoint saved at the same step."""
    state = {
        "iteration": raw_model.iteration,
        "epoch_id": epoch_id,
        "num_steps": model_logger.num_steps,
        "ema": ema,
        "last_val": last_val,
        "best_val": best_val,
        "bad_val_events": bad_val_events,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "error_banks": raw_model.error_bank.banks,
        "rng": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp_path = state_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, state_path)


def launch_svi_training_task(accelerator, dataset, model, model_logger, args, val_loader=None):
    """Training loop with a global tqdm bar (loss + EMA + error-bank occupancy)
    and grad clipping 1.0 (paper Table 9). Forked from
    diffsynth.diffusion.runner.launch_training_task.

    If val_loader is given, runs a deterministic clean-input validation every
    args.val_every steps. All metrics are appended per step to
    <output_path>/metrics.csv (loss, ema, grad_norm, lr, sec/step, bank
    occupancy, peak GPU memory, and val_loss/best_val/is_best on val steps) —
    one file, ready for plotting. Saves best-val.safetensors whenever val
    improves; stops early if val doesn't improve for args.early_stop_patience
    steps (0 disables), and prints the best checkpoint at the end of training.

    Every args.save_steps, alongside the step-N LoRA safetensors, a full
    training-state snapshot (optimizer, scheduler, counters, EMA, early-stopping
    state, error banks, RNG) is written to training_state.pt. --resume_auto
    performs an exact full-state resume when the snapshot matches the newest
    checkpoint, otherwise falls back to weights-only resume."""
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=args.dataset_num_workers)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    raw_model = model.module if hasattr(model, "module") else model

    os.makedirs(args.output_path, exist_ok=True)
    metrics_path = os.path.join(args.output_path, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("step,epoch,unix_time,loss,ema,grad_norm,lr,sec_per_step,"
                    "bank_vid,bank_noi,gpu_max_mem_gb,val_loss,best_val,is_best\n")

    # ---- Resume: full-state if the snapshot matches the checkpoint, else
    # weights-only (LoRA weights were already loaded via args.lora_checkpoint).
    state_path = os.path.join(args.output_path, "training_state.pt")
    start_step, start_epoch, skip_batches = 0, 0, 0
    ema, last_val = None, None
    best_val, bad_val_events, stop_early = None, 0, False
    if args.resume_auto and os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        ckpt_step = None
        if args.lora_checkpoint is not None:
            m = re.match(r"step-(\d+)\.safetensors$", os.path.basename(args.lora_checkpoint))
            ckpt_step = int(m.group(1)) if m else None
        if ckpt_step is not None and state["iteration"] == ckpt_step:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            raw_model.iteration = state["iteration"]
            raw_model.error_bank.banks = state["error_banks"]
            model_logger.num_steps = state["num_steps"]
            ema, last_val = state["ema"], state["last_val"]
            best_val, bad_val_events = state["best_val"], state["bad_val_events"]
            rng = state["rng"]
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch_cpu"])
            if rng["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            steps_per_epoch = len(dataloader)
            start_step = raw_model.iteration
            start_epoch, skip_batches = divmod(start_step, steps_per_epoch)
            occ = raw_model.error_bank.occupancy()
            print(f"[svi] FULL-STATE RESUME at step {start_step} "
                  f"(epoch {start_epoch}, skipping {skip_batches} batches): optimizer/scheduler/RNG "
                  f"restored, banks reloaded (vid {occ['vid']['total']} + noi {occ['noi']['total']})", flush=True)
        else:
            print(f"[svi] training_state.pt (iteration={state['iteration']}) does not match the resume "
                  f"checkpoint (step={ckpt_step}) - falling back to weights-only resume.", flush=True)
    postfix = {}
    total_steps = len(dataloader) * args.num_epochs
    with tqdm(total=total_steps, initial=start_step, desc="svi-train", dynamic_ncols=True) as pbar:
        for epoch_id in range(start_epoch, args.num_epochs):
            active_loader = dataloader
            if epoch_id == start_epoch and skip_batches > 0:
                active_loader = accelerator.skip_first_batches(dataloader, skip_batches)
            for data in active_loader:
                t0 = time.time()
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    loss = model(data)
                    accelerator.backward(loss)
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    model_logger.on_step_end(accelerator, model, args.save_steps)
                    if args.save_steps is not None and model_logger.num_steps % args.save_steps == 0:
                        save_training_state(state_path, model_logger, optimizer, scheduler, raw_model,
                                            epoch_id, ema, last_val, best_val, bad_val_events)
                    scheduler.step()
                v = loss.detach().float().item()
                ema = v if ema is None else 0.98 * ema + 0.02 * v
                step = raw_model.iteration
                val_field, best_field, is_best_field = "", "", ""
                if val_loader is not None and step % args.val_every == 0:
                    def _val_hook(i, n, m, _base=dict(postfix)):
                        _base["val_run"] = f"{i}/{n} {m:.4f}"
                        pbar.set_postfix(_base)
                    last_val = raw_model.validate(val_loader, args.val_batches, progress_hook=_val_hook)
                    is_best = best_val is None or last_val < best_val - args.early_stop_min_delta
                    if is_best:
                        best_val, bad_val_events = last_val, 0
                        model_logger.save_model(accelerator, model, "best-val.safetensors")
                        print(f"[svi] new best val {best_val:.5f} -> saved best-val.safetensors", flush=True)
                    else:
                        bad_val_events += 1
                        if args.early_stop_patience > 0 and bad_val_events * args.val_every >= args.early_stop_patience:
                            print(f"[svi] EARLY STOP at iter={step}: no val improvement for "
                                  f"{bad_val_events * args.val_every} steps (best val {best_val:.5f})", flush=True)
                            stop_early = True
                    val_field, best_field, is_best_field = f"{last_val:.6f}", f"{best_val:.6f}", str(int(is_best))
                    print(f"[svi] VALIDATION iter={step} val_loss={last_val:.5f} train_ema={ema:.5f}", flush=True)
                occ = raw_model.error_bank.occupancy()
                gpu_gb = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                with open(metrics_path, "a") as f:
                    f.write(f"{step},{epoch_id},{int(time.time())},{v:.6f},{ema:.6f},{float(grad_norm):.4f},"
                            f"{optimizer.param_groups[0]['lr']:.2e},{time.time() - t0:.2f},"
                            f"{occ['vid']['total']},{occ['noi']['total']},{gpu_gb:.2f},"
                            f"{val_field},{best_field},{is_best_field}\n")
                postfix = {"loss": f"{v:.4f}", "ema": f"{ema:.4f}",
                           "bank_vid": occ["vid"]["total"], "bank_noi": occ["noi"]["total"],
                           "epoch": epoch_id}
                if last_val is not None:
                    postfix["val"] = f"{last_val:.4f}"
                if best_val is not None:
                    postfix["best_val"] = f"{best_val:.4f}"
                pbar.set_postfix(postfix)
                pbar.update(1)
                if stop_early:
                    break
            if stop_early:
                break
            if args.save_steps is None:
                model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, args.save_steps)
    if accelerator.is_main_process:
        if best_val is not None:
            print(f"[svi] TRAINING DONE at iter={raw_model.iteration}. "
                  f"Best checkpoint: {os.path.join(args.output_path, 'best-val.safetensors')} "
                  f"(val_loss {best_val:.5f})", flush=True)
        else:
            print(f"[svi] TRAINING DONE at iter={raw_model.iteration}. "
                  f"Validation was disabled - no best checkpoint; use the step-*.safetensors "
                  f"checkpoints in {args.output_path}", flush=True)


class LoadSVIVideoPair(DataProcessingOperator):
    """Load a (previous-clip, window) pair of consecutive `num_frames` blocks from
    one video, plus an anchor frame, for SVI chained-clip training.

    Returns {"video": window_frames, "prev_video": prev_frames_or_[],
             "anchor_image": PIL, "first_clip": bool}.
    Videos are expected pre-normalized to 24fps by scripts/prepare_dataset.py.
    """
    def __init__(self, num_frames=121, frame_processor=lambda x: x, p_anchor_random=0.5):
        self.num_frames = num_frames
        self.frame_processor = frame_processor
        self.p_anchor_random = p_anchor_random

    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        total = int(reader.count_frames())
        num_frames = self.num_frames
        max_start = total - num_frames
        if max_start <= 0:
            start = 0
        else:
            start = random.randint(0, max_start)
        has_prev = start >= num_frames

        def read_frame(i):
            return self.frame_processor(Image.fromarray(reader.get_data(i)))

        window_frames = [read_frame(i) for i in range(start, min(start + num_frames, total))]
        if has_prev:
            prev_frames = [read_frame(i) for i in range(start - num_frames, start)]
        else:
            prev_frames = []

        # Anchor: first frame of the video by default (the "user first frame"
        # proxy); with probability p_anchor_random, a random frame from the
        # previous region (SVI 2.0 "strong first-frame augmentation").
        if random.random() < self.p_anchor_random and start > 0:
            anchor_idx = random.randint(max(0, start - num_frames), start - 1)
        else:
            anchor_idx = 0
        anchor_image = read_frame(anchor_idx)

        reader.close()
        return {
            "video": window_frames,
            "prev_video": prev_frames,
            "anchor_image": anchor_image,
            "first_clip": not has_prev,
        }


class WanSVITrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        device="cpu",
        task="svi",
        svi_config: SVIConfig = None,
        num_motion_latent=1,
        svi_max_errors_per_grid=200,
        svi_spatial_pool=2,
        vae_tiled=False,
        fp8_models=None,
        offload_models=None,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is forcibly enabled by the training framework.")
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        self.pipe = WanVideoSviPro5BPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=model_configs, tokenizer_config=tokenizer_config,
            audio_processor_config=None, redirect_common_files=False,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, None, lora_base_model)

        # Training mode: 1000-step shifted training schedule + freeze + LoRA
        self.switch_pipe_to_training_mode(
            self.pipe, None,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # SVI error replay memory: 50 grids aligned to the 50-step inference
        # schedule (shift=5, the inference-time sigma_shift).
        anchor_sched = FlowMatchScheduler("Wan")
        anchor_sched.set_timesteps(50, shift=5.0)
        self.error_bank = ErrorReplayBank(
            anchor_sched.timesteps,
            max_errors_per_grid=svi_max_errors_per_grid,
            spatial_pool=svi_spatial_pool,
        )
        # Validation state (lazy): dummy bank (capacity 1) so val forwards never
        # pollute the training replay memory, and a clean-input config.
        self._bank_anchor_timesteps = anchor_sched.timesteps
        self._bank_spatial_pool = svi_spatial_pool
        self._val_bank = None
        self._val_config = None
        self.svi_config = svi_config if svi_config is not None else SVIConfig()
        self.iteration = 0
        self.num_motion_latent = num_motion_latent
        self.vae_tiled = vae_tiled
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.task = task
        self.task_to_loss = {
            "svi": lambda pipe, inputs_shared, inputs_posi, inputs_nega: SVIErrorRecyclingLoss(
                pipe, self.error_bank, self.svi_config, self.iteration,
                **inputs_shared, **inputs_posi,
            ),
        }

    def get_pipeline_inputs(self, data):
        pack = data["video"]  # dict produced by LoadSVIVideoPair
        video = pack["video"]
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": video,
            "svi_prev_video": pack["prev_video"],
            "svi_anchor_image": pack["anchor_image"],
            "num_motion_latent": self.num_motion_latent,
            "height": video[0].size[1],
            "width": video[0].size[0],
            "num_frames": len(video),
            "cfg_scale": 1,
            "cfg_merge": False,
            "vace_scale": 1,
            "tiled": self.vae_tiled,
            "tile_size": (30, 52),
            "tile_stride": (15, 26),
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        self.iteration += 1
        if self.iteration % 20 == 0:
            print(f"[svi] iter={self.iteration} loss={loss.item():.5f} bank={self.error_bank.occupancy()}", flush=True)
        return loss

    @torch.no_grad()
    def validate(self, val_dataloader, num_batches=8, seed=1234, progress_hook=None):
        """Deterministic clean-input validation on the held-out split.

        Same fixed batches, window/anchor draws, timesteps and noise on every
        call (seeded per batch index), so the val-loss curve is comparable
        across steps. Uses p_clean=1.0 (no error injection) — it measures
        whether the LoRA preserves base-generation quality on unseen videos.
        Errors curated during val go to a throwaway bank (capacity 1), never
        into the training replay memory. Global RNG states are restored
        afterwards so the training stream is unaffected. progress_hook(i, n,
        running_mean) is called after each batch (live progress display).
        """
        if self._val_bank is None:
            self._val_bank = ErrorReplayBank(
                self._bank_anchor_timesteps, max_errors_per_grid=1,
                spatial_pool=self._bank_spatial_pool,
            )
            self._val_config = SVIConfig(
                p_clean=1.0,
                mode_probs=self.svi_config.mode_probs,
                warmup_iter=self.svi_config.warmup_iter,
                loss_mask_cond_frames=self.svi_config.loss_mask_cond_frames,
            )
        py_state = random.getstate()
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        total = min(num_batches, len(val_dataloader))
        losses = []
        try:
            for i, data in enumerate(val_dataloader):
                if i >= num_batches:
                    break
                random.seed(seed + i)
                torch.manual_seed(seed + i)
                inputs = self.get_pipeline_inputs(data)
                inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
                for unit in self.pipe.units:
                    inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
                inputs_shared, inputs_posi, _ = inputs
                loss = SVIErrorRecyclingLoss(
                    self.pipe, self._val_bank, self._val_config, self.iteration,
                    rng_seed=seed + i, **inputs_shared, **inputs_posi,
                )
                losses.append(loss.detach().float().item())
                if progress_hook is not None:
                    progress_hook(i + 1, total, sum(losses) / len(losses))
        finally:
            random.setstate(py_state)
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        return sum(losses) / max(1, len(losses))


def svi_parser():
    parser = argparse.ArgumentParser(description="SVI-Pro ERFT training for Wan2.2-TI2V-5B.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--num_motion_latent", type=int, default=1, help="Motion latents taken from the previous clip.")
    parser.add_argument("--p_anchor_random", type=float, default=0.5, help="Probability of a random (non-first) anchor frame.")
    parser.add_argument("--vae_tiled", default=False, action="store_true", help="Tiled VAE encode during training.")
    # SVI ERFT knobs (paper Table 9 / shipped SVI 1.0 defaults)
    parser.add_argument("--svi_p_vid", type=float, default=0.9)
    parser.add_argument("--svi_p_img", type=float, default=0.9)
    parser.add_argument("--svi_p_noi", type=float, default=0.01)
    parser.add_argument("--svi_p_clean", type=float, default=0.5)
    parser.add_argument("--svi_warmup_iter", type=int, default=50)
    parser.add_argument("--svi_max_errors_per_grid", type=int, default=200)
    parser.add_argument("--svi_spatial_pool", type=int, default=2)
    parser.add_argument("--svi_mode_probs", type=str, default="0.45,0.15,0.30,0.10",
                        help="Comma probs for i2v_chained,i2v_first,t2v_chained,t2v_pure.")
    parser.add_argument("--svi_loss_mask_cond_frames", type=int, default=1, choices=[0, 1])
    parser.add_argument("--resume_auto", action="store_true",
                        help="Auto-resume from the newest checkpoint in --output_path (weights only).")
    # Validation (held-out split)
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of videos held out for validation (seeded split).")
    parser.add_argument("--val_every", type=int, default=200, help="Validate every N training steps (0 disables).")
    parser.add_argument("--val_batches", type=int, default=8, help="Fixed batches per validation event.")
    parser.add_argument("--early_stop_patience", type=int, default=2000,
                        help="Stop training if val loss hasn't improved for this many steps (0 disables; needs validation on).")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                        help="Minimum val-loss improvement that counts as progress (resets patience).")
    return parser


def find_latest_checkpoint(output_path):
    """Newest step-*.safetensors / epoch-*.safetensors in output_path, by mtime."""
    import glob
    cks = glob.glob(os.path.join(output_path, "step-*.safetensors")) + \
          glob.glob(os.path.join(output_path, "epoch-*.safetensors"))
    if not cks:
        return None
    return max(cks, key=os.path.getmtime)


if __name__ == "__main__":
    parser = svi_parser()
    args = parser.parse_args()
    if args.resume_auto and args.lora_checkpoint is None:
        latest = find_latest_checkpoint(args.output_path)
        if latest is not None:
            args.lora_checkpoint = latest
            print(f"[svi] AUTO-RESUME: loading LoRA weights from {latest}", flush=True)
            print("[svi] note: optimizer state and error banks reset; banks refill during warmup iters.", flush=True)
        else:
            print(f"[svi] AUTO-RESUME: no checkpoint in {args.output_path}, starting fresh.", flush=True)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=["video"],
        main_data_operator=(
            ToAbsolutePath(args.dataset_base_path)
            >> LoadSVIVideoPair(
                num_frames=args.num_frames,
                frame_processor=ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
                p_anchor_random=args.p_anchor_random,
            )
        ),
    )
    # Deterministic 90/10 train/holdout split (fixed seed -> identical split on
    # every run, so resumed training never trains on the val set).
    val_loader = None
    if args.val_every > 0 and args.val_split > 0:
        import copy
        order = list(range(len(dataset.data)))
        random.Random(42).shuffle(order)
        n_val = max(1, int(len(order) * args.val_split))
        val_dataset = copy.copy(dataset)
        val_dataset.data = [dataset.data[i] for i in order[:n_val]]
        dataset.data = [dataset.data[i] for i in order[n_val:]]
        # workers=0: window/anchor draws run in-process under our seeds.
        val_loader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=0)
        print(f"[svi] holdout split: {len(order) - n_val} train / {n_val} val videos", flush=True)
    mode_probs = [float(x) for x in args.svi_mode_probs.split(",")]
    svi_config = SVIConfig(
        p_vid=args.svi_p_vid,
        p_img=args.svi_p_img,
        p_noi=args.svi_p_noi,
        p_clean=args.svi_p_clean,
        mode_probs=dict(zip(["i2v_chained", "i2v_first", "t2v_chained", "t2v_pure"], mode_probs)),
        warmup_iter=args.svi_warmup_iter,
        loss_mask_cond_frames=bool(args.svi_loss_mask_cond_frames),
    )
    model = WanSVITrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        task="svi",
        svi_config=svi_config,
        num_motion_latent=args.num_motion_latent,
        svi_max_errors_per_grid=args.svi_max_errors_per_grid,
        svi_spatial_pool=args.svi_spatial_pool,
        vae_tiled=args.vae_tiled,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launch_svi_training_task(accelerator, dataset, model, model_logger, args, val_loader=val_loader)
