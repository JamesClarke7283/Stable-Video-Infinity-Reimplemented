# PLAN — SVI-Pro LoRA for Wan2.2-TI2V-5B

## 1. Objective

Train an **SVI-Pro-style error-recycling LoRA** for **Wan2.2-TI2V-5B**, the model
missing from the Stable-Video-Infinity family, so that the 5B model can generate
infinite-length videos via clip chaining with:

- an **anchor latent** (user first frame, shared across all clips) for identity consistency,
- a **motion latent** (last latent of the previous clip, pure latent hand-off — never
  decode/re-encode) for cross-clip continuity,
- **Error-Recycling Fine-Tuning (ERFT)** so the DiT actively corrects its own
  accumulated errors instead of drifting.

Everything runs at the 5B model's **maximum specification: 720P (1280×704 landscape /
704×1248 portrait), 121 frames per clip (~5s), 24fps** — training and inference alike.

We implement **SVI 2.0 Pro** semantics only (no non-Pro variant), on the **svi_wan22** codebase
(DiffSynth-Studio 2.0), learning the training algorithm from SVI 1.0 (`main` branch)
and the paper (`SVI-Pro-Technical-Report.pdf`, arXiv 2510.09212).

## 2. Sources of truth (analyzed)

| Source | What it gives us |
|---|---|
| Paper (PDF in repo root) | ERFT math: error injection (Eq. 3), bidirectional one-step error curation (Eq. 4), replay memory with 50 timestep grids, Z=500, selective resampling (Eq. 5), loss (Eq. 6), hyperparams (Table 9), data (Appx. D) |
| `/tmp/Stable-Video-Infinity` branch `main` (SVI 1.0, Wan2.1) | Reference ERFT implementation: `train_svi.py` (banks, injection, self-corrected target), `scripts/train/svi_*.sh` (shipped probs: p_vid=0.9, p_img=0.9, p_noi=0.01, p_clean=0.5/0.2), `diffsynth/pipelines/svi_video.py` (conditioning + chaining) |
| `/tmp/svi-wan22` (branch `svi_wan22`, SVI 2.0 Pro, Wan2.2-A14B) | Pro conditioning `y = concat([anchor_latent, motion_latent, padding])` (`diffsynth/pipelines/wan_video_svi_pro.py:467-532`), latent-only chaining (`inference_svi_2.0_pro.py`), the DiffSynth 2.0 training framework (`examples/wanvideo/model_training/train.py`, `diffsynth/diffusion/{training_module,runner,loss,flow_match}.py`), and full Wan2.2-TI2V-5B support (configs, fused embedder, per-frame timestep). **SVI 2.0 (Pro) training code is NOT released — confirmed by grep; we re-implement it.** |

Key SVI-2.0-Pro deltas vs SVI 1.0 (from `docs/svi/svi_2.0_pro.md`):
1. Anchor redesign: anchor = first latent slot only (not padded across all 80 positions).
2. Latent conditioning: last-**latent** conditioning (no decode/re-encode of last frames).
3. Data scaling: more training videos incl. generated ones.

## 3. The core design decision: porting SVI-Pro conditioning to 5B

Wan2.2-TI2V-5B differs structurally from the 14B A14B SVI-Pro base:

| | Wan2.2-I2V-A14B (SVI 2.0 Pro) | Wan2.2-TI2V-5B (ours) |
|---|---|---|
| DiT | 2 experts (high/low noise, boundary 0.875) | single DiT, full timestep range |
| DiT config | dim 5120, 40 layers, in_dim 36 (16+20) | dim 3072, ffn 14336, 30 layers, heads 24, **in_dim = out_dim = 48**, patch [1,2,2], `seperated_timestep=True`, `fuse_vae_embedding_in_latents=True` |
| VAE | Wan2.1 VAE (z_dim 16, spatial 8×) | **Wan2.2 VAE `WanVideoVAE38` (z_dim 48, spatial 16×)**, per-channel mean/std |
| Image cond. | `y`-concat: 4-ch mask + VAE latents, `in_dim 16→36` | **fused**: overwrite `latents[:,:,0:1]=z`, clamp after each step, per-frame timestep (frame 0 → t=0). No `y`, no mask channel, no CLIP |
| CLIP | none | none |
| T5 | umt5-xxl | umt5-xxl (same file) |
| Native res/frames | 480×832×81 | **720P (1280×704 landscape / 704×1248 portrait) @ 24fps, 121 frames/clip (~5s)** — we train and infer at this maximum spec |

The 14B Pro trick (`y = [anchor_latent, motion_latent, zeros]` in extra input channels)
**cannot** be ported directly: the 5B DiT has no conditioning channels (in_dim 48 = z_dim).

**Chosen approach — "SVI-Pro fused conditioning"**: generalize the 5B's native
single-frame hold to an **N-frame clean hold** (N ∈ {0,1,2}), covering both the
model's native tasks — **I2V and T2V** (Wan2.2-TI2V-5B is a joint TI2V model):

- `cond_latents = concat([anchor_latent?, motion_latent?], dim=1)` — 0, 1 or 2 latent frames.
- Before denoising and **after every sampling step**: `latents[:, :, 0:num_cond] = cond_latents`
  (generalizes the existing `first_frame_latents` clamp in `wan_video.py:307-308`).
- **Per-frame timestep map**: frames `0..num_cond-1` → t=0, frames `num_cond..T-1` → t
  (generalizes the `seperated_timestep` block in `model_fn_wan_video`, `wan_video.py:1215-1219`,
  which currently hard-codes exactly one t=0 frame).
- Conditioning modes:
  - **I2V first clip**: `cond = [anchor]` (num_cond=1 — exactly native TI2V I2V behavior).
  - **I2V later clips**: `cond = [anchor, motion]` (num_cond=2).
  - **T2V first clip**: `cond = []` (num_cond=0 — exactly native T2V).
  - **T2V later clips**: `cond = [motion]` (num_cond=1); optionally
    `cond = [anchor_generated, motion]` (num_cond=2) where `anchor_generated` is the first
    latent of clip 1 — a *generated* anchor giving T2V chains the same cross-clip identity
    anchor I2V gets from the user image. Inference flag `--t2v_anchor_mode {none, generated}`,
    default `generated`. Training covers both by mixing anchor-free and anchored samples.
  - motion = `prev_last_latent[:, -num_motion_latent:]`, default `num_motion_latent=1`.
- The 14B "zero padding" slots need no port: they were filler inside a fixed-length `y`;
  in the fused scheme the remaining frames are simply the denoised ones.
- Stitching: clip 0 keeps all 121 pixel frames; later clips drop the first
  `num_overlap_frame` pixel frames (default 5; latent frames 0-1 ≈ pixel frames 0-4
  duplicate the anchor + previous tail), then concatenate.

This preserves every SVI-Pro property — anchor shared across clips, pure-latent
cross-clip hand-off, zero extra inference cost, no decode/re-encode — with **pipeline-only
changes** (no DiT surgery), so a LoRA can learn the behavior.

Rejected alternative: modify the 5B patch-embedding to accept a `y` concat
(48 → 96+mask channels) — requires training new input weights from scratch; defeats LoRA.

## 4. Training: Error-Recycling Fine-Tuning for 5B

We re-implement SVI 1.0's ERFT (paper Sec. 4, `main:train_svi.py`) inside the
DiffSynth 2.0 framework. Flow-matching convention here:
`x_t = (1-σ)x0 + σε`, velocity `v = ε − x0`, σ: 1→0, scheduler `FlowMatchScheduler("Wan")`,
shift=5 (`diffsynth/diffusion/flow_match.py:30-39`).

### 4.1 Training-sample construction (simulates both tasks and both clip positions)

Per sample, dataset returns a **previous-clip block (121 frames)** and a **window block
(121 frames)** taken consecutively from one video (`[s-121..s-1]` and `[s..s+120]`), plus an
**anchor frame** (default: first frame of the video; with augmentation probability
`p_anchor_random≈0.5`, a random frame from the previous-clip region — mirrors SVI 1.0-Shot's
`random_ref_frame` and SVI 2.0's "strong first-frame augmentation", README:185).

**Task sampling** — the same dataset serves both tasks (one code path, `num_cond ∈ {0,1,2}`):

| Mode | Prob. | cond_latents | Simulates |
|---|---|---|---|
| I2V chained | ~0.45 | [anchor, motion] (2) | I2V clip k>1 |
| I2V first clip | ~0.15 | [anchor] (1) | I2V clip 1 / native I2V |
| T2V chained | ~0.30 | [motion] (1) | T2V clip k>1 |
| T2V pure | ~0.10 | [] (0) | T2V clip 1 / native T2V |

(knob `--svi_mode_probs`; also teaches the generated-anchor case because the anchor is
VAE-encoded the same way whether it came from a user image or clip 1 of a generated chain.)

Two separate VAE encodes (both blocks are valid 4k+1 counts; a single 242-frame encode
is not, and latent-boundary alignment must be exact):

- `window_latents` = VAE(window) → 31 latents — the "clean" x0.
- `prev_latents` = VAE(prev block) → 31 latents; `motion_latent = prev_latents[:, :, -1:]`.
- `anchor_latent` = VAE(anchor frame) → 1 latent (skipped for T2V-mode samples).

If the random window starts too early for a previous block, fall back to the matching
**first-clip mode** (no motion) — also teaches clip-1 behavior.

### 4.2 Error injection (paper Eq. 3; SVI 1.0 `train_svi.py:1090-1135`)

Per step, independent Bernoulli draws then a clean override:

| Term | Prob. (default) | Sampled from | Injected into |
|---|---|---|---|
| E_vid (latent err) | 0.9 | `bank_vid`, current timestep grid | generated window latents (frames ≥ num_cond) |
| E_img (cond err) | 0.9 | `bank_vid`, **all grids** (Unif_T), slice `num_cond` consecutive latent frames | `cond_latents` (whichever of anchor/motion are present) |
| E_noi (noise err) | 0.01 | `bank_noi`, current grid | noise ε |
| clean override | 0.5 | — | disables all three (preserve generation ability) |

Corrupted input `x̃_t = (1-σ)(x0 + E_vid) + σ(ε + E_noi)` on generated frames;
cond frames are placed **unnoised** (t=0) into `x̃_t[:, :, 0:num_cond] = cond_latents + E_img`.

**Target (self-corrected)**: `V_rcy = (ε + E_noi) − x0` with the **clean** x0 — the model
sees corrupted input but must predict velocity toward clean data (paper Eq. 6).

Loss: `MSE(v̂, V_rcy)` in fp32 × bsmntw `training_weight(σ)` (`flow_match.py:176-179`),
**masked to generated frames only** (cond frames are clamped; their loss is meaningless.
Flag `--svi_loss_mask_cond_frames`, default True — small, deliberate deviation from stock).

### 4.3 Bidirectional error curation (paper Eq. 4; `train_svi.py:1151-1160`)

Under `no_grad`, from the same forward pass's `v̂` (detached) and `V_rcy`:

- clean endpoint: `x̂0 = x̃_t − σ·v̂`, `x0_corr = x̃_t − σ·V_rcy` → `E_clean = x̂0 − x0_corr = σ(V_rcy − v̂)` → **bank_vid**
- noise endpoint: `ε̂ = x̃_t + (1-σ)·v̂`, `ε_corr = x̃_t + (1-σ)·V_rcy` → `E_noise = ε̂ − ε_corr = (1-σ)(v̂ − V_rcy)` → **bank_noi**

### 4.4 Error replay memory (paper Sec. 4.3; `train_svi.py:684-938`)

New module `diffsynth/diffusion/error_replay.py`, class `ErrorReplayBank`:

- Two banks (vid, noise): `dict[grid_idx -> list[tensor]]`, CPU bf16 storage.
- 50 grids: anchors = timesteps of the 50-step inference schedule (shift=5); a training
  timestep maps to its nearest anchor grid.
- Cap Z=500 per grid; when full, replace the **most L2-similar** stored error
  (vectorized "l2_batch" policy) to preserve diversity.
- Warmup: first `buffer_warmup_iter=50` iterations bank every step (with cross-rank
  all-gather if world_size>1; single-GPU here → local). Afterwards local updates;
  clean-input steps bank only w.p. 0.1. Banks persist across epochs.
- Samplers: `sample_current_grid(bank, t)`, `sample_all_grids(bank)` + temporal slicing
  helper for the 2-frame cond injection.

### 4.5 DiT forward in training

The loss calls a forked `model_fn_wan_video` (living in our new pipeline module) that:

- builds the per-frame timestep map with `num_cond` t=0 frames instead of 1,
- receives `latents = x̃_t` (cond frames already clamped),
- runs with `use_gradient_checkpointing` (and optional CPU offload).

Single DiT → **full timestep range** (no min/max_timestep_boundary split, unlike A14B
high/low expert training).

## 5. Code layout (new code in new files; minimal edits to shared files)

Base: import the `svi_wan22` tree into this repo (see 8.0). Then:

| File | Status | Contents |
|---|---|---|
| `diffsynth/diffusion/error_replay.py` | NEW | `ErrorReplayBank` (banking, L2 replacement, samplers, grid mapping) |
| `diffsynth/diffusion/loss.py` | EDIT (append) | `SVIErrorRecyclingLoss(pipe, **inputs)` implementing 4.2-4.3 |
| `diffsynth/pipelines/wan_video_svi_pro_5b.py` | NEW | `WanVideoSviPro5BPipeline(WanVideoPipeline)`, mirroring the structure of the 14B `wan_video_svi_pro.py`: generalized fused unit `WanVideoUnit_ImageEmbedderFusedSVI` (emits `cond_latents`, `num_cond_frames`, `prev_last_latent` plumbing), forked `model_fn_wan_video` with N-frame timestep map + N-frame clamp in the denoise loop, `__call__` returning `{video, prev_last_latent}` |
| `examples/wanvideo/model_training/svi_pro_5b/train_svi.py` | NEW | fork of `../train.py`: task `"svi"` / `"svi:data_process"` / `"svi:train"` in `task_to_loss`+`launcher_map`, SVI dataset operators (242-frame blocks, anchor frame), CLI args for all ERFT knobs (defaults from paper/shipped SVI) |
| `examples/wanvideo/model_training/svi_pro_5b/svi_pro_5b.sh` | NEW | launch config (Sec. 6) |
| `inference_svi_pro_5b.py` | NEW | **CLI/behavior mirror of the repo's 14B `inference_svi_2.0_pro.py`**: same argument surface (`--prompt` stream file, `--image`, `--lora_path`, `--num_clips`, `--num_motion_latent`, `--num_overlap_frame`, `--cfg_scale`, `--num_inference_steps`, `--sigma_shift`, `--seed_multiplier`, `--height/--width/--num_frames`, `--output_path`), prompt-stream parsing via `ast.literal_eval`, fixed negative prompt, per-clip `seed = clip_idx * seed_multiplier`, anchor always the user first frame, latent hand-off between clips, overlap-drop stitching, per-clip mp4 + final stitched mp4 at fps=24 — but loading the single 5B DiT + Wan2.2_VAE (no high/low experts, no boundary). **Pro only — no non-Pro variant is trained or shipped**; the eval baseline is this same script running the base model without the LoRA |
| `scripts/prepare_dataset.py` | NEW | dataset download/transcode/caption → `metadata.csv` (`video,prompt`) |
| `validate_svi_pro_5b.py` | NEW | short chained generation for smoke + eval |

Training framework reuse (verified): `UnifiedDataset` CSV format, `LoadVideo`/
`ImageCropAndResize` operators, peft LoRA injection (`training_module.py:29-40`),
`ModelLogger` saving trainable-only safetensors with `pipe.dit.` prefix stripped —
loadable via `pipe.load_lora(pipe.dit, path, alpha=1)`.

Split-task pattern: `svi:data_process` runs VAE/T5 units once per sample and caches
`.pth` (extend the unit split so our prev-clip/anchor encodes cache too);
`svi:train` trains DiT-only from cache. This amortizes the 2× VAE encode and fits
VRAM easily. (Single-stage `svi` stays available for smoke tests.)

Gotchas to handle (verified in code): patch out the default S2V audio-processor
download in the forked train script (`train.py:36`); model auto-detection is by
state-dict hash — use the untouched official 5B files; H/W must be divisible by 32
(VAE 2.2, `wan_video.py:152-154`); `num_frames % 4 == 1`.

## 6. Training configuration (defaults)

| Knob | Value | Source |
|---|---|---|
| Base models | `Wan-AI/Wan2.2-TI2V-5B: diffusion_pytorch_model*.safetensors, models_t5_umt5-xxl-enc-bf16.pth, Wan2.2_VAE.pth` | 5B training/inference examples |
| LoRA | rank 128, alpha 128, kaiming, targets `q,k,v,o,ffn.0,ffn.2` | paper Table 9 |
| Optimizer | AdamW, lr **1e-4** (constant; paper says 2e-5, shipped SVI 1.0 + DiffSynth 5B scripts use 1e-4 — start 1e-4, ablate 2e-5), wd 0.01, grad-clip 1.0 | |
| Epochs / batch | 10 epochs, batch 1, grad-accum 1 | paper |
| Resolution / frames | **720P @ 24fps — the 5B model's maximum spec**: 1280×704 landscape (primary) and 704×1248 portrait (config flag), **121-frame window (+121 prev block)** → 31 latents; both dims ÷32-ok, 121 = 4k+1-ok. Smoke tests may use 480×832×81 for speed, but training and final validation run at 720P/121 | 5B max spec |
| Clip fps | **24fps** (the 5B's native/max frame rate) — dataset normalized to 24fps, output saved at fps=24 | 5B max spec |
| Precision | bf16 everything, loss in fp32, LoRA params fp32 | SVI 1.0 |
| Grad checkpointing | on (+ optional CPU offload) | |
| ERFT | p_vid 0.9, p_img 0.9, p_noi 0.01, p_clean 0.5, warmup 50 iters, 50 grids, Z 500 | paper/shipped |
| num_motion_latent | 1 | SVI 2.0 Pro README |
| Data | ~6K videos (MixKit split, as SVI 1.0 Shot/Film); captions via VLM or source metadata | paper Appx. D |
| Hardware | 1× RTX PRO 6000 Blackwell 98GB — 720P×121-frame training needs gradient checkpointing (+ CPU offload if tight); the `svi:data_process` → `svi:train` split keeps the steady-state train pass DiT-only | this machine |

## 7. Inference configuration (defaults)

- **`num_frames=121` at 720P (1280×704 / 704×1248), fps=24 — max spec**, steps 50, cfg 5.0,
  sigma_shift 5.0, `num_motion_latent=1`, `num_overlap_frame=5`, per-clip
  `seed = clip_idx * 42`, fixed negative prompt (same long Chinese negative prompt as the
  14B scripts). Load SVI LoRA via `pipe.load_lora(pipe.dit, lora_path, alpha=1)`
  (alpha is test-time error-recycling intensity, paper Table 7: keep 1.0; degrade ≤0.8 hurts).
- **Both tasks supported**: `--task i2v` (anchor = `--image`) and `--task t2v`
  (`--t2v_anchor_mode {none, generated}`, default `generated` — clip 1's first latent
  becomes the shared anchor for clips ≥2).
- Single DiT → no `switch_DiT_boundary`, no high/low expert LoRA pair; one LoRA covers the
  full timestep range.
- Optional later: lightx2v-style step-distillation LoRA on top (ComfyUI-only for A14B today;
  Pro was redesigned to not conflict with it) — out of scope for v1.

## 8. Milestones (each ends with a validation gate + commit)

### 8.0 Repo & environment
- Import `svi_wan22` tree into this repo (remote `https://github.com/vita-epfl/Stable-Video-Infinity`,
  branch `svi_wan22`; fall back to the local `/tmp` clone), commit as baseline.
- `uv venv` (Python ≥3.10) + `uv pip install -e .` with **torch cu128 build** (Blackwell
  sm_120 requires CUDA ≥12.8 wheels), `accelerate`, `peft`, etc.
- Download Wan2.2-TI2V-5B DiT + Wan2.2_VAE + umt5 (`DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`).
- Gate: `python -c` import diffsynth; stock `examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py`
  (T2V and I2V) generates a video.

### 8.1 Baseline stock LoRA training sanity run
- Run the shipped `lora/Wan2.2-TI2V-5B.sh` (49 frames, example dataset, rank 32, few hundred steps).
- Gate: loss decreases; checkpoint loads with `validate_lora/Wan2.2-TI2V-5B.py` and visibly
  affects output. This de-risks env/framework before any SVI code.

### 8.2 SVI-Pro 5B inference pipeline (no training yet)
- Implement `wan_video_svi_pro_5b.py` + `inference_svi_pro_5b.py` (mirroring the repo's
  14B `inference_svi_2.0_pro.py` CLI/behavior). Pro only — the eval baseline is this same
  pipeline running the **base model** (no SVI LoRA); quality will drift
  (expected — that's what the LoRA will fix) but mechanics must be exact: cond clamp,
  per-frame t-map, latent hand-off, stitching.
- Unit checks: first clip ≡ stock TI2V I2V output given same seed/params; t-map has
  num_cond zeros; latents[:, :, :num_cond] unchanged after each step.
- Gate: 4-clip chained video saved at 720P/24fps, no OOM, no shape errors.

### 8.3 ERFT training implementation
- `error_replay.py`, `SVIErrorRecyclingLoss`, forked `train_svi.py` + dataset operators
  (242-frame blocks, anchor frame), `svi_pro_5b.sh`.
- Unit checks (tiny script, CPU/GPU): bank fill/replace/sample shapes; injection gating
  respects probabilities incl. clean override; error tensors equal the analytic
  `σ(V_rcy−v̂)` / `(1-σ)(v̂−V_rcy)`; first-clip fallback path.
- Gate: 20-step smoke run on a handful of videos — loss finite and decreasing trend,
  banks filling, checkpoint written, `load_lora` round-trip works.

### 8.4 Dataset build
- `scripts/prepare_dataset.py`: fetch MixKit (~6K; Open-Sora-Plan mixkit split),
  transcode/filter (**normalize to 24fps; min length ≥ 242 frames (≈10s @24fps) for full
  samples**, shorter ones marked first-clip-only), center-crop/resize to 1280×704
  (or 704×1248 for portrait sources),
  captions (Pexels titles where present; else local VLM pass, e.g. Qwen2.5-VL),
  write `metadata.csv` + 90/10 train/holdout split.
- Optional phase 2 (SVI 2.0 Pro "data scaling"): add self-generated Wan2.2-5B clips.
- Gate: dataset stats printed; a full epoch length is sensible; spot-check 5 samples visually.

### 8.5 Full training run
- `svi:data_process` cache pass, then `svi:train`: rank 128 LoRA, **720P×121 frames @24fps**, ~10 epochs.
- Monitor: loss curve, bank occupancy per grid, periodic chained validation clips every N steps.
- Gate: chained 10-clip videos at 720P/24fps with the LoRA show visibly less drift than
  8.2 base-model chains.

### 8.6 Evaluation & iteration
- Consistent-setting eval (single prompt, 10-20 clips) and creative-setting (prompt stream):
  base 5B vs 5B+SVI-LoRA, VBench++ subset (subject/background consistency, imaging/aesthetic
  quality, dynamic degree, motion smoothness) on a small fixed suite; watch for the
  metric-fooling failure modes (paper Table 5: near-zero dynamic degree = bad).
  I2V anchors are generated with lodestones/Chroma1-HD (35 steps, CFG 4.0); error/drift
  analysis via ffmpeg frame splitting + per-frame metrics (`scripts/analyze_video.py`).
- Ablations if time: p_img off (expect the big drop, paper Tab. 4), LoRA alpha at test,
  lr 2e-5 vs 1e-4, num_motion_latent 1 vs 2.
- Known risk from paper (B.4): color shift when test style diverges from training
  distribution → mitigate via data diversity if observed.

### 8.7 Publish the checkpoint
- Push the validated LoRA safetensors to **HuggingFace `Impulse2000/svi-model-pro-5b`**
  (auth as Impulse2000 already present): create the model repo if needed, upload
  `svi_pro_5b_lora.safetensors` + a model card (base model, training config, usage with
  `inference_svi_pro_5b.py`, eval results, license note per upstream Apache-2.0).

## 9. Risks / open questions

1. **2-frame clean hold vs Wan 2.2's pretraining** (single held frame): the model must learn
   the 2nd held slot (motion) — that's precisely what the LoRA trains; if rank 128 struggles,
   try holding motion only at frame 1 and anchor only via cross-attn-free means (drop anchor),
   or num_motion_latent>1.
2. **Error-bank cold start**: first ~50 iters train nearly error-free; warmup banking +
   clean-override keep it sane (as SVI 1.0).
3. **VAE 2.2 asymmetry**: latent stats differ (z_dim 48, per-channel mean/std already applied
   inside `WanVideoVAE38.encode`) — errors banked in this space are self-consistent by
   construction; no extra normalization planned.
4. **Frame-boundary alignment**: motion latent must equal "last latent of a full 121-frame
   encode" — hence two separate 4k+1 encodes (Sec. 4.1), never slice a mid-stream encode.
5. **Captions**: MixKit captions are weak; if prompt-following is poor, re-caption with a VLM.
6. **Single GPU**: no cross-rank bank gather (keep the code path, it just no-ops).
7. **720P×121-frame cost**: ~31 latents × 22×39 tokens/frame ≈ 26.6k tokens per sample —
   attention activations dominate VRAM; mitigations: gradient checkpointing (+ CPU offload),
   split `data_process`/`train` so VAE/T5 are absent from the train pass, VAE spatial tiling
   in the cache pass if needed. The 2×121-frame VAE encodes are amortized by the cache pass.
   If OOM persists, drop to 960×544 (still ÷32) as an intermediate rung before 720P.

## 10. Deliverable definition of done

- `inference_svi_pro_5b.py` (CLI mirror of the repo's 14B `inference_svi_2.0_pro.py`)
  generates arbitrarily long chained videos with the trained `svi_pro_5b_lora.safetensors`
  (rank 128) on Wan2.2-TI2V-5B at **720P, 121 frames/clip, 24fps**, **for both tasks** —
  `--task i2v` (user-image anchor) and `--task t2v` (generated or no anchor) — with
  anchor-consistent identity and no visible drift accumulation over ≥10 clips,
  quantitatively beating base 5B chaining on the eval suite.
- All code committed regularly; PLAN.md updated as reality diverges from plan.

## 11. Implementation log (running)

- **8.0 done (2026-08-10)**: svi_wan22 imported as baseline (commit c748715); uv venv
  py3.12 + torch 2.11.0+cu128 (Blackwell sm_120 OK); Wan2.2-TI2V-5B DiT/T5/VAE
  pre-downloaded into `./models/`; stock gate passed — T2V+I2V at 720P/121f/24fps.
  **Deviation**: `redirect_common_files=True` points at a dead HF repo — all our
  entry points pass `redirect_common_files=False` and use the original `.pth` files.
- **8.2/8.3 done (2026-08-10)**: pipeline + ERFT implemented as planned
  (`wan_video_svi_pro_5b.py`, `error_replay.py`, `SVIErrorRecyclingLoss`,
  `train_svi.py`, `inference_svi_pro_5b.py`). Unit tests 18/18; GPU smoke 6/6 —
  SVI first clip is **bit-identical** to stock I2V; clamp checks pass for I2V
  chained, T2V chained, and generated-anchor modes.
- **8.3b done (2026-08-10)**: 60-step smoke training on 10-clip toy set (480×832×121,
  single-stage): loss finite (0.154 @ iter 20), banks filling, 600-tensor bf16 LoRA
  ckpt round-trips through `pipe.load_lora` (300 modules fused), generation sane.
  ~4.3s/step at 480p single-stage.
- **8.4 dataset decision (2026-08-11)**: FastVideo's `mixkit_filtered_6k_wan1.3_t2v`
  holds Wan2.1 latents (unusable); `FastVideo/Mixkit-Src` clips are all exactly 6s
  (too short for chained samples). Using **LanguageBind/Open-Sora-Plan-v1.0.0
  `mixkit.tar.gz` (27GB, the original MixKit pool that SVI 1.0's 6K came from)**,
  streamed via `scripts/prepare_dataset.py`: keep ≥5.1s, normalize to 24fps,
  captions from `llava_path_cap_64x512x512.json` (434K LLaVA captions) with
  de-slug fallback. Target ~1500 kept videos (disk-constrained).
- **Error-bank adaptation (2026-08-10)**: at 720P a full-size bank (50 grids × Z=500
  × 10.5MB) needs ~260GB RAM. Errors are spatially avg-pooled ×2 before banking and
  bilinear-upsampled on injection; Z=200 → ~26GB. Knobs: `--svi_spatial_pool`,
  `--svi_max_errors_per_grid`.
- **8.1 merged into 8.3b**: stock-framework LoRA sanity is covered by the SVI smoke
  run (same accelerate/runner/ModelLogger path), so no separate stock run.
- **Hardware correction + OOM fix (2026-08-13)**: this machine is an **RTX 5090 32GB**,
  not the 98GB RTX PRO 6000 assumed in Sec. 6 — 720P×121f single-stage training OOM'd
  with all-bf16 models. Fix in `svi_pro_5b.sh`: fp8 storage for T5+VAE
  (`--fp8_models`, compute stays bf16), `--vae_tiled`, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Verified: sustained training
  cycles at 100% GPU, peak ~30.0GB. Dataset rebuilt properly at true 720p:
  1500 MixKit CDN originals, all 1280×720@24fps (360p fallback removed from
  `prepare_dataset_v2.py`), 95% chained-capable, 279 LLaVA + 1221 cleaned-slug
  captions. If OOM recurs: add `--use_gradient_checkpointing_offload`, then fall
  back to 960×544 (Risk 7).
- **Holdout validation (2026-08-13)**: `train_svi.py` now does a deterministic
  seeded 90/10 train/val split (fixed across resumes) and, every `--val_every 200`
  steps, evaluates `--val_batches 8` fixed held-out batches with clean inputs
  (p_clean=1.0, seeded RNG incl. new `rng_seed` arg on `SVIErrorRecyclingLoss`,
  errors banked to a throwaway capacity-1 bank) — appended to
  `<output_path>/val_log.csv` and shown in the tqdm postfix. Smoke-tested:
  events fired, CSV rows written, no GPU-memory change.
