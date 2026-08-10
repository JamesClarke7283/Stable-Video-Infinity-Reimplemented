# GPU smoke test for the SVI-Pro 5B inference pipeline (milestone 8.2 gate).
# Runs at reduced spec (480x832, 49 frames, 20 steps) for speed. Checks:
#   1. SVI first clip (I2V) is pixel-equivalent to the stock WanVideoPipeline I2V
#      (same seed/params) -- proves the num_cond=1 path reproduces native behavior.
#   2. I2V chaining: cond clamp holds (prev_last_latent keeps anchor + motion).
#   3. T2V chaining with generated anchor: clamps hold, videos save.
# Run: .venv/bin/python scripts/smoke_svi_inference.py
import os
import numpy as np
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.pipelines.wan_video_svi_pro_5b import WanVideoSviPro5BPipeline

MODEL_CONFIGS = [
    ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
    ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
    ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
]
TOKENIZER = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
PROMPT = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
H, W, F, STEPS = 480, 832, 49, 20
OUT = "outputs/smoke"
os.makedirs(OUT, exist_ok=True)
PASS = []


def check(name, cond):
    PASS.append((name, bool(cond)))
    print(("PASS" if cond else "FAIL"), "-", name)


def frames_of(video):
    return [(np.asarray(f).astype(np.float32)) for f in video]


# Use the gate T2V frame as the I2V anchor image.
anchor_image = Image.open("outputs/gate/stock_t2v_frame0.png").resize((W, H))

# ---------------------------------------------------------------- SVI pipeline
svi = WanVideoSviPro5BPipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=MODEL_CONFIGS, tokenizer_config=TOKENIZER, redirect_common_files=False,
)

# Clip 1 (I2V): anchor = input image, no prev -> num_cond = 1
out1 = svi(prompt=PROMPT, negative_prompt=NEG, seed=0, tiled=True, height=H, width=W,
           input_image=anchor_image, num_frames=F, cfg_scale=5.0,
           num_inference_steps=STEPS, sigma_shift=5.0)
save_video(out1["video"], f"{OUT}/svi_i2v_clip1.mp4", fps=24, quality=5)

# Clamp check: the returned latents' frame 0 must equal the VAE-encoded anchor.
anchor_latent = svi.vae.encode(
    [svi.preprocess_image(anchor_image).transpose(0, 1)], device="cuda", tiled=True,
    tile_size=(30, 52), tile_stride=(15, 26),
).to(torch.bfloat16)
check("i2v clip1: frame-0 latent == anchor latent (clamp held)",
      torch.allclose(out1["prev_last_latent"][:, :, :1].cpu(), anchor_latent.cpu(), atol=1e-5))

# Clip 2 (I2V chained): anchor + motion -> num_cond = 2
out2 = svi(prompt=PROMPT, negative_prompt=NEG, seed=42, tiled=True, height=H, width=W,
           input_image=anchor_image, anchor=anchor_image,
           prev_last_latent=out1["prev_last_latent"], num_motion_latent=1,
           num_frames=F, cfg_scale=5.0, num_inference_steps=STEPS, sigma_shift=5.0)
save_video(out2["video"], f"{OUT}/svi_i2v_clip2.mp4", fps=24, quality=5)
check("i2v clip2: frame-0 latent == anchor latent (clamp held)",
      torch.allclose(out2["prev_last_latent"][:, :, :1].cpu(), anchor_latent.cpu(), atol=1e-5))
check("i2v clip2: frame-1 latent == clip1 last latent (motion clamp held)",
      torch.allclose(out2["prev_last_latent"][:, :, 1:2].cpu(), out1["prev_last_latent"][:, :, -1:].cpu(), atol=1e-5))

# T2V chain: clip 1 pure T2V, clip 2 with generated anchor + motion
tout1 = svi(prompt=PROMPT, negative_prompt=NEG, seed=1, tiled=True, height=H, width=W,
            num_frames=F, cfg_scale=5.0, num_inference_steps=STEPS, sigma_shift=5.0)
save_video(tout1["video"], f"{OUT}/svi_t2v_clip1.mp4", fps=24, quality=5)
gen_anchor = tout1["prev_last_latent"][:, :, 0:1].clone()
tout2 = svi(prompt=PROMPT, negative_prompt=NEG, seed=43, tiled=True, height=H, width=W,
            anchor_latent=gen_anchor, prev_last_latent=tout1["prev_last_latent"],
            num_motion_latent=1, num_frames=F, cfg_scale=5.0,
            num_inference_steps=STEPS, sigma_shift=5.0)
save_video(tout2["video"], f"{OUT}/svi_t2v_clip2.mp4", fps=24, quality=5)
check("t2v clip2: frame-0 latent == generated anchor (clamp held)",
      torch.allclose(tout2["prev_last_latent"][:, :, :1].cpu(), gen_anchor.cpu(), atol=1e-5))
check("t2v clip2: frame-1 latent == clip1 last latent (motion clamp held)",
      torch.allclose(tout2["prev_last_latent"][:, :, 1:2].cpu(), tout1["prev_last_latent"][:, :, -1:].cpu(), atol=1e-5))

svi_video_clip1 = [Image.fromarray((f.cpu().numpy() * 255).astype(np.uint8)) if isinstance(f, torch.Tensor) else f for f in out1["video"]]
del svi
torch.cuda.empty_cache()

# -------------------------------------------------------------- stock pipeline
stock = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=MODEL_CONFIGS, tokenizer_config=TOKENIZER, redirect_common_files=False,
)
stock_video = stock(prompt=PROMPT, negative_prompt=NEG, seed=0, tiled=True, height=H, width=W,
                    input_image=anchor_image, num_frames=F, cfg_scale=5.0,
                    num_inference_steps=STEPS, sigma_shift=5.0)
save_video(stock_video, f"{OUT}/stock_i2v.mp4", fps=24, quality=5)
stock_frames = [Image.fromarray((f.cpu().numpy() * 255).astype(np.uint8)) if isinstance(f, torch.Tensor) else f for f in stock_video]

diffs = [np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).mean()
         for a, b in zip(svi_video_clip1, stock_frames)]
mean_diff = float(np.mean(diffs))
print(f"SVI-first-clip vs stock-I2V mean abs pixel diff: {mean_diff:.4f} (of 255)")
check("SVI first clip == stock I2V (mean abs diff < 0.5)", mean_diff < 0.5)

failed = [n for n, ok in PASS if not ok]
print(f"\n{len(PASS) - len(failed)}/{len(PASS)} passed")
if failed:
    raise SystemExit(1)
print("SMOKE_OK")
