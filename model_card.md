---
license: apache-2.0
base_model: Wan-AI/Wan2.2-TI2V-5B
tags:
  - video-generation
  - lora
  - long-video
  - stable-video-infinity
  - wan2.2
---

# SVI-Pro LoRA for Wan2.2-TI2V-5B

An **SVI-Pro-style error-recycling LoRA** that gives **Wan2.2-TI2V-5B** infinite-length
video generation via clip chaining — the missing 5B member of the
[Stable-Video-Infinity](https://github.com/vita-epfl/Stable-Video-Infinity) family
(SVI 2.0 Pro ships only for Wan2.2-I2V-A14B).

- **Tasks**: T2V and I2V (the TI2V base does both)
- **Spec**: 720P (1280×704 / 704×1248), 121 frames/clip (~5s), 24fps, chained to arbitrary length
- **Conditioning** (SVI-Pro port to the 5B fused scheme): `cond = [anchor_latent, motion_latent]`
  held clean (t=0, clamped every step) — anchor = user first frame (I2V) or generated
  first-clip latent (T2V), motion = last latent of the previous clip (pure latent hand-off)
- **Training**: Error-Recycling Fine-Tuning (arXiv 2510.09212) re-implemented for the
  single-DiT 5B model — DiT self-errors banked into a 50-grid replay memory and
  re-injected during training (p_vid=0.9, p_img=0.9, p_noi=0.01, p_clean=0.5),
  LoRA rank 128 on `q,k,v,o,ffn.0,ffn.2`, 1500 MixKit original videos, 10 epochs, 720P

## Files

- `svi_pro_5b_lora.safetensors` — the LoRA (rank 128, bf16, `pipe.dit.`-stripped keys)

## Usage (DiffSynth-Studio / SVI codebase)

Code: [Stable-Video-Infinity-Reimplemented](https://huggingface.co/Impulse2000) —
`inference_svi_pro_5b.py`:

```bash
python inference_svi_pro_5b.py \
  --task i2v --ref_image_path anchor.png --prompt_path prompts.txt \
  --lora_path svi_pro_5b_lora.safetensors --num_clips 20 \
  --height 704 --width 1280 --frames_per_clip 121 --fps 24
```

The LoRA loads with `pipe.load_lora(pipe.dit, path, alpha=1)`. Keep alpha=1.0
(alpha is the error-recycling intensity; ≤0.8 measurably degrades correction).

## Provenance / licenses

- Training algorithm: *Stable Video Infinity* (VITA @ EPFL), Apache-2.0, arXiv 2510.09212
- Framework: DiffSynth-Studio (ModelScope Team), Apache-2.0
- Base model: Wan2.2-TI2V-5B (Wan-AI / Alibaba), Apache-2.0
- Training data: MixKit free stock videos (MixKit license), normalized to 24fps

*Eval results and training curves are added to this card after validation.*
