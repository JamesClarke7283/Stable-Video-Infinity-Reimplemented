#!/bin/bash
# SVI-Pro ERFT training for Wan2.2-TI2V-5B — production config (PLAN.md Sec. 6).
# Max spec: 720P (1280x704), 121-frame clips, 24fps data, LoRA rank 128.
# Usage: bash examples/wanvideo/model_training/svi_pro_5b/svi_pro_5b.sh
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

.venv/bin/accelerate launch --num_processes 1 examples/wanvideo/model_training/svi_pro_5b/train_svi.py \
  --dataset_base_path data/svi_mixkit \
  --dataset_metadata_path data/svi_mixkit/metadata.csv \
  --height 704 \
  --width 1280 \
  --num_frames 121 \
  --dataset_repeat 1 \
  --num_epochs 10 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --lora_base_model dit \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 128 \
  --use_gradient_checkpointing \
  --output_path models/train/svi_pro_5b_lora \
  --remove_prefix_in_ckpt "pipe.dit." \
  --save_steps 1000 \
  --dataset_num_workers 2 \
  --num_motion_latent 1 \
  --svi_p_vid 0.9 \
  --svi_p_img 0.9 \
  --svi_p_noi 0.01 \
  --svi_p_clean 0.5 \
  --svi_warmup_iter 50 \
  --svi_max_errors_per_grid 200 \
  --svi_spatial_pool 2 \
  --svi_mode_probs "0.45,0.15,0.30,0.10" \
  --svi_loss_mask_cond_frames 1
