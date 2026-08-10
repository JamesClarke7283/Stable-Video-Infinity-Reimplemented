# SPDX-License-Identifier: Apache-2.0
# Derived from Stable-Video-Infinity inference_svi_2.0_pro.py
# (https://github.com/vita-epfl/Stable-Video-Infinity, branch svi_wan22),
# Copyright VITA @ EPFL, licensed under the Apache License 2.0.
# Adapted for Wan2.2-TI2V-5B (single DiT, fused SVI-Pro conditioning, T2V + I2V)
# by the Stable-Video-Infinity-Reimplemented authors.
import torch
from PIL import Image
import numpy as np
import os
import ast
import argparse
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video_svi_pro_5b import WanVideoSviPro5BPipeline, ModelConfig

NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


class StreamingVideoProcessor:
    def __init__(self, lora_path=None, lora_alpha=1.0, task="i2v", t2v_anchor_mode="generated",
                 seed_multiplier=42, num_motion_latent=1, num_overlap_frame=5,
                 cfg_scale=5.0, num_inference_steps=50, sigma_shift=5.0, tiled=True):
        self.lora_path = lora_path
        self.lora_alpha = lora_alpha
        self.pipe = None
        self.initialize_pipeline()

        # Configuration (defaults: Wan2.2-TI2V-5B max spec)
        self.frames_per_clip = 121
        self.height = 704
        self.width = 1280
        self.fps = 24
        self.num_clips = 15
        self.task = task
        self.t2v_anchor_mode = t2v_anchor_mode
        self.seed_multiplier = seed_multiplier
        self.num_motion_latent = num_motion_latent
        self.num_overlap_frame = num_overlap_frame
        self.cfg_scale = cfg_scale
        self.num_inference_steps = num_inference_steps
        self.sigma_shift = sigma_shift
        self.tiled = tiled

    def initialize_pipeline(self):
        print("Initializing Wan2.2-TI2V-5B SVI-Pro pipeline...")
        self.pipe = WanVideoSviPro5BPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
            ],
            redirect_common_files=False,
        )
        if self.lora_path is not None and self.lora_path != "none":
            print(f"Loading SVI LoRA: {self.lora_path} (alpha={self.lora_alpha})")
            self.pipe.load_lora(self.pipe.dit, self.lora_path, alpha=self.lora_alpha)
        print("Pipeline initialized successfully!")

    def load_prompts_from_file(self, prompt_file_path):
        """Load prompts from a text file containing a Python list"""
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if 'prompts = [' in content:
                start_idx = content.find('prompts = [')
                prompts_str = content[start_idx + len('prompts = '):]
                prompts = ast.literal_eval(prompts_str)
            else:
                prompts = ast.literal_eval(content.strip())
            return prompts
        except Exception as e:
            print(f"Error loading prompts from {prompt_file_path}: {e}")
            return []

    def generate_streaming_video(self, input_image_path=None, prompt_path=None, output_dir=".", sample_name="sample"):
        if self.task == "i2v":
            if input_image_path is None or not os.path.exists(input_image_path):
                print(f"Warning: I2V task needs a valid --ref_image_path, got {input_image_path}")
                return
            input_image = Image.open(input_image_path).resize((self.width, self.height))
            anchor_image = input_image
        else:
            input_image = None
            anchor_image = None

        if prompt_path is not None:
            prompts = self.load_prompts_from_file(prompt_path)
        else:
            prompts = [self.single_prompt]
        if not prompts:
            print(f"Warning: No valid prompts found in {prompt_path}")
            return
        print(f"Task: {self.task} | clips: {self.num_clips} | prompts: {len(prompts)}")

        all_video_frames = []
        prev_last_latent = None
        generated_anchor_latent = None

        for clip_idx in range(self.num_clips):
            prompt = prompts[clip_idx % len(prompts)]
            print(f"\nGenerating clip {clip_idx + 1}/{self.num_clips}... Prompt: {prompt[:100]}")

            # SVI-Pro conditioning per task/clip position
            anchor = None
            anchor_latent = None
            if self.task == "i2v":
                anchor = anchor_image  # shared across all clips
            elif self.t2v_anchor_mode == "generated" and generated_anchor_latent is not None:
                anchor_latent = generated_anchor_latent  # clip 1's first latent

            video_clip_dict = self.pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                seed=clip_idx * self.seed_multiplier,
                tiled=self.tiled,
                height=self.height,
                width=self.width,
                input_image=input_image,
                anchor=anchor,
                anchor_latent=anchor_latent,
                prev_last_latent=prev_last_latent,
                num_motion_latent=self.num_motion_latent,
                num_frames=self.frames_per_clip,
                cfg_scale=self.cfg_scale,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
            )
            video_clip = video_clip_dict["video"]
            prev_last_latent = video_clip_dict["prev_last_latent"]
            if self.task == "t2v" and clip_idx == 0:
                generated_anchor_latent = prev_last_latent[:, :, 0:1].clone()

            if isinstance(video_clip, torch.Tensor):
                video_frames = [Image.fromarray((frame.cpu().numpy() * 255).astype(np.uint8)) if video_clip.max() <= 1
                                else Image.fromarray(frame.cpu().numpy().astype(np.uint8))
                                for frame in video_clip]
            else:
                video_frames = video_clip if isinstance(video_clip, list) else [video_clip]

            # First clip keeps all frames; later clips drop the overlap (anchor +
            # motion slots duplicate the previous tail).
            if clip_idx == 0:
                all_video_frames.extend(video_frames)
            else:
                all_video_frames.extend(video_frames[self.num_overlap_frame:])

            print(f"Clip {clip_idx + 1} generated: {len(video_frames)} frames")
            intermediate_output = os.path.join(output_dir, f"{sample_name}_clip_{clip_idx + 1}.mp4")
            save_video(all_video_frames, intermediate_output, fps=self.fps, quality=7)
            print(f"Saved intermediate: {intermediate_output} ({len(all_video_frames)} frames)")

        final_output = os.path.join(output_dir, f"{sample_name}_streaming_final.mp4")
        print(f"\nSaving final video with {len(all_video_frames)} frames...")
        save_video(all_video_frames, final_output, fps=self.fps, quality=5)
        print(f"Final video saved: {final_output} ({len(all_video_frames)} frames at {self.fps} FPS)")
        return final_output


def main():
    parser = argparse.ArgumentParser(description="SVI-Pro streaming video generation for Wan2.2-TI2V-5B")

    # Path arguments
    parser.add_argument("--output_root", type=str, default="./outputs/svi_pro_5b", help="Path to the output directory")
    parser.add_argument("--lora_path", type=str, default="none", help="Path to the SVI LoRA safetensors (or 'none' for the base model)")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha (test-time error-recycling intensity)")
    parser.add_argument("--ref_image_path", type=str, default=None, help="Path to the reference image (I2V task)")
    parser.add_argument("--prompt_path", type=str, default=None, help="Path to the prompt-stream file (Python list)")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt for all clips (consistent setting)")
    parser.add_argument("--sample_name", type=str, default="sample", help="Output file name prefix")

    # Task
    parser.add_argument("--task", type=str, default="i2v", choices=["i2v", "t2v"], help="Generation task")
    parser.add_argument("--t2v_anchor_mode", type=str, default="generated", choices=["none", "generated"],
                        help="T2V chaining: 'generated' anchors on clip 1's first latent, 'none' uses motion only")

    # Processing arguments
    parser.add_argument("--num_clips", type=int, default=15, help="Number of clips to generate")

    # Model/generation arguments (defaults: Wan2.2-TI2V-5B max spec)
    parser.add_argument("--height", type=int, default=704, help="Video height")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--fps", type=int, default=24, help="Video frames per second")
    parser.add_argument("--frames_per_clip", type=int, default=121, help="Number of frames per clip")
    parser.add_argument("--seed_multiplier", type=int, default=42, help="Seed multiplier (seed = clip_idx * seed_multiplier)")
    parser.add_argument("--num_overlap_frame", type=int, default=5, help="Number of overlapping frames dropped between clips")
    parser.add_argument("--num_motion_latent", type=int, default=1, help="Number of motion latents carried between clips")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Scheduler sigma shift")
    parser.add_argument("--tiled", action="store_true", help="Tiled VAE encode/decode")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    processor = StreamingVideoProcessor(
        lora_path=args.lora_path,
        lora_alpha=args.lora_alpha,
        task=args.task,
        t2v_anchor_mode=args.t2v_anchor_mode,
        seed_multiplier=args.seed_multiplier,
        num_motion_latent=args.num_motion_latent,
        num_overlap_frame=args.num_overlap_frame,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        tiled=args.tiled,
    )
    processor.frames_per_clip = args.frames_per_clip
    processor.height = args.height
    processor.width = args.width
    processor.fps = args.fps
    processor.num_clips = args.num_clips
    processor.single_prompt = args.prompt

    processor.generate_streaming_video(args.ref_image_path, args.prompt_path, args.output_root, args.sample_name)
    print("\n🎉 All processing completed!")


if __name__ == "__main__":
    main()
