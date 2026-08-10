import torch
from PIL import Image
import numpy as np
import os
import glob
import ast
import argparse
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video_svi_pro import WanVideoSviProPipeline, ModelConfig

class StreamingVideoProcessor:
    def __init__(self, lora_path_high="",lora_path_low="", use_anchor=False, seed_multiplier=123, num_motion_frame=1,num_motion_latent=2, num_overlap_frame=1, cfg_scale=7.0):
        self.lora_path_high = lora_path_high
        self.lora_path_low = lora_path_low
        self.pipe = None
        self.initialize_pipeline()
        
        # Configuration
        self.frames_per_clip = 81  # Frames in each clip
        self.height = 480
        self.width = 832
        self.fps = 15
        self.num_clips = 15  # Default number of clips
        self.use_anchor = use_anchor 
        self.seed_multiplier = seed_multiplier  # Seed multiplier for random generation
        self.num_motion_frame = num_motion_frame
        self.num_motion_latent = num_motion_latent
        self.num_overlap_frame = num_overlap_frame
        self.cfg_scale = cfg_scale
        
    def initialize_pipeline(self):
        """Initialize the WanVideo pipeline"""
        print("Initializing WanVideo pipeline...")
        self.pipe = WanVideoSviProPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
            ],
        )
        self.pipe.load_lora(self.pipe.dit, self.lora_path_high, alpha=1)
        self.pipe.load_lora(self.pipe.dit2, self.lora_path_low, alpha=1)


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
                # Try to parse directly as a list
                prompts = ast.literal_eval(content.strip())
                
            return prompts
        except Exception as e:
            print(f"Error loading prompts from {prompt_file_path}: {e}")
            return []
    
    def generate_streaming_video(self, input_image_path, prompt_path, output_dir):
        """Generate streaming video using multiple prompts"""
        sample_name = os.path.dirname(input_image_path).split("/")[-1]
        print(f"\nProcessing sample: {sample_name}")
        if not os.path.exists(input_image_path):
            print(f"Warning: Input image not found in {input_image_path}")
            return
        print(f"Input image: {input_image_path}")
        prompts = self.load_prompts_from_file(prompt_path)
        if not prompts:
            print(f"Warning: No valid prompts found in {prompt_path}")
            return
        print(f"Number of prompts: {len(prompts)}")
        
        # Load input image
        input_image = Image.open(input_image_path).resize((self.width, self.height))
        
        # Generate clips using different prompts
        all_video_frames = []
        current_input_image = input_image
        
        num_clips = self.num_clips
        prev_last_latent = None
        for clip_idx in range(num_clips):
            print(f"\nGenerating clip {clip_idx + 1}/{num_clips}...")
            print(f"Prompt: {prompts[clip_idx]}...")  # Show first 100 characters
            
            # try:
            video_clip_dict = self.pipe(
                prompt=prompts[clip_idx],
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                seed=clip_idx * self.seed_multiplier,
                tiled=False,
                height=self.height,
                width=self.width,
                input_image=current_input_image,
                num_frames=self.frames_per_clip,
                anchor=input_image,
                prev_last_latent = prev_last_latent,
                num_motion_latent=self.num_motion_latent,
                cfg_scale=self.cfg_scale,
            )
            video_clip = video_clip_dict["video"]
            if self.num_motion_latent > 0:
                prev_last_latent = video_clip_dict["prev_last_latent"]
                
            # Convert video_clip to list of PIL Images if needed
            if isinstance(video_clip, torch.Tensor):
                # Assuming video_clip is [T, H, W, 3] in range [0, 1] or [0, 255]
                video_frames = [Image.fromarray((frame.cpu().numpy() * 255).astype(np.uint8)) if video_clip.max() <= 1 
                                else Image.fromarray(frame.cpu().numpy().astype(np.uint8)) 
                                for frame in video_clip]
            else:
                video_frames = video_clip if isinstance(video_clip, list) else [video_clip]
            
            # For first clip, save all frames; for subsequent clips, skip first frame to avoid duplication
            if clip_idx == 0:
                all_video_frames.extend(video_frames)
            else:

                # all_video_frames = all_video_frames[:-self.num_overlap_frame]  # Remove overlapping frames
                all_video_frames = all_video_frames 
                all_video_frames.extend(video_frames[self.num_overlap_frame:])


            current_input_image = video_frames[-self.num_motion_frame:]
            
            print(f"Clip {clip_idx + 1} generated: {len(video_frames)} frames")
            
            intermediate_output = os.path.join(output_dir, f"{sample_name}_clip_{clip_idx + 1}.mp4")
            save_video(all_video_frames, intermediate_output, fps=self.fps, quality=7)
            print(f"Saved intermediate: {intermediate_output} ({len(all_video_frames)} frames)")
                

        # Save the final full video
        final_output = os.path.join(output_dir, f"{sample_name}_streaming_final.mp4")
        print(f"\nSaving final video with {len(all_video_frames)} frames...")
        save_video(all_video_frames, final_output, fps=self.fps, quality=5)
        print(f"Final video saved: {final_output} ({len(all_video_frames)} frames at {self.fps} FPS)")
        
        return final_output

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Streaming Video Generation with WanVideo")
    
    # Path arguments
    parser.add_argument(
        "--output_root",
        type=str,
        default="/iopsstor/scratch/cscs/wuli/DiffSynth-Studio/videos/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--lora_path_high",
        type=str,
        default="none",
        help="Path to the high noise LoRA model file"
    )
    parser.add_argument(
        "--lora_path_low",
        type=str,
        default="none",
        help="Path to the low noise LoRA model file"
    )

    parser.add_argument(
        "--ref_image_path",
        type=str,
        default="./data/toy_test/frame.jpg",
        help="Path to the reference image"
    )

    parser.add_argument(
        "--prompt_path",
        type=str,
        default="./data/toy_test/prompt.txt",
        help="Path to the prompt file"
    )
    
    # Processing arguments
    parser.add_argument(
        "--num_clips",
        type=int,
        default=15,
        help="Number of clips to generate per sample"
    )
    
    # Model/generation arguments
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Video frames per second"
    )
    parser.add_argument(
        "--frames_per_clip",
        type=int,
        default=81,
        help="Number of frames per clip"
    )

    parser.add_argument(
        "--seed_multiplier",
        type=int,
        default=42,
        help="Seed multiplier for random generation (seed = clip_idx * seed_multiplier)"
    )
    parser.add_argument(
        "--use_latent_conditioning",
        action="store_true",
        help="Enable latent conditioning for video generation"
    )
    parser.add_argument(
        "--num_overlap_frame",
        type=int,
        default=4,
        help="Number of overlapping frames between clips"
    )
    
    parser.add_argument(
        "--num_motion_frame",
        type=int,
        default=4,
        help="Number of frames to look back for the next input image (default: 1)"
    )

    parser.add_argument(
        "--num_motion_latent",
        type=int,
        default=1,
        help="Number of latents to use from input frames (default: 2)"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="CFG scale for classifier-free guidance (default: 5.0)"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_root, exist_ok=True)
    
    # Initialize processor
    processor = StreamingVideoProcessor(
        lora_path_high=args.lora_path_high,
        lora_path_low=args.lora_path_low,
        use_anchor=True,
        seed_multiplier=args.seed_multiplier, 
        num_motion_frame=args.num_motion_frame, 
        num_motion_latent=args.num_motion_latent, 
        num_overlap_frame=args.num_overlap_frame,
        cfg_scale=args.cfg_scale,
        )
    
    # Update processor configuration
    processor.frames_per_clip = args.frames_per_clip
    processor.height = args.height
    processor.width = args.width
    processor.fps = args.fps
    processor.num_clips = args.num_clips

    processor.generate_streaming_video(args.ref_image_path, args.prompt_path, args.output_root)
    
    print("\n🎉 All processing completed!")

if __name__ == "__main__":
    main()