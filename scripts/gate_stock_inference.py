# Gate script: stock Wan2.2-TI2V-5B inference (T2V + I2V) via DiffSynth.
# Original code pattern from examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py.
import os, subprocess, sys
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
OUT = "outputs/gate"
os.makedirs(OUT, exist_ok=True)

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    redirect_common_files=False,
)

NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
PROMPT = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"

# T2V
video = pipe(prompt=PROMPT, negative_prompt=NEG, seed=0, tiled=True,
             height=704, width=1248, num_frames=121)
save_video(video, f"{OUT}/stock_t2v.mp4", fps=24, quality=5)

# I2V: use frame 0 of the T2V video as input image
frame0 = f"{OUT}/stock_t2v_frame0.png"
subprocess.run(["/tmp/ffmpeg-static/ffmpeg", "-y", "-i", f"{OUT}/stock_t2v.mp4",
                "-frames:v", "1", frame0], check=True, capture_output=True)
input_image = Image.open(frame0).resize((1248, 704))
video = pipe(prompt=PROMPT, negative_prompt=NEG, seed=0, tiled=True,
             height=704, width=1248, input_image=input_image, num_frames=121)
save_video(video, f"{OUT}/stock_i2v.mp4", fps=24, quality=5)
print("GATE_OK")
