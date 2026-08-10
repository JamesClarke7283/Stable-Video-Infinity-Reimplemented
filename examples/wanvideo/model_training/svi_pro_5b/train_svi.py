# SPDX-License-Identifier: Apache-2.0
# Derived from DiffSynth-Studio examples/wanvideo/model_training/train.py
# (https://github.com/modelscope/DiffSynth-Studio; SVI branch:
# https://github.com/vita-epfl/Stable-Video-Infinity, branch svi_wan22),
# Copyright ModelScope Team, licensed under the Apache License 2.0.
# SVI Error-Recycling Fine-Tuning training entry for Wan2.2-TI2V-5B by the
# Stable-Video-Infinity-Reimplemented authors.
import os, argparse, random, warnings
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
from diffsynth.diffusion.runner import launch_training_task
from diffsynth.diffusion.parsers import add_general_config, add_video_size_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        if self.iteration % 100 == 0:
            print(f"[svi] iter={self.iteration} loss={loss.item():.5f} bank={self.error_bank.occupancy()}")
        return loss


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
    return parser


if __name__ == "__main__":
    parser = svi_parser()
    args = parser.parse_args()
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
    launch_training_task(accelerator, dataset, model, model_logger, args=args)
