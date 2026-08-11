# Generate anchor images for SVI-Pro I2V validation with lodestones/Chroma1-HD.
# Recommended settings (per model card / user instruction): 35 steps, CFG 4.0.
# Run: .venv/bin/python scripts/gen_anchor_images.py --out_dir data/anchors
import argparse
import os

import torch

PROMPTS = [
    "a fluffy orange tabby cat sitting on a wooden dock at golden hour, photorealistic, cinematic lighting",
    "a red fox standing in fresh snowfall in a pine forest, photorealistic, shallow depth of field",
    "a vintage blue sports car parked on a coastal road at dusk, cinematic, dramatic sky",
    "a cozy mountain cabin with warm lights in the windows under a starry night sky, photorealistic",
    "a majestic sailing ship on turquoise ocean waves, sunny day, cinematic wide shot",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/anchors")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--model_id", default="lodestones/Chroma1-HD")
    args = parser.parse_args()

    from diffusers import ChromaPipeline

    os.makedirs(args.out_dir, exist_ok=True)
    pipe = ChromaPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    prompts = PROMPTS[: args.num_images] if args.num_images else PROMPTS
    for i, prompt in enumerate(prompts):
        image = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=torch.Generator("cuda").manual_seed(1234 + i),
        ).images[0]
        path = os.path.join(args.out_dir, f"anchor_{i:02d}.png")
        image.save(path)
        print(f"saved {path}: {prompt[:60]}")


if __name__ == "__main__":
    main()
