# SVI-Pro 5B validation runner (milestone 8.6): generates chained videos with a
# trained LoRA (or the base model for the baseline) for both I2V (Chroma1-HD
# anchors) and T2V (generated anchor), then runs the ffmpeg drift/error analysis
# on each output. Compare base vs LoRA runs to verify error recycling.
# Run: .venv/bin/python scripts/validate_svi.py --lora_path models/train/svi_pro_5b_lora/epoch-9.safetensors --num_clips 8
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

I2V_CASES = [
    # (anchor_image, prompt stream)
    ("data/anchors/anchor_00.png", [
        "a fluffy orange tabby cat sitting on a wooden dock at golden hour, the cat lazily stretches its front paws",
        "a fluffy orange tabby cat on a wooden dock at golden hour, the cat stands up and shakes itself",
        "a fluffy orange tabby cat walking along a wooden dock at golden hour, tail raised high",
        "a fluffy orange tabby cat on a wooden dock at golden hour, it pauses and looks at the water",
        "a fluffy orange tabby cat on a wooden dock at golden hour, it sits down and grooms its paw",
        "a fluffy orange tabby cat on a wooden dock at golden hour, the sun sets lower, long shadows",
        "a fluffy orange tabby cat on a wooden dock at golden hour, it yawns widely showing its fangs",
        "a fluffy orange tabby cat on a wooden dock at dusk, the cat curls up to rest",
    ]),
    ("data/anchors/anchor_02.png", [
        "a vintage blue sports car parked on a coastal road at dusk, its headlights turn on",
        "a vintage blue sports car on a coastal road at dusk, it slowly pulls forward onto the road",
        "a vintage blue sports car driving along the coastal road at dusk, waves crashing nearby",
        "a vintage blue sports car driving along the coastal road at dusk, passing palm trees",
        "a vintage blue sports car driving along the coastal road at dusk, the sky darkens to deep blue",
        "a vintage blue sports car driving along the coastal road at night, headlights illuminating the road",
        "a vintage blue sports car driving along the coastal road at night, it rounds a bend",
        "a vintage blue sports car parked on the coastal road at night, stars visible above",
    ]),
]

T2V_CASES = [
    [
        "a majestic sailing ship on turquoise ocean waves, sailing steadily forward, sunny day, cinematic wide shot",
        "a majestic sailing ship on turquoise ocean waves, the wind picks up and the sails fill",
        "a majestic sailing ship on turquoise ocean waves, it cuts through larger swells, spray at the bow",
        "a majestic sailing ship on turquoise ocean waves, seagulls circle above the mast",
        "a majestic sailing ship on turquoise ocean waves, the camera slowly orbits the ship",
        "a majestic sailing ship on turquoise ocean waves, golden light begins to color the water",
        "a majestic sailing ship on turquoise ocean waves at golden hour, sails glowing warm",
        "a majestic sailing ship on turquoise ocean waves at sunset, silhouetted against the sun",
    ],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", default="none")
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--num_clips", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--skip_t2v", action="store_true")
    parser.add_argument("--skip_i2v", action="store_true")
    args = parser.parse_args()

    from inference_svi_pro_5b import StreamingVideoProcessor

    tag = args.tag or (os.path.basename(args.lora_path).replace(".safetensors", "") if args.lora_path != "none" else "base")
    out_root = os.path.join("outputs", "validate", tag)
    os.makedirs(out_root, exist_ok=True)

    processor = StreamingVideoProcessor(
        lora_path=args.lora_path, lora_alpha=args.lora_alpha, task="i2v",
        seed_multiplier=42, num_motion_latent=1, num_overlap_frame=5,
        cfg_scale=args.cfg, num_inference_steps=args.steps,
    )
    processor.num_clips = args.num_clips

    prompt_files = []

    if not args.skip_i2v:
        for i, (anchor, prompts) in enumerate(I2V_CASES):
            if not os.path.exists(anchor):
                print(f"skip missing anchor {anchor}")
                continue
            pf = os.path.join(out_root, f"i2v_{i}_prompts.txt")
            with open(pf, "w") as f:
                f.write("prompts = " + repr(prompts))
            prompt_files.append(pf)
            processor.task = "i2v"
            processor.generate_streaming_video(anchor, pf, out_root, sample_name=f"i2v_{i}")

    if not args.skip_t2v:
        processor.task = "t2v"
        for i, prompts in enumerate(T2V_CASES):
            pf = os.path.join(out_root, f"t2v_{i}_prompts.txt")
            with open(pf, "w") as f:
                f.write("prompts = " + repr(prompts))
            processor.generate_streaming_video(None, pf, out_root, sample_name=f"t2v_{i}")

    # Analysis pass (ffmpeg frame splitting + per-frame metrics)
    analyze = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_video.py")
    for f in sorted(os.listdir(out_root)):
        if f.endswith("_streaming_final.mp4"):
            video = os.path.join(out_root, f)
            print(f"\n=== analyzing {f} ===")
            subprocess.run([sys.executable, analyze, video, "--every", "24"])


if __name__ == "__main__":
    main()
