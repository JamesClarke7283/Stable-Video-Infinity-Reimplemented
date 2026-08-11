# Video error/drift analysis for SVI validation (per project instructions):
# uses FFMPEG to probe + split a video into frames, computes per-frame metrics
# (luminance, color means, Laplacian blur score, PSNR vs reference frame and vs
# previous frame) and writes a CSV + a grid of sample frames for one-by-one
# visual inspection.
# Run: .venv/bin/python scripts/analyze_video.py video.mp4 --out_dir out --every 12
import argparse
import json
import os
import subprocess

import numpy as np

FFMPEG = "/tmp/ffmpeg-static/ffmpeg"
FFPROBE = "/tmp/ffmpeg-static/ffprobe"


def ffprobe_info(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
         "-of", "json", path],
        capture_output=True, text=True,
    ).stdout
    return json.loads(out)["streams"][0]


def extract_frames(video, out_dir, every):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        [FFMPEG, "-y", "-i", video, "-vf", f"select='not(mod(n\\,{every}))'", "-vsync", "vfr",
         os.path.join(out_dir, "f%05d.png")],
        check=True, capture_output=True,
    )
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png"))


def laplacian_var(gray):
    lap = (-4 * gray
           + np.roll(gray, 1, axis=0) + np.roll(gray, -1, axis=0)
           + np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1))
    return float(lap[1:-1, 1:-1].var())


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--every", type=int, default=12, help="analyze every Nth frame")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.splitext(args.video)[0] + "_analysis"
    info = ffprobe_info(args.video)
    print(f"video: {info['width']}x{info['height']} {info['nb_frames']} frames @ {info['r_frame_rate']} ({info.get('duration','?')}s)")

    frames = extract_frames(args.video, out_dir, args.every)
    print(f"extracted {len(frames)} frames to {out_dir}")

    import imageio.v2 as imageio
    rows = [("frame_file", "luma", "r_mean", "g_mean", "b_mean", "blur_lapvar", "psnr_vs_first", "psnr_vs_prev")]
    ref = None
    prev = None
    for fp in frames:
        img = imageio.imread(fp).astype(np.float32)
        gray = img.mean(axis=2)
        luma = float(gray.mean())
        r, g, b = (float(img[:, :, i].mean()) for i in range(3))
        blur = laplacian_var(gray)
        p_first = psnr(img, ref) if ref is not None else 99.0
        p_prev = psnr(img, prev) if prev is not None else 99.0
        if ref is None:
            ref = img.copy()
        prev = img
        rows.append((os.path.basename(fp), f"{luma:.2f}", f"{r:.2f}", f"{g:.2f}", f"{b:.2f}",
                     f"{blur:.1f}", f"{p_first:.2f}", f"{p_prev:.2f}"))

    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
    print(f"metrics written to {csv_path}")
    # Print a compact summary for quick spotting of drift/degradation
    lumas = [float(r[1]) for r in rows[1:]]
    blurs = [float(r[5]) for r in rows[1:]]
    p_firsts = [float(r[6]) for r in rows[1:]]
    print(f"luma: first={lumas[0]:.1f} mid={lumas[len(lumas)//2]:.1f} last={lumas[-1]:.1f}")
    print(f"blur: first={blurs[0]:.0f} mid={blurs[len(blurs)//2]:.0f} last={blurs[-1]:.0f}")
    print(f"psnr_vs_first: mid={p_firsts[len(p_firsts)//2]:.1f} last={p_firsts[-1]:.1f}")


if __name__ == "__main__":
    main()
