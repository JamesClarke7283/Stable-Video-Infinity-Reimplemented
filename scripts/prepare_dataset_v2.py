# SVI dataset builder v2: fetch ORIGINAL full-length MixKit videos from the
# MixKit CDN (assets.mixkit.co/videos/<id>/<id>-<res>.mp4), using the source
# (slug, id) pairs recovered from the Open-Sora-Plan/FastVideo filename lists.
# The OSP/FastVideo mp4s themselves are 2.7-6s scene-cut clips - too short for
# SVI chained training (needs >= 242 frames @24fps); the originals are 10-60s+.
#
# Resolution note: Wan2.2-TI2V-5B maxes out at 720P (1280x704) and the training
# loader resizes/crops every frame to 1280x704 anyway, so the 720p CDN variant
# (1280x720) is the right source - 1080p would be downscaled with zero benefit
# at 2-3x the bandwidth/disk cost.
#
# Streaming design: each file is downloaded -> probed -> normalized -> deleted,
# so raw videos never accumulate on disk.
# Run: .venv/bin/python scripts/prepare_dataset_v2.py
import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

FFMPEG = "/tmp/ffmpeg-static/ffmpeg"
FFPROBE = "/tmp/ffmpeg-static/ffprobe"


def probe(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip().split(",")
    # ffprobe emits: width,height,r_frame_rate,nb_frames (observed order)
    try:
        w, h = int(out[0]), int(out[1])
        fps = eval(out[2]) if "/" in out[2] else float(out[2])
        frames = int(out[3])
        return w, h, fps, frames, frames / fps
    except Exception:
        return None


def fetch_probe_normalize(job):
    vid, slug, out_dir, min_dur = job
    base = f"{slug}-{vid}.mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp")
    tmp.close()
    try:
        url = f"https://assets.mixkit.co/videos/{vid}/{vid}-720.mp4"
        r = subprocess.run(["curl", "-sL", "--fail", "-o", tmp.name, url], capture_output=True)
        if r.returncode != 0 or os.path.getsize(tmp.name) <= 100000:
            return None
        info = probe(tmp.name)
        if info is None:
            return None
        w, h, fps, frames, dur = info
        # 720p landscape originals only - no 360p fallback, no portrait.
        if (w, h) != (1280, 720):
            return (base, w, h, fps, dur, 0)
        if dur < min_dur:
            return (base, w, h, fps, dur, 0)
        out_path = os.path.join(out_dir, "videos", base)
        r = subprocess.run([FFMPEG, "-y", "-i", tmp.name, "-r", "24", "-an",
                            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                            "-movflags", "+faststart", out_path], capture_output=True)
        if r.returncode != 0:
            return (base, w, h, fps, dur, 0)
        return (base, w, h, fps, dur, 1)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="/tmp/mixkit_sources.txt")
    parser.add_argument("--captions_json", default="/tmp/llava_cap.json")
    parser.add_argument("--out_dir", default="data/svi_mixkit")
    parser.add_argument("--max_videos", type=int, default=1500)
    parser.add_argument("--min_duration", type=float, default=5.1)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "videos"), exist_ok=True)

    sources = [tuple(line.strip().split("\t")) for line in open(args.sources)]
    print(f"{len(sources)} unique sources", flush=True)

    caps = {}
    data = json.load(open(args.captions_json))
    for item in data:
        p = os.path.basename(item.get("path", ""))
        cap = item.get("cap")
        cap = cap[0] if isinstance(cap, list) and cap else cap
        m = re.match(r"(mixkit-.+-\d+)_\d+\.mp4$", p)
        if m and isinstance(cap, str):
            caps.setdefault(m.group(1), cap.strip())
    print(f"{len(caps)} caption keys", flush=True)

    rows, stats = [], []
    kept = 0
    jobs = [(vid, slug, args.out_dir, args.min_duration) for vid, slug in sources]
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for i, res in enumerate(ex.map(fetch_probe_normalize, jobs)):
            if res is None:
                continue
            base, w, h, fps, dur, ok = res
            stats.append((base, w, h, fps, f"{dur:.2f}", ok))
            if ok and kept < args.max_videos:
                caption = caps.get(base[:-4]) or base[:-4].replace("-", " ")
                rows.append((f"videos/{base}", caption))
                kept += 1
            if (i + 1) % 100 == 0:
                print(f"progress {i+1}/{len(jobs)} kept={kept}", flush=True)
            if kept >= args.max_videos:
                break

    with open(os.path.join(args.out_dir, "metadata.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["video", "prompt"])
        wcsv.writerows(rows)
    with open(os.path.join(args.out_dir, "probe_stats.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["file", "width", "height", "fps", "duration", "kept"])
        wcsv.writerows(stats)
    chained = sum(1 for s in stats if s[5] == 1 and float(s[4]) >= 10.1)
    print(f"DONE: kept={kept} chained-capable={chained}")


if __name__ == "__main__":
    main()
