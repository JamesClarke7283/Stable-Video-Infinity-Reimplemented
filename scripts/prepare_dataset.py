# Dataset preparation for SVI-Pro 5B training (milestone 8.4).
# Streams the Open-Sora-Plan mixkit.tar.gz, probes each video, keeps long-enough
# ones, normalizes to 24fps, and writes metadata.csv (video,prompt) with
# LLaVA captions when available (llava_path_cap_64x512x512.json) or de-slugged
# filename captions otherwise.
# Run: .venv/bin/python scripts/prepare_dataset.py [--max_videos 2500]
import argparse
import csv
import json
import os
import re
import subprocess
import tarfile
import tempfile

FFMPEG = "/tmp/ffmpeg-static/ffmpeg"
FFPROBE = "/tmp/ffmpeg-static/ffprobe"


def probe(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,r_frame_rate,duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    # returns e.g. "30/1,180,6.000000" (order follows requested entries)
    parts = out.split(",")
    try:
        fps = eval(parts[0]) if "/" in parts[0] else float(parts[0])
        duration = float(parts[-1])
        return fps, duration
    except Exception:
        return None, None


def slug_to_caption(fn):
    s = os.path.basename(fn)[:-4] if fn.endswith(".mp4") else os.path.basename(fn)
    s = re.sub(r"^mixkit-", "", s)
    s = re.sub(r"-\d+$", "", s)
    s = s.replace("-", " ")
    return s.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_path", default="/tmp/mixkit.tar.gz")
    parser.add_argument("--captions_json", default="/tmp/llava_cap.json")
    parser.add_argument("--out_dir", default="data/svi_mixkit")
    parser.add_argument("--max_videos", type=int, default=2500)
    parser.add_argument("--min_duration", type=float, default=5.1, help=">=121 frames at 24fps")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "videos"), exist_ok=True)

    # LLaVA captions: map basename -> caption
    caps = {}
    if os.path.exists(args.captions_json):
        data = json.load(open(args.captions_json))
        for item in data:
            p = item.get("path", "")
            cap = item.get("cap")
            if isinstance(cap, list) and len(cap) > 0:
                cap = cap[0]
            if p and isinstance(cap, str):
                caps[os.path.basename(p)] = cap.strip()
        print(f"loaded {len(caps)} captions")

    rows, kept, seen, chained_capable = [], 0, 0, 0
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp")
    stats_path = os.path.join(args.out_dir, "probe_stats.csv")
    with open(stats_path, "w", newline="") as stats_f:
        stats_w = csv.writer(stats_f)
        stats_w.writerow(["file", "fps", "duration", "kept"])
        with tarfile.open(args.tar_path, "r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(".mp4"):
                    continue
                if kept >= args.max_videos:
                    break
                seen += 1
                base = os.path.basename(member.name)
                f = tar.extractfile(member)
                with open(tmp.name, "wb") as out:
                    while True:
                        chunk = f.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                fps, duration = probe(tmp.name)
                keep = duration is not None and duration >= args.min_duration
                stats_w.writerow([base, fps, duration, int(keep)])
                if keep:
                    out_name = base if base.endswith(".mp4") else base + ".mp4"
                    out_path = os.path.join(args.out_dir, "videos", out_name)
                    # Normalize to 24fps, drop audio.
                    r = subprocess.run([FFMPEG, "-y", "-i", tmp.name, "-r", "24", "-an",
                                        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                                        "-movflags", "+faststart", out_path],
                                       capture_output=True)
                    if r.returncode != 0:
                        print("ffmpeg failed for", base)
                        continue
                    caption = caps.get(base) or caps.get(base.replace(".mp4", "")) or slug_to_caption(base)
                    rows.append((f"videos/{out_name}", caption))
                    kept += 1
                    if duration >= 10.1:
                        chained_capable += 1
                    if kept % 100 == 0:
                        print(f"kept {kept}/{seen} (chained-capable {chained_capable})")
    os.unlink(tmp.name)

    with open(os.path.join(args.out_dir, "metadata.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "prompt"])
        w.writerows(rows)
    print(f"DONE: seen={seen} kept={kept} chained-capable(>=10.1s)={chained_capable}")


if __name__ == "__main__":
    main()
