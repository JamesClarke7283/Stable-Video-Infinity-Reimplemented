# Project Conventions

## Python environment
- Use **`uv`** for all Python virtual environments and Python version management (`uv venv`, `uv pip`, `uv python`). Do not use `python -m venv`, `virtualenv`, or `conda`.
- Keep venvs inside the project (e.g. `.venv/`) or under `/tmp` for throwaway tooling.

## Git
- Make **regular commits** as work progresses — commit each meaningful, self-contained slice (setup, dataset tooling, training code, inference code, docs) rather than one giant commit at the end.
- Write clear, conventional commit messages.

## Reference material
- `SVI-Pro-Technical-Report.pdf` (repo root): the SVI paper (arXiv 2510.09212) — error-recycling fine-tuning.
- Reference upstream repo clone: `/tmp/Stable-Video-Infinity` (branch `main` = SVI 1.0/Wan2.1; branch `svi_wan22` = SVI 2.0 Pro/Wan2.2, worktree at `/tmp/svi-wan22`).
- `PLAN.md` (repo root): the high-level implementation plan for the Wan2.2-5B SVI-Pro LoRA — keep it updated as implementation progresses.
