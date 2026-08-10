# SPDX-License-Identifier: Apache-2.0
# Error replay memory for SVI Error-Recycling Fine-Tuning.
# Algorithm from "Stable Video Infinity: Infinite-Length Video Generation with
# Error Recycling" (arXiv 2510.09212, Copyright VITA @ EPFL, Apache License 2.0),
# as implemented in Stable-Video-Infinity (main branch, train_svi.py).
# Re-implemented for the DiffSynth-Studio 2.0 framework by the
# Stable-Video-Infinity-Reimplemented authors.
#
# Design (paper Sec. 4.3):
#   - Two banks ("vid": clean-endpoint residuals, "noi": noise-endpoint residuals).
#   - The training timestep is discretized to the nearest of `num_grids` anchor
#     timesteps taken from the inference schedule (50 steps, shift=5).
#   - Each grid stores at most Z errors; when full, the most L2-similar stored
#     error is replaced (keeps error diversity).
#   - Sampling: "current" = uniform inside the current timestep grid (E_vid,
#     E_noi); "all" = uniform across grids (E_img, simulating errors accumulated
#     over the whole trajectory).
#
# Memory adaptation for single-workstation use: at 720P x121 frames one raw
# error is ~10.5MB (48 x 31 x 44 x 80 bf16), so 50 grids x Z=500 would need
# ~260GB RAM. Errors are therefore avg-pooled `spatial_pool`-fold on H/W before
# banking and bilinearly upsampled on injection (model-induced errors are smooth,
# blur/color-shift-like - paper Fig. 6). With pool=2 and Z=200 the bank needs
# ~26GB RAM. Both knobs are configurable.
import torch
import torch.nn.functional as F


class ErrorReplayBank:
    def __init__(
        self,
        anchor_timesteps: torch.Tensor,
        max_errors_per_grid: int = 200,
        spatial_pool: int = 2,
        store_dtype: torch.dtype = torch.bfloat16,
    ):
        self.anchors = anchor_timesteps.detach().float().cpu().flatten()
        self.num_grids = int(self.anchors.numel())
        self.max_errors_per_grid = max_errors_per_grid
        self.spatial_pool = max(1, int(spatial_pool))
        self.store_dtype = store_dtype
        self.banks = {"vid": {}, "noi": {}}  # name -> grid_idx -> list of (tensor, orig_hw)

    # ------------------------------------------------------------------ grids
    def grid_of(self, timestep) -> int:
        """Map a timestep (scalar tensor or float, 0..1000 scale) to the nearest
        anchor grid index."""
        if isinstance(timestep, torch.Tensor):
            timestep = float(timestep.detach().float().cpu().item())
        return int(torch.argmin((self.anchors - timestep).abs()).item())

    # ----------------------------------------------------------------- update
    def _pool(self, error: torch.Tensor):
        orig_hw = error.shape[-2:]
        if self.spatial_pool > 1:
            error = F.avg_pool3d(error.float(), kernel_size=(1, self.spatial_pool, self.spatial_pool))
        return error.to(self.store_dtype), orig_hw

    def update(self, bank: str, timestep, error: torch.Tensor):
        grid = self.grid_of(timestep)
        entry, orig_hw = self._pool(error.detach().float().cpu())
        entries = self.banks[bank].setdefault(grid, [])
        if len(entries) < self.max_errors_per_grid:
            entries.append((entry, orig_hw))
            return
        # Bank full: replace the most L2-similar stored error (diversity keeping).
        new_flat = entry.flatten().float()
        dists = torch.stack([torch.linalg.vector_norm(e.flatten().float() - new_flat) for e, _ in entries])
        entries[int(torch.argmin(dists).item())] = (entry, orig_hw)

    # ----------------------------------------------------------------- sample
    def _unpool(self, entry: torch.Tensor, orig_hw):
        out = entry.float()
        if self.spatial_pool > 1 and tuple(out.shape[-2:]) != tuple(orig_hw):
            out = F.interpolate(out, size=(out.shape[2], *orig_hw), mode="trilinear", align_corners=False)
        return out

    def _sample_entry(self, bank: str, grid: int):
        entries = self.banks[bank].get(grid)
        if not entries:
            return None
        entry, orig_hw = entries[int(torch.randint(len(entries), (1,)).item())]
        return self._unpool(entry, orig_hw)

    def sample(self, bank: str, timestep, mode: str = "current", device=None, dtype=None):
        """Sample one error. mode="current": from the timestep-aligned grid;
        mode="all": from a uniformly chosen non-empty grid (Unif over the whole
        schedule, for E_img). Returns None if the bank is empty there."""
        if mode == "current":
            grids = [self.grid_of(timestep)]
        else:
            grids = [g for g, lst in self.banks[bank].items() if len(lst) > 0]
        if len(grids) == 0:
            return None
        grid = grids[int(torch.randint(len(grids), (1,)).item())]
        out = self._sample_entry(bank, grid)
        if out is None:
            return None
        if device is not None:
            out = out.to(device)
        if dtype is not None:
            out = out.to(dtype)
        return out

    def sample_cond_error(self, num_cond_frames: int, timestep, device=None, dtype=None):
        """E_img: sample from the vid bank across all grids, then slice a random
        temporal window of `num_cond_frames` (paper Eq. 5, Unif_T)."""
        err = self.sample("vid", timestep, mode="all", device=device, dtype=dtype)
        if err is None:
            return None
        total = err.shape[2]
        if total > num_cond_frames:
            start = int(torch.randint(0, total - num_cond_frames + 1, (1,)).item())
        else:
            start = 0
        return err[:, :, start:start + num_cond_frames]

    # ------------------------------------------------------------------ stats
    def occupancy(self):
        return {
            bank: {"grids": len(grids), "total": sum(len(v) for v in grids.values())}
            for bank, grids in self.banks.items()
        }
