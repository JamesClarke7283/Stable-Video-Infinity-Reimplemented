# Unit tests for SVI ERFT components (CPU, mocked DiT). Run:
#   .venv/bin/python scripts/unit_test_svi.py
import torch
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.diffusion.error_replay import ErrorReplayBank
from diffsynth.diffusion.loss import SVIConfig, SVIErrorRecyclingLoss

PASS = []


def check(name, cond):
    PASS.append((name, bool(cond)))
    print(("PASS" if cond else "FAIL"), "-", name)


def make_bank(Z=3, pool=2):
    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(50, shift=5.0)
    return ErrorReplayBank(sched.timesteps, max_errors_per_grid=Z, spatial_pool=pool), sched


# ------------------------------------------------------------------ bank tests
def test_bank():
    bank, sched = make_bank()
    t_mid = sched.timesteps[25]
    err = torch.randn(1, 48, 6, 8, 16)
    for _ in range(5):
        bank.update("vid", t_mid, err)
    # capped at Z=3
    g = bank.grid_of(t_mid)
    check("bank capped at Z", len(bank.banks["vid"][g]) == 3)
    # pooled storage: 8x16 -> 4x8
    stored, orig_hw = bank.banks["vid"][g][0]
    check("bank pools spatially", tuple(stored.shape[-2:]) == (4, 8) and tuple(orig_hw) == (8, 16))
    # sample restores shape
    s = bank.sample("vid", t_mid, mode="current")
    check("sample unpools to orig shape", tuple(s.shape) == (1, 48, 6, 8, 16))
    # all-grids sampling works
    s2 = bank.sample("vid", t_mid, mode="all")
    check("sample mode=all", s2 is not None and tuple(s2.shape) == (1, 48, 6, 8, 16))
    # empty grid returns None
    t_other = sched.timesteps[40]
    check("empty grid -> None", bank.sample("noi", t_other, mode="current") is None)
    # cond slicing
    e = bank.sample_cond_error(2, t_mid)
    check("cond slice shape", tuple(e.shape) == (1, 48, 2, 8, 16))
    # grid mapping sanity: anchors are the 50-step schedule
    check("grid_of maps to nearest anchor", bank.grid_of(sched.timesteps[25]) == 25)


# ------------------------------------------------------------------ loss tests
class FakePipe:
    def __init__(self, record):
        self.device = "cpu"
        self.torch_dtype = torch.float32
        self.scheduler = FlowMatchScheduler("Wan")
        self.scheduler.set_timesteps(1000, training=True)
        self.dit = None
        self.record = record

    def model_fn(self, dit, latents, timestep, context, fuse_vae_embedding_in_latents,
                 num_cond_frames, use_gradient_checkpointing, use_gradient_checkpointing_offload):
        self.record["latents"] = latents.detach().clone()
        self.record["timestep"] = timestep.detach().clone()
        self.record["fuse"] = fuse_vae_embedding_in_latents
        self.record["num_cond_frames"] = num_cond_frames
        return torch.zeros_like(latents)  # v_hat = 0


def test_loss_i2v_chained():
    torch.manual_seed(0)
    record = {}
    pipe = FakePipe(record)
    bank, _ = make_bank(Z=10, pool=1)
    cfg = SVIConfig(p_vid=1.0, p_img=1.0, p_noi=1.0, p_clean=0.0,
                    mode_probs={"i2v_chained": 1.0}, warmup_iter=1, loss_mask_cond_frames=True)

    x0 = torch.randn(1, 48, 6, 4, 4)
    anchor = torch.randn(1, 48, 1, 4, 4)
    motion = torch.randn(1, 48, 1, 4, 4)
    ctx = torch.randn(1, 8, 16)

    # Pre-seed ALL grids with constant errors so injections are deterministic.
    # (The loss samples E_vid/E_noi from the current grid, E_img across grids;
    # banking happens after sampling, so injected errors are the constants.)
    e_const = torch.full((1, 48, 6, 4, 4), 0.5)
    n_const = torch.full((1, 48, 6, 4, 4), -0.25)
    sched50 = FlowMatchScheduler("Wan")
    sched50.set_timesteps(50, shift=5.0)
    for ts in sched50.timesteps:
        for _ in range(5):
            bank.update("vid", ts, e_const)
            bank.update("noi", ts, n_const)

    torch.manual_seed(42)  # fix noise draw
    loss = SVIErrorRecyclingLoss(
        pipe, bank, cfg, iteration=100,
        input_latents=x0, svi_anchor_latent=anchor, svi_motion_latent=motion,
        context=ctx,
    )
    # Replay the loss's torch RNG consumption: timestep_id randint, then noise.
    torch.manual_seed(42)
    tid_replay = torch.randint(0, 1000, (1,))
    noise = torch.randn_like(x0)

    tid = torch.argmin((pipe.scheduler.timesteps - record["timestep"]).abs())
    sigma = pipe.scheduler.sigmas[tid]
    check("timestep replay matches", torch.equal(tid_replay, tid.unsqueeze(0).reshape(1)))

    check("fuse flag set", record["fuse"] is True)
    check("num_cond_frames == 2", record["num_cond_frames"] == 2)

    # cond frames clamped to cond + E_img (constant 0.5 bank, sliced 2 frames)
    cond_expected = torch.cat([anchor, motion], dim=2) + 0.5
    check("cond clamp = cond + E_img",
          torch.allclose(record["latents"][:, :, :2], cond_expected, atol=1e-5))

    # generated frames: (1-sigma)(x0+E_vid) + sigma(noise+E_noi)
    gen_expected = (1 - sigma) * (x0[:, :, 2:] + 0.5) + sigma * (noise[:, :, 2:] - 0.25)
    check("generated frames = corrupted interpolation",
          torch.allclose(record["latents"][:, :, 2:], gen_expected, atol=1e-5))

    # target: V_rcy = (noise + E_noi) - x0 ; v_hat = 0 -> loss = mean(V_rcy^2) * w over generated frames
    v_rcy = (noise - 0.25) - x0
    w = pipe.scheduler.training_weight(record["timestep"])
    expected_loss = (v_rcy[:, :, 2:] ** 2).mean() * w
    check("loss = weighted MSE to V_rcy (masked)",
          torch.allclose(loss, expected_loss, rtol=1e-4, atol=1e-8))

    # banking: E_clean = sigma * V_rcy (cond frames zeroed), E_noise = -(1-sigma) * V_rcy
    # (stored as bf16 — compare through the same quantization)
    g = bank.grid_of(record["timestep"])
    e_clean_stored, _ = bank.banks["vid"][g][-1]
    e_clean_expected = sigma * v_rcy
    e_clean_expected[:, :, :2] = 0
    check("banked E_clean = sigma*(V_rcy - v_hat)",
          torch.equal(e_clean_stored, e_clean_expected.to(torch.bfloat16)))
    e_noise_stored, _ = bank.banks["noi"][g][-1]
    e_noise_expected = -(1 - sigma) * v_rcy
    e_noise_expected[:, :, :2] = 0
    check("banked E_noise = (1-sigma)*(v_hat - V_rcy)",
          torch.equal(e_noise_stored, e_noise_expected.to(torch.bfloat16)))


def test_loss_t2v_pure_and_clean():
    torch.manual_seed(1)
    record = {}
    pipe = FakePipe(record)
    bank, _ = make_bank(Z=10, pool=1)
    # T2V pure: no cond at all
    cfg = SVIConfig(p_vid=1.0, p_img=1.0, p_noi=1.0, p_clean=0.0,
                    mode_probs={"t2v_pure": 1.0}, warmup_iter=0, loss_mask_cond_frames=True)
    x0 = torch.randn(1, 48, 6, 4, 4)
    ctx = torch.randn(1, 8, 16)
    loss = SVIErrorRecyclingLoss(pipe, bank, cfg, iteration=10,
                                 input_latents=x0, context=ctx)
    check("t2v_pure: fuse flag off", record["fuse"] is False)
    check("t2v_pure: num_cond_frames == 0", record["num_cond_frames"] == 0)

    # clean override: all injections off
    record2 = {}
    pipe2 = FakePipe(record2)
    bank2, _ = make_bank(Z=10, pool=1)
    cfg2 = SVIConfig(p_vid=1.0, p_img=1.0, p_noi=1.0, p_clean=1.0,
                     mode_probs={"i2v_chained": 1.0}, warmup_iter=0)
    torch.manual_seed(7)
    anchor2 = torch.randn(1, 48, 1, 4, 4)
    motion2 = torch.randn(1, 48, 1, 4, 4)
    SVIErrorRecyclingLoss(pipe2, bank2, cfg2, iteration=10,
                          input_latents=x0, svi_anchor_latent=anchor2,
                          svi_motion_latent=motion2, context=ctx)
    torch.manual_seed(7)
    _a = torch.randn(1, 48, 1, 4, 4)
    _m = torch.randn(1, 48, 1, 4, 4)
    _tid_replay = torch.randint(0, 1000, (1,))
    noise = torch.randn_like(x0)
    tid = torch.argmin((pipe2.scheduler.timesteps - record2["timestep"]).abs())
    sigma = pipe2.scheduler.sigmas[tid]
    tid_mask = torch.zeros(6, dtype=torch.bool)
    tid_mask[2:] = True
    gen = (1 - sigma) * x0 + sigma * noise
    check("clean override: generated frames are plain flow-matching input",
          torch.allclose(record2["latents"][:, :, 2:], gen[:, :, 2:], atol=1e-5))


if __name__ == "__main__":
    test_bank()
    test_loss_i2v_chained()
    test_loss_t2v_pure_and_clean()
    failed = [n for n, ok in PASS if not ok]
    print(f"\n{len(PASS) - len(failed)}/{len(PASS)} passed")
    if failed:
        raise SystemExit(1)
    print("UNIT_OK")
