"""Step 3.1: Ding-Sun-He barrier-Lyapunov anti-swing controller (Sensors 2023) on the Skydio X2 + C1/P1 MuJoCo model.

Rest-to-rest move of the drone (default (20, 20, 5) m, the paper's scale) with the paper's gains, comparing
    BLF        : Theorem 2, P = 0.05, swing bound kappa = 15 deg
    Lyapunov   : Theorem 1 (P = 0), no swing constraint
Real X2 rotor limits (0-13 N) are active; the repo's mixer saturates with PX4 priorities.

    python experiments/e31_blf_antiswing.py [--delta 20 20 5] [--T 25] [--att-w 30] [--z-coupling drop|paper] [--ramp 3.0]

Gate (all_passed): finite run, final position error < 0.2 m, BLF peak swing <= kappa (+0.5 deg) and <= unconstrained peak.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.checks_blf import _trapz, settling_time
from sim.checks_flatness import params
from sim.config import load_config
from sim.controllers.blf_antiswing import AttitudeGains, BLFAntiSwing, BLFGains, QSLTSParams, quintic_ramp
from sim.run_io import RunRecorder
from sim.simulator import Simulator
from sim.state import get_state


def run_blf(cfg, P, delta, T, att_w=30.0, z_coupling="drop", kappa_deg=15.0, frame_fn=None, ramp=3.0):
    """Closed-loop run. frame_fn(sim) is called every control tick (used by the video script). ramp > 0 moves the
    setpoint along a quintic ramp of that duration [s] (D5); ramp = 0 is the paper's raw step."""
    sim = Simulator(cfg, 500)
    p = QSLTSParams.from_flat(params(cfg))
    s0 = get_state(sim.model, sim.data)
    target = s0.x_Q + np.asarray(delta, float)
    gains = BLFGains(P=P, kappa_deg=kappa_deg)
    ctrl = BLFAntiSwing(p, gains, AttitudeGains.from_bandwidth(p.J, att_w), target_xyz=target, z_coupling=z_coupling)
    n = int(round(T / sim.control_dt))
    keys = ("t", "x_Q", "x_L", "ab", "F", "wrench", "sat", "V1", "sigma2", "Phi", "Phi_d")
    log = {k: [] for k in keys}
    for _ in range(n):
        s = get_state(sim.model, sim.data)
        if ramp > 0:
            ctrl.target = s0.x_Q + (target - s0.x_Q) * quintic_ramp(s.t / ramp)
        w = ctrl.step(s, sim.control_dt)
        sat = float(not np.allclose(sim.mixer(w), sim.mixer.unconstrained(w), atol=1e-9))
        for k, v in (("t", s.t), ("x_Q", s.x_Q), ("x_L", s.x_L), ("ab", ctrl.last["ab"]), ("F", ctrl.last["F"]),
                     ("wrench", w), ("sat", sat), ("V1", ctrl.last["V1"]), ("sigma2", ctrl.last["sigma2"]),
                     ("Phi", ctrl.last["Phi"]), ("Phi_d", ctrl.last["Phi_d"])):
            log[k].append(np.array(v, dtype=float))
        sim.step_ctbm(w)
        if frame_fn is not None:
            frame_fn(sim)
        if not np.all(np.isfinite(sim.data.qpos)):
            break
    out = {k: np.array(v) for k, v in log.items()}
    out["target"] = target
    out["x0"] = s0.x_Q
    out["p"] = p
    return out


def metrics_of(r, delta):
    t, x = r["t"], r["x_Q"]
    m = {}
    for i, name in enumerate("xyz"):
        band = 0.02 * max(abs(delta[i]), 1e-9)
        m[f"t_s{name}_s"] = settling_time(t, x[:, i], r["target"][i], band)
    m["final_err_m"] = [float(v) for v in (x[-1] - r["target"])]
    m["swing_max_alpha_deg"] = float(np.degrees(np.abs(r["ab"][:, 0]).max()))
    m["swing_max_beta_deg"] = float(np.degrees(np.abs(r["ab"][:, 1]).max()))
    m["tilt_max_deg"] = float(np.degrees(np.abs(r["Phi"][:, :2]).max()))
    m["E_u"] = float(_trapz(r["F"] ** 2, t))      # paper: int (Ux^2 + Uy^2 + (Uz + (Mq+Ml) g)^2) dt = int F^2 dt
    m["F_max_N"] = float(r["F"].max())
    m["saturated_fraction"] = float(r["sat"].mean())
    m["finite"] = bool(np.all(np.isfinite(x)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, nargs=3, default=[20.0, 20.0, 5.0])
    ap.add_argument("--T", type=float, default=25.0)
    ap.add_argument("--att-w", type=float, default=30.0)
    ap.add_argument("--z-coupling", default="drop", choices=["drop", "paper"])
    ap.add_argument("--kappa", type=float, default=15.0)
    ap.add_argument("--ramp", type=float, default=3.0, help="setpoint ramp duration [s]; 0 = raw step (paper), diverges for >~16 m on the X2")
    args = ap.parse_args()

    cfg = load_config()
    run = RunRecorder("phase3", "e31_blf_antiswing", cfg,
                      description=f"Step 3.1: Ding-Sun-He (2023) BLF anti-swing controller, X2 + C1/P1, L0, move {args.delta} m, "
                                  f"att bandwidth {args.att_w} rad/s, z_coupling={args.z_coupling}, kappa={args.kappa} deg, ramp={args.ramp} s.")
    results, metrics, data = {}, {}, {}
    for name, P in (("BLF", 0.05), ("Lyapunov", 0.0)):
        r = run_blf(cfg, P, args.delta, args.T, args.att_w, args.z_coupling, args.kappa, ramp=args.ramp)
        results[name] = r
        for k, v in metrics_of(r, args.delta).items():
            metrics[f"{name}.{k}"] = v
        for k in ("t", "x_Q", "x_L", "ab", "F", "wrench", "sat"):
            data[f"{name}_{k}"] = r[k]
    b, l = metrics["BLF.swing_max_alpha_deg"], metrics["Lyapunov.swing_max_alpha_deg"]
    bb, lb = metrics["BLF.swing_max_beta_deg"], metrics["Lyapunov.swing_max_beta_deg"]
    err_ok = all(np.linalg.norm(metrics[f"{n}.final_err_m"]) < 0.2 for n in results)
    metrics["all_passed"] = bool(err_ok and all(metrics[f"{n}.finite"] for n in results)
                                 and max(b, bb) <= args.kappa + 0.5 and max(b, bb) <= max(l, lb) + 1e-6)
    run.save_metrics(metrics)
    run.save_data(**data)

    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    col = {"BLF": "tab:red", "Lyapunov": "tab:blue"}
    for name, r in results.items():
        for i, c in enumerate("xyz"):
            ax[0, i].plot(r["t"], r["x_Q"][:, i], color=col[name], label=name)
            ax[0, i].axhline(r["target"][i], color="k", ls="--", lw=0.8)
            ax[0, i].set(title=f"drone {c} [m]", xlabel="t [s]")
        ax[1, 0].plot(r["t"], np.degrees(r["ab"][:, 0]), color=col[name], label=name)
        ax[1, 1].plot(r["t"], np.degrees(r["ab"][:, 1]), color=col[name], label=name)
        ax[1, 2].plot(r["t"], r["F"], color=col[name], label=name)
    for a, nm in ((ax[1, 0], "alpha"), (ax[1, 1], "beta")):
        a.axhline(args.kappa, color="tab:green", ls="-.", lw=1)
        a.axhline(-args.kappa, color="tab:green", ls="-.", lw=1)
        a.set(title=f"payload swing {nm} [deg]  (green: +-kappa)", xlabel="t [s]")
    ax[1, 2].axhline(4 * 13.0, color="gray", ls="--", lw=0.8)
    ax[1, 2].set(title="collective thrust F [N] (dashed: 4 x 13 N)", xlabel="t [s]")
    ax[0, 0].legend()
    fig.tight_layout()
    run.save_figure(fig, "blf_antiswing")

    out = run.finish(notes="Controller: sim/controllers/blf_antiswing.py. The numpy-only validation (paper replication, Table 2) "
                           "is in tests/test_blf_antiswing.py. Deviations D1-D4 are listed in that module's docstring.")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(out)


if __name__ == "__main__":
    main()
