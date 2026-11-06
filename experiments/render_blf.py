"""Render the BLF anti-swing run (experiments/e31_blf_antiswing.py) to an MP4 with MuJoCo's offscreen renderer.

    MUJOCO_GL=egl python -u experiments/render_blf.py --P 0.05 --T 25 --out blf.mp4      (GPU runtime)
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -u experiments/render_blf.py ...      (CPU runtime, slower)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import imageio
import mujoco
import numpy as np
from tqdm.auto import tqdm

from e31_blf_antiswing import metrics_of, run_blf
from sim.config import load_config
from sim.state import get_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=float, default=0.05, help="barrier scale (0 = unconstrained Lyapunov law)")
    ap.add_argument("--delta", type=float, nargs=3, default=[20.0, 20.0, 5.0])
    ap.add_argument("--T", type=float, default=25.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--att-w", type=float, default=30.0)
    ap.add_argument("--z-coupling", default="drop", choices=["drop", "paper"])
    ap.add_argument("--ramp", type=float, default=3.0, help="setpoint ramp [s]; 0 = raw step")
    ap.add_argument("--out", default="blf_antiswing.mp4")
    args = ap.parse_args()

    cfg = load_config()
    print("Creating renderer...", flush=True)
    state = {"renderer": None, "writer": imageio.get_writer(args.out, fps=args.fps), "next": 0.0, "n": 0}
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 7.0, 135, -20
    n_ticks = int(round(args.T * 500))
    bar = tqdm(total=n_ticks, desc="simulating + rendering", mininterval=1.0)

    def frame_fn(sim):
        bar.update(1)
        if state["renderer"] is None:
            state["renderer"] = mujoco.Renderer(sim.model, height=args.height, width=args.width)
        if sim.data.time >= state["next"]:
            cam.lookat[:] = get_state(sim.model, sim.data).x_Q
            state["renderer"].update_scene(sim.data, camera=cam)
            state["writer"].append_data(state["renderer"].render())
            state["n"] += 1
            state["next"] += 1.0 / args.fps

    r = run_blf(cfg, args.P, args.delta, args.T, args.att_w, args.z_coupling, frame_fn=frame_fn, ramp=args.ramp)
    bar.close()
    state["writer"].close()
    print("frames:", state["n"], "->", args.out)
    for k, v in metrics_of(r, args.delta).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
