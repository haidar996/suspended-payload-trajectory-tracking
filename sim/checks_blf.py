"""Checks for the Ding-Sun-He anti-swing controller (sim/controllers/blf_antiswing.py). No MuJoCo needed here.

Two numpy plants:
  * paper_plant    : the paper's own model, Eq. (1) (translation + swing, Lagrange) and Eq. (2) (Euler-angle attitude).
                     Used to replicate the paper's simulation (Sec. 4, Table 2).
  * rigid3d_plant  : full 3D rigid-body drone (SO(3) attitude, body torques, rotor-free wrench input) + spherical-pendulum
                     payload, attached at the drone CoM (same physics as the repo's C1 + P1 MuJoCo model, Paper 1 Eqs. 6-8).
                     Used to test the repo-facing adapter (State -> wrench) without MuJoCo.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .controllers.blf_antiswing import (AttitudeGains, BLFAntiSwing, BLFGains, QSLTSParams, desired_attitude,
                                        euler_from_R, mass_and_bias, quintic_ramp)

PAPER_PARAMS = QSLTSParams(M_q=1.0, M_l=0.2, l=1.0, J=np.diag([0.009, 0.009, 0.015]), g=9.8)   # Sec. 4
PAPER_TARGET = (20.0, 20.0, 20.0)
# Skydio X2 + C1/P1 of configs/base.yaml (mass/inertia from sim/checks_x2.py EXPECTED_*)
X2_PARAMS = QSLTSParams(M_q=1.325, M_l=0.2, l=0.5,
                        J=np.array([[0.03665, 0.0, -0.0021], [0.0, 0.02541, 0.0], [-0.0021, 0.0, 0.06053]]), g=9.81)
X2_LIMITS = dict(F_max=4 * 13.0, tau_xy_max=0.18 * 2 * 13.0, tau_z_max=0.0201 * 2 * 13.0)   # rough, from rotor geometry
PAPER_TABLE2 = {   # Table 2 of the paper
    "BLF": dict(t_sx=15.01, t_sy=14.96, t_sz=3.37, t_sa=5.66, t_sb=5.69, a_max=9.57, b_max=9.76, E=3203),
    "Lyapunov": dict(t_sx=13.96, t_sy=13.69, t_sz=3.37, t_sa=5.95, t_sb=5.96, a_max=17.52, b_max=19.11, E=3215),
    "LQR": dict(t_sx=4.44, t_sy=4.44, t_sz=4.42, t_sa=5.52, t_sb=5.67, a_max=70.65, b_max=45.67, E=3266),
}


_trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy 1.x / 2.x


def rk4(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# --------------------------------------------------------------------------------------------- paper plant
def paper_plant_rhs(x, F, tau, p: QSLTSParams):
    """x = [xi(3), ab(2), xi_dot(3), ab_dot(2), Phi(3), Phi_dot(3)]; inputs: thrust F, torque tau (paper Eq. 2)."""
    pos_vel = x[:10]
    Phi, Phid = x[10:13], x[13:16]
    ph, th, ps = Phi
    U = np.array([F * (np.sin(th) * np.cos(ph) * np.cos(ps) + np.sin(ph) * np.sin(ps)),
                  F * (np.sin(th) * np.cos(ph) * np.sin(ps) - np.sin(ph) * np.cos(ps)),
                  F * np.cos(th) * np.cos(ph) - (p.M_q + p.M_l) * p.g])
    M, E, h = mass_and_bias(p, x[3:5], x[8:10])
    acc = np.linalg.solve(M, E @ U + h)
    Ix, Iy, Iz = np.diag(p.J)
    Phidd = np.array([((Iy - Iz) * Phid[1] * Phid[2] + tau[0]) / Ix,
                      ((Iz - Ix) * Phid[0] * Phid[2] + tau[1]) / Iy,
                      ((Ix - Iy) * Phid[0] * Phid[1] + tau[2]) / Iz])
    return np.concatenate([pos_vel[5:10], acc, Phid, Phidd])


def run_paper_replication(P: float = 0.05, T: float = 20.0, dt: float = 1e-3, params: QSLTSParams = PAPER_PARAMS,
                          target=PAPER_TARGET, gains: BLFGains | None = None, att_w: float = 30.0,
                          z_coupling: str = "drop") -> dict:
    gains = gains or BLFGains(P=P)
    if gains.P != P:
        gains.P = P
    att = AttitudeGains.from_bandwidth(params.J, w=att_w)
    ctrl = BLFAntiSwing(params, gains, att, target_xyz=target, z_coupling=z_coupling)
    x = np.zeros(16)
    n = int(round(T / dt))
    log = {k: [] for k in ("t", "xi", "ab", "U", "F", "V1", "q1", "Phi", "Phi_d", "sigma2")}
    for i in range(n):
        xi, ab, xid, abd = x[0:3], x[3:5], x[5:8], x[8:10]
        U = ctrl.position_forces(xi[:2], xid[:2], xi[2], xid[2], ab, abd, dt)
        F, Phi_d = desired_attitude(params, U, 0.0)
        # paper's Euler-angle plant (Eq. 2): gyroscopic term W cancelled with the sign consistent with Eq. (2)
        Phi, Phid = x[10:13], x[13:16]
        Ix, Iy, Iz = np.diag(params.J)
        W = np.array([(Iy - Iz) * Phid[1] * Phid[2], (Iz - Ix) * Phid[0] * Phid[2], (Ix - Iy) * Phid[0] * Phid[1]])
        tau = -att.Kp @ (Phi - Phi_d) - att.Kv @ Phid - W
        log["t"].append(i * dt); log["xi"].append(xi.copy()); log["ab"].append(ab.copy()); log["U"].append(U.copy())
        log["F"].append(F); log["V1"].append(ctrl.last["V1"]); log["q1"].append(ctrl.last["q1"].copy())
        log["Phi"].append(Phi.copy()); log["Phi_d"].append(Phi_d.copy()); log["sigma2"].append(ctrl.last["sigma2"])
        x = rk4(lambda s: paper_plant_rhs(s, F, tau, params), x, dt)
    out = {k: np.array(v) for k, v in log.items()}
    out["params"] = params
    out["target"] = np.asarray(target, float)
    return out


def settling_time(t, y, y_final, band):
    """First time after which |y - y_final| stays within `band` until the end (NaN if never)."""
    out = np.where(np.abs(y - y_final) > band)[0]
    if len(out) == 0:
        return float(t[0])
    return float(t[out[-1] + 1]) if out[-1] + 1 < len(t) else float("nan")


def paper_metrics(res: dict, band_frac: float = 0.02, ang_band_deg: float = 1.0) -> dict:
    t, xi, ab, U, p = res["t"], res["xi"], res["ab"], res["U"], res["params"]
    tgt = res["target"]
    m = {}
    for i, name in enumerate("xyz"):
        m[f"t_s{name}"] = settling_time(t, xi[:, i], tgt[i], band_frac * abs(tgt[i]))
    m["t_sa"] = settling_time(t, np.degrees(ab[:, 0]), 0.0, ang_band_deg)
    m["t_sb"] = settling_time(t, np.degrees(ab[:, 1]), 0.0, ang_band_deg)
    m["a_max"] = float(np.degrees(np.abs(ab[:, 0]).max()))
    m["b_max"] = float(np.degrees(np.abs(ab[:, 1]).max()))
    Fv2 = U[:, 0] ** 2 + U[:, 1] ** 2 + (U[:, 2] + (p.M_q + p.M_l) * p.g) ** 2   # paper's E_u = int(Ux^2+Uy^2+(Uz+Mg)^2)
    m["E"] = float(_trapz(Fv2, t))
    m["x_final_err"] = [float(v) for v in (xi[-1] - tgt)]
    m["q1_final_err"] = [float(v) for v in (res["q1"][-1] - tgt[:2])]
    m["sigma2_max"] = float(res["sigma2"].max())
    return m


# --------------------------------------------------------------------------------------------- 3D rigid-body plant
def _hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + _hat(w)
    K = _hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Rigid3DPlant:
    """Drone (attitude on SO(3), body torques) + point-mass payload on a rigid massless cable attached at the drone CoM.
    Equations of Paper 1 (Sreenath-Lee-Kumar 2013, Eqs. 6-8) with q = unit vector drone -> payload:
        omega' = -(q x F) / (m_Q l),  q' = omega x q,  (m_Q+m_L) x_Q'' + m_L l q'' = F - (m_Q+m_L) g e3,  F = f R e3.
    Control input = (f, tau_x, tau_y, tau_z) held constant over each step, no actuator limits (isolates the controller).
    """

    def __init__(self, p: QSLTSParams, x_Q0=(0.0, 0.0, 0.0)):
        self.p = p
        self.t = 0.0
        self.x_Q = np.array(x_Q0, float)
        self.v_Q = np.zeros(3)
        self.R = np.eye(3)
        self.Om = np.zeros(3)
        self.q = np.array([0.0, 0.0, -1.0])
        self.omega = np.zeros(3)

    def _rhs(self, s, wrench):
        p = self.p
        x, v, R, Om, q, om = s
        f, tau = wrench[0], wrench[1:]
        F = f * R[:, 2]
        om_dot = -np.cross(q, F) / (p.M_q * p.l)
        q_dot = np.cross(om, q)
        q_ddot = np.cross(om_dot, q) - (om @ om) * q
        Mt = p.M_q + p.M_l
        a = (F - Mt * p.g * np.array([0, 0, 1.0]) - p.M_l * p.l * q_ddot) / Mt
        Om_dot = np.linalg.solve(p.J, tau - np.cross(Om, p.J @ Om))
        return v, a, R @ _hat(Om), Om_dot, q_dot, om_dot

    def step(self, wrench, dt):
        s0 = (self.x_Q, self.v_Q, self.R, self.Om, self.q, self.omega)

        def add(s, d, h):
            return (s[0] + h * d[0], s[1] + h * d[1], s[2] + h * d[2], s[3] + h * d[3], s[4] + h * d[4], s[5] + h * d[5])

        k1 = self._rhs(s0, wrench)
        k2 = self._rhs(add(s0, k1, dt / 2), wrench)
        k3 = self._rhs(add(s0, k2, dt / 2), wrench)
        k4 = self._rhs(add(s0, k3, dt), wrench)
        new = [s0[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(6)]
        self.x_Q, self.v_Q, self.R, self.Om, self.q, self.omega = new
        u, _, vt = np.linalg.svd(self.R)                        # re-orthonormalise
        self.R = u @ vt
        self.q = self.q / np.linalg.norm(self.q)
        self.omega = self.omega - self.q * (self.q @ self.omega)
        self.t += dt

    def state(self):
        """Same fields as sim.state.State (the ones the controller reads)."""
        q_dot = np.cross(self.omega, self.q)
        return SimpleNamespace(t=self.t, x_Q=self.x_Q.copy(), v_Q=self.v_Q.copy(), R=self.R.copy(), Omega=self.Om.copy(),
                               q=self.q.copy(), q_dot=q_dot, x_L=self.x_Q + self.p.l * self.q,
                               v_L=self.v_Q + self.p.l * q_dot, omega=self.omega.copy())


def run_rigid3d(P: float, target_delta, T: float = 20.0, dt: float = 0.002, params: QSLTSParams | None = None,
                gains: BLFGains | None = None, att_w: float = 30.0, wrench_limit=None, z_coupling: str = "drop", ramp: float = 0.0) -> dict:
    """Run the repo-facing adapter BLFAntiSwing.step(State) on the 3D plant. target_delta: displacement from the start."""
    params = params or PAPER_PARAMS
    gains = gains or BLFGains(P=P)
    gains.P = P
    plant = Rigid3DPlant(params)
    tgt = plant.x_Q + np.asarray(target_delta, float)
    ctrl = BLFAntiSwing(params, gains, AttitudeGains.from_bandwidth(params.J, w=att_w), target_xyz=tgt,
                        z_coupling=z_coupling)
    n = int(round(T / dt))
    log = {k: [] for k in ("t", "x_Q", "x_L", "ab", "F", "tau", "Phi", "Phi_d")}
    x0 = plant.x_Q.copy()
    for _ in range(n):
        if ramp > 0:
            ctrl.target = x0 + (tgt - x0) * quintic_ramp(plant.t / ramp)
        s = plant.state()
        w = ctrl.step(s, dt)
        if wrench_limit is not None:                           # simple box saturation of (f, tau_xy, tau_z)
            w = np.array([np.clip(w[0], 0.0, wrench_limit["F_max"]),
                          *np.clip(w[1:3], -wrench_limit["tau_xy_max"], wrench_limit["tau_xy_max"]),
                          np.clip(w[3], -wrench_limit["tau_z_max"], wrench_limit["tau_z_max"])])
        log["t"].append(plant.t); log["x_Q"].append(s.x_Q); log["x_L"].append(s.x_Q + params.l * s.q)
        log["ab"].append(ctrl.last["ab"]); log["F"].append(w[0]); log["tau"].append(w[1:])
        log["Phi"].append(ctrl.last["Phi"]); log["Phi_d"].append(ctrl.last["Phi_d"])
        plant.step(w, dt)
    out = {k: np.array(v) for k, v in log.items()}
    out["target"] = tgt
    out["params"] = params
    return out
