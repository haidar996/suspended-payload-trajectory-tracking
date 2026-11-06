"""Anti-swing controller of Ding, Sun, He, "Anti-Swing Control for Quadrotor-Slung Load Transportation System with
Underactuated State Constraints", Sensors 23, 8995 (2023).  [Paper 4 in this repo's numbering]

Pure numpy (no MuJoCo import), so it can be unit-tested anywhere. Works with the repo's C1 + P1 model (rigid massless
cable, point-mass payload, cable attached at the drone CoM).

Notation of the paper
---------------------
xi1 = (x, y)  drone position     xi2 = z  drone altitude     xi3 = (alpha, beta)  payload swing angles
payload position = drone + l * (sin a cos b, sin b, -cos a cos b)   =>   q (repo) = that unit vector.
U = (Ux, Uy, Uz): net force on the drone+payload system, gravity of the whole system already removed from Uz.

Structure (paper Fig. 2): outer loop  = position / anti-swing law  (Theorem 1, or Theorem 2 with barrier term)
                          Eq. (20)    = force -> desired thrust F and desired Euler angles (phi_d, theta_d)
                          inner loop  = PD attitude law, Eq. (21)

Cascade (paper Eq. 3, with the true sign f1 = -G):
    q1' = m31^-1 q2,   q2' = f1(q3) (+ f2, neglected),   q3' = q4,   q4' = g1 + b1 u,   q5' = q6,   q6' = g2 + b2 u
    f1(q3) = -g (sin a, cos a sin b)
Back-stepping errors  z1 = e1, z2 = e2 + k1 z1, z3 = e3 + k2 z2, z4 = e4 + k3 z3, z5 = e5, z6 = e6 + k4 z5.

Deviations from the printed paper (each one deliberate, see the notes at the bottom of this docstring):
  D1  sign of df1/dq3: the paper prints +g[...]; with q2' = -G the consistent value is -g[...]  (the control structure is
      unchanged: u1/u2 below make z4' = -(th1-k3) z4 - Gamma z3 and z6' = -(th2-k4) z6 - z5 exactly).
  D2  attitude loop: the paper's W term is replaced by the rigid-body gyroscopic term  Omega x J Omega  (the paper's
      sign is inconsistent with its own Eq. (2)); gains are scaled with the inertia (Kp = w^2 J, Kv = 2 zeta w J) because
      the paper does not give Kp, Kv. Desired Euler rates are taken as zero (paper does not specify them).
  D3  barrier argument clipped to sigma^T sigma <= 1 - 1e-6 so a transient that leaves the barrier does not produce NaN.
  D4  z-coupling (option z_coupling="drop", the default). The paper's q1 contains the path-dependent integral
      int m31^-1 m32 z' dt (m32 = (sin a, cos a sin b)). At rest q1 -> q1d only gives xi1 = q1d - iota_inf, and iota_inf is
      NOT zero after a climb with a swinging load (paper's proof assumes it is). On the paper's own example (20 m climb) the
      literal law ends 2.4 m (BLF) / 3.4 m (P = 0) away from the target. With "drop" the regulated variables are
          q1 = xi1 + int m31^-1 m33 xi3' dt  (-> xi1 + l xi3 for small angles),   q2 = m31 xi1' + m33 xi3'
      so the load of the altitude motion, -m32 z'', is a transient disturbance of the q2 equation (it vanishes once z
      settles) and the end position is exact. z_coupling="paper" keeps the literal definition (used to show the offset).
  D5  reference shaping (not a change of the law). The paper applies a raw 20 m step, which asks the attitude loop for a
      +-20 deg tilt step with ~11 N*m at t = 0. At the thrust level of such a move the X2's rotors (0-13 N) can give only
      ~+-2.4 N*m, so the attitude loop lags, the position loop reacts to the lag and the run diverges (seen with the repo's
      real mixer for steps >= ~16 m; steps <= ~14 m work). Moving `target` along a quintic ramp (quintic_ramp, >= 2-3 s
      for 20 m) keeps the demand inside the actuator limits and the unchanged law converges without saturation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

E3 = np.array([0.0, 0.0, 1.0])


def quintic_ramp(u):
    """Smooth 0 -> 1 ramp (zero velocity and acceleration at both ends), u = t / duration; use it to move `target`."""
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10.0 - 15.0 * u + 6.0 * u * u)


# --------------------------------------------------------------------------------------------------- parameters
@dataclass
class QSLTSParams:
    M_q: float                 # drone mass [kg]
    M_l: float                 # payload mass [kg]
    l: float                   # cable length [m]
    J: np.ndarray              # 3x3 drone inertia about its CoM, body axes
    g: float = 9.81

    @staticmethod
    def from_flat(p) -> "QSLTSParams":
        """From the repo's FlatParams (sim.flatness) / checks_flatness.params(cfg)."""
        return QSLTSParams(M_q=p.m_Q, M_l=p.m_L, l=p.l, J=np.asarray(p.J, float), g=p.g)


@dataclass
class BLFGains:
    """Paper Sec. 4 values by default. Conditions (5)/(12): k_i > 0, th1 > k3, th2 > k4, Q_j > 0 (the effective damping
    of z4 is th1 - k3 and of z6 is th2 - k4)."""
    k1: np.ndarray = None
    k2: np.ndarray = None
    k3: np.ndarray = None
    k4: float = 0.25
    th1: np.ndarray = None
    th2: float = 2.0
    P: float = 0.05            # barrier scale; P = 0 -> Theorem 1 (no swing constraint)
    kappa_deg: float = 15.0    # swing-angle bound |alpha|, |beta| <= kappa

    def __post_init__(self):
        d = lambda v: np.asarray(v, float) * np.ones(2)
        self.k1 = d([0.5, 0.5] if self.k1 is None else self.k1)
        self.k2 = d([2.5, 2.5] if self.k2 is None else self.k2)
        self.k3 = d([20.0, 20.0] if self.k3 is None else self.k3)
        self.th1 = d([20.25, 20.25] if self.th1 is None else self.th1)

    def f1_bar(self, g: float) -> float:
        """Upper bound of |e3| = |f1|: |sin(alpha)|, |cos(alpha) sin(beta)| <= sin(kappa)  ->  f1_bar = g sin(kappa).
        The paper's Note 3 (f1_bar > k2) holds with its numbers: 9.8 sin 15deg = 2.536 > 2.5."""
        return g * float(np.sin(np.radians(self.kappa_deg)))

    def check(self, g: float) -> dict:
        """Necessary conditions (5)/(12) with m31 ~ I (paper Note 2)."""
        I = np.eye(2)
        k1, k2, k3 = map(np.diag, (self.k1, self.k2, self.k3))
        K1 = I - k1 @ k1
        K2 = I + k2 @ k1 - k2 @ k2
        K3 = k2 @ k1 @ k1
        Q1 = k1 - 0.25 * K1.T @ K1 - 0.25 * K3.T @ K3
        Q2 = k2 - k1 - 1.25 * I
        Q3 = k3 - k2 - K2.T @ K2 - I
        out = {"Q1_min_eig": float(np.linalg.eigvalsh(Q1).min()), "Q2_min_eig": float(np.linalg.eigvalsh(Q2).min()),
               "Q3_min_eig": float(np.linalg.eigvalsh(Q3).min()),
               "th1_minus_k3_min": float(np.min(self.th1 - self.k3)), "th2_minus_k4": float(self.th2 - self.k4),
               "k4": self.k4, "f1_bar_minus_k2_min": float(self.f1_bar(g) - self.k2.max())}
        out["ok"] = bool(out["Q1_min_eig"] > 0 and out["Q2_min_eig"] > 0 and out["Q3_min_eig"] > 0
                         and out["th1_minus_k3_min"] > 0 and out["th2_minus_k4"] > 0 and out["k4"] > 0
                         and (self.P == 0 or out["f1_bar_minus_k2_min"] > 0))
        return out


@dataclass
class AttitudeGains:
    Kp: np.ndarray
    Kv: np.ndarray

    @staticmethod
    def from_bandwidth(J: np.ndarray, w: float = 30.0, zeta: float = 0.9) -> "AttitudeGains":
        Jd = np.diag(np.diag(np.asarray(J, float)))
        return AttitudeGains(Kp=w**2 * Jd, Kv=2 * zeta * w * Jd)


# --------------------------------------------------------------------------------------------------- model blocks
def swing_angles(q: np.ndarray) -> np.ndarray:
    """(alpha, beta) from the unit cable vector q = (sin a cos b, sin b, -cos a cos b)."""
    return np.array([np.arctan2(q[0], -q[2]), np.arcsin(np.clip(q[1], -1.0, 1.0))])


def swing_rates(q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    cb = np.sqrt(max(1.0 - q[1] ** 2, 1e-12))
    return np.array([(-q[2] * q_dot[0] + q[0] * q_dot[2]) / max(q[0] ** 2 + q[2] ** 2, 1e-12), q_dot[1] / cb])


def m_blocks(p: QSLTSParams, a: float, b: float):
    """m31 (2x2), m32 (2,), m33 (2x2) of paper Eq. (1)."""
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    m31 = np.array([[ca, 0.0], [-sa * sb, cb]])
    m32 = np.array([sa, ca * sb])
    m33 = p.l * np.array([[cb, 0.0], [0.0, 1.0]])
    return m31, m32, m33


def mass_and_bias(p: QSLTSParams, ab: np.ndarray, abd: np.ndarray):
    """Paper Eq. (1) written as  M(q) acc = E u + h,  acc = (x'', y'', z'', alpha'', beta''),  u = (Ux, Uy, Uz)."""
    a, b = ab
    ad, bd = abd
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    Mt, Ml, l, g = p.M_q + p.M_l, p.M_l, p.l, p.g
    m31, m32, m33 = m_blocks(p, a, b)
    M = np.zeros((5, 5))
    M[0:2, 0:2] = Mt * np.eye(2)
    M[0:2, 3:5] = Ml * l * np.array([[ca * cb, -sa * sb], [0.0, cb]])
    M[2, 2] = Mt
    M[2, 3:5] = Ml * l * np.array([sa * cb, ca * sb])
    M[3:5, 0:2] = m31
    M[3:5, 2] = m32
    M[3:5, 3:5] = m33
    C1 = Ml * l * np.array([-sa * cb * ad**2 - sa * cb * bd**2 - 2 * ca * sb * ad * bd, -sb * bd**2])
    C2 = Ml * l * (ca * cb * ad**2 + ca * cb * bd**2 - 2 * sa * sb * ad * bd)
    C3 = l * np.array([-2 * sb * ad * bd, cb * sb * ad**2])
    G = g * np.array([sa, ca * sb])
    h = -np.concatenate([C1, [C2], C3 + G])
    E = np.zeros((5, 3))
    E[0, 0] = E[1, 1] = E[2, 2] = 1.0
    return M, E, h


def f1_of(p: QSLTSParams, ab: np.ndarray) -> np.ndarray:
    a, b = ab
    return -p.g * np.array([np.sin(a), np.cos(a) * np.sin(b)])


def df1_dq3(p: QSLTSParams, ab: np.ndarray) -> np.ndarray:
    a, b = ab
    return -p.g * np.array([[np.cos(a), 0.0], [-np.sin(a) * np.sin(b), np.cos(a) * np.cos(b)]])


def ddf1_dq3_dt(p: QSLTSParams, ab: np.ndarray, abd: np.ndarray) -> np.ndarray:
    a, b = ab
    ad, bd = abd
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    return -p.g * np.array([[-sa * ad, 0.0],
                            [-(ca * sb * ad + sa * cb * bd), -sa * cb * ad - ca * sb * bd]])


# --------------------------------------------------------------------------------------------------- Euler helpers
def euler_from_R(R: np.ndarray) -> np.ndarray:
    """ZYX Euler angles (phi, theta, psi) with R = Rz(psi) Ry(theta) Rx(phi)  (paper's convention: Eq. 1 / Eq. 20)."""
    return np.array([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1.0, 1.0)), np.arctan2(R[1, 0], R[0, 0])])


def euler_rates(Phi: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """Euler-angle rates from body angular velocity (ZYX)."""
    ph, th, _ = Phi
    p_, q_, r_ = Omega
    t = np.tan(th)
    return np.array([p_ + (q_ * np.sin(ph) + r_ * np.cos(ph)) * t,
                     q_ * np.cos(ph) - r_ * np.sin(ph),
                     (q_ * np.sin(ph) + r_ * np.cos(ph)) / np.cos(th)])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def desired_attitude(p: QSLTSParams, U: np.ndarray, psi_d: float = 0.0):
    """Paper Eq. (20): thrust magnitude and desired roll / pitch from the commanded forces."""
    Fv = np.array([U[0], U[1], U[2] + (p.M_q + p.M_l) * p.g])
    F = float(np.linalg.norm(Fv))
    F = max(F, 1e-6)
    phi_d = np.arcsin(np.clip((U[0] * np.sin(psi_d) - U[1] * np.cos(psi_d)) / F, -1.0, 1.0))
    th_arg = (U[0] * np.cos(psi_d) + U[1] * np.sin(psi_d)) / (F * np.cos(phi_d))
    theta_d = np.arcsin(np.clip(th_arg, -1.0, 1.0))
    return F, np.array([phi_d, theta_d, psi_d])


# --------------------------------------------------------------------------------------------------- controller
class BLFAntiSwing:
    """Position (anti-swing) law of Theorems 1/2 + Eq. (20) + PD attitude law Eq. (21).

    Call `step(...)` every control tick. Internal state: the integral that defines q1 (paper Eq. 3):
        q1 = xi1 + int m31^-1 (m32 z' + m33 xi3') dt,   q1(0) = xi1(0).
    """

    def __init__(self, p: QSLTSParams, gains: BLFGains | None = None, att: AttitudeGains | None = None,
                 target_xyz=(0.0, 0.0, 0.0), psi_d: float = 0.0, z_coupling: str = "drop"):
        if z_coupling not in ("drop", "paper"):
            raise ValueError(z_coupling)
        self.z_coupling = z_coupling
        self.p = p
        self.k = gains or BLFGains()
        self.att = att or AttitudeGains.from_bandwidth(p.J)
        self.target = np.asarray(target_xyz, float)
        self.psi_d = psi_d
        self.iota = np.zeros(2)
        self._iota_dot_prev = None
        self.last: dict = {}

    def reset(self):
        self.iota = np.zeros(2)
        self._iota_dot_prev = None

    # ----------------------------------------------------------------- outer loop
    def position_forces(self, xi1, xi1_dot, z, z_dot, ab, abd, dt) -> np.ndarray:
        p, k = self.p, self.k
        a, b = ab
        m31, m32, m33 = m_blocks(p, a, b)
        m31i = np.linalg.inv(m31)

        # integral state of q1 (trapezoid). "paper": iota includes the m32 z' term; "drop": it does not (D4)
        w = m33 @ abd + (m32 * z_dot if self.z_coupling == "paper" else 0.0)
        iota_dot = m31i @ w
        if self._iota_dot_prev is not None:
            self.iota = self.iota + 0.5 * dt * (iota_dot + self._iota_dot_prev)
        self._iota_dot_prev = iota_dot

        q1 = np.asarray(xi1, float) + self.iota
        q2 = m31 @ xi1_dot + m33 @ abd + (m32 * z_dot if self.z_coupling == "paper" else 0.0)
        f1 = f1_of(p, ab)
        Fq = df1_dq3(p, ab)
        e1 = q1 - self.target[:2]
        e2 = q2
        e3 = f1
        e4 = Fq @ abd
        e5 = z - self.target[2]
        e6 = z_dot
        z1 = e1
        z2 = e2 + k.k1 * z1
        z3 = e3 + k.k2 * z2
        z4 = e4 + k.k3 * z3
        z5 = e5
        z6 = e6 + k.k4 * z5

        # barrier terms (Theorem 2); P = 0 gives Gamma = 1 and no extra term (Theorem 1)
        f1b = k.f1_bar(p.g)
        sigma = e3 / f1b
        s2 = min(float(sigma @ sigma), 1.0 - 1e-6)                     # D3
        Gamma = 1.0 + k.P * np.log(1.0 / (1.0 - s2))
        extra = k.P * sigma * float(z3 @ z3) / (f1b * (1.0 - s2))

        M, E, h = mass_and_bias(p, ab, abd)
        Minv = np.linalg.inv(M)
        gv = Minv @ h
        B = Minv @ E
        g1, b1 = gv[3:5], B[3:5, :]
        g2, b2 = gv[2], B[2, :]

        K1, K2, K3 = np.diag(k.k1), np.diag(k.k2), np.diag(k.k3)
        u1 = (ddf1_dq3_dt(p, ab, abd) @ abd + Fq @ g1 + k.th1 * z4 + (Gamma + k.k3 * k.k2 - k.k3 * k.k3) * z3
              + (K2 @ K3) @ (K1 @ m31i - K2) @ z2 - (K3 @ K2 @ K1) @ m31i @ (K1 @ z1) + extra)
        u2 = g2 + k.th2 * z6 + (1.0 - k.k4 * k.k4) * z5
        Lam = np.vstack([Fq @ b1, b2[None, :]])
        U = -np.linalg.solve(Lam, np.concatenate([u1, [u2]]))

        V1 = 0.5 * float(sum(zz @ zz for zz in (z1, z2, z3, z4))) + 0.5 * (z5**2 + z6**2)
        self.last.update(z=(z1, z2, z3, z4, z5, z6), e=(e1, e2, e3, e4, e5, e6), sigma2=s2, Gamma=Gamma, q1=q1,
                         V1=V1, U=U)
        return U

    # ----------------------------------------------------------------- inner loop
    def attitude_torque(self, Phi, Phi_dot, Phi_d, Omega=None) -> np.ndarray:
        """Eq. (21): PD on Euler angles + gyroscopic compensation (D2). Returns the body torque."""
        e_phi = wrap(np.asarray(Phi) - np.asarray(Phi_d))
        e_om = np.asarray(Phi_dot)                                     # desired Euler rates taken as 0
        tau = -self.att.Kp @ e_phi - self.att.Kv @ e_om
        if Omega is not None:
            tau = tau + np.cross(Omega, self.p.J @ Omega)
        return tau

    # ----------------------------------------------------------------- full loop on a repo-style State
    def step(self, s, dt: float) -> np.ndarray:
        """s: object with x_Q, v_Q, R, Omega, q, q_dot (sim.state.State). Returns the CTBM wrench (f, tau_x, tau_y, tau_z)."""
        ab = swing_angles(s.q)
        abd = swing_rates(s.q, s.q_dot)
        U = self.position_forces(s.x_Q[:2], s.v_Q[:2], s.x_Q[2], s.v_Q[2], ab, abd, dt)
        F, Phi_d = desired_attitude(self.p, U, self.psi_d)
        Phi = euler_from_R(s.R)
        tau = self.attitude_torque(Phi, euler_rates(Phi, s.Omega), Phi_d, s.Omega)
        self.last.update(F=F, Phi_d=Phi_d, Phi=Phi, ab=ab, abd=abd)
        return np.array([F, *tau])
