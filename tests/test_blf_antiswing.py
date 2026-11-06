"""Tests of the Ding-Sun-He anti-swing controller. numpy only: no MuJoCo needed (see sim/checks_blf.py)."""
import numpy as np

from sim import checks_blf as C
from sim.controllers import blf_antiswing as B


def _d(a, b):
    return np.array([np.sin(a) * np.cos(b), np.sin(b), -np.cos(a) * np.cos(b)])


def test_model_blocks_satisfy_newton():
    """M(q) acc = E u + h (paper Eq. 1) must reproduce Newton's laws of drone + payload computed from the geometry."""
    rng = np.random.default_rng(1)
    p = C.PAPER_PARAMS
    for _ in range(50):
        ab, abd, u = rng.uniform(-0.6, 0.6, 2), rng.normal(size=2), rng.normal(size=3) * 5
        M, E, h = B.mass_and_bias(p, ab, abd)
        acc = np.linalg.solve(M, E @ u + h)
        xi_dd, ab_dd = acc[:3], acc[3:]
        eps = 1e-4
        f = lambda t: _d(*(ab + abd * t + 0.5 * ab_dd * t * t))
        p_dd = xi_dd + p.l * (f(eps) - 2 * f(0) + f(-eps)) / eps**2
        assert np.abs(p.M_q * xi_dd + p.M_l * p_dd - u).max() < 1e-5
        g3 = np.array([0, 0, p.g])
        for da in ((_d(ab[0] + eps, ab[1]) - _d(ab[0] - eps, ab[1])) / (2 * eps),
                   (_d(ab[0], ab[1] + eps) - _d(ab[0], ab[1] - eps)) / (2 * eps)):
            assert abs(da @ (p_dd + g3)) < 1e-5


def test_helpers_round_trip():
    rng = np.random.default_rng(2)
    for _ in range(20):
        ab, abd = rng.uniform(-0.8, 0.8, 2), rng.normal(size=2)
        q = _d(*ab)
        qd = (_d(*(ab + abd * 1e-6)) - _d(*(ab - abd * 1e-6))) / 2e-6
        assert np.abs(B.swing_angles(q) - ab).max() < 1e-12
        assert np.abs(B.swing_rates(q, qd) - abd).max() < 1e-8
    Phi = np.array([0.3, -0.4, 0.7])
    Rz = lambda t: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    Ry = lambda t: np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    Rx = lambda t: np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    assert np.abs(B.euler_from_R(Rz(Phi[2]) @ Ry(Phi[1]) @ Rx(Phi[0])) - Phi).max() < 1e-12
    U = np.array([2.0, -1.0, 3.0])
    F, Pd = B.desired_attitude(C.PAPER_PARAMS, U, 0.0)
    Rd = Rz(Pd[2]) @ Ry(Pd[1]) @ Rx(Pd[0])
    assert np.abs(F * Rd[:, 2] - (U + np.array([0, 0, 1.2 * 9.8]))).max() < 1e-12


def test_paper_gains_satisfy_stability_conditions():
    chk = B.BLFGains().check(9.8)
    assert chk["ok"], chk
    assert abs(chk["Q1_min_eig"] - 0.2617) < 1e-3                     # hand-computed from the paper's gains


def test_paper_replication_swing_angles():
    """Literal paper law (z_coupling='paper') reproduces the paper's Table 2 swing peaks and the BLF/Lyapunov ordering."""
    res = {n: C.paper_metrics(C.run_paper_replication(P=P, T=8.0, z_coupling="paper"), ) for n, P in (("BLF", 0.05), ("Lyap", 0.0))}
    assert abs(res["BLF"]["a_max"] - 9.57) < 0.5 and abs(res["BLF"]["b_max"] - 9.76) < 0.5
    assert abs(res["Lyap"]["a_max"] - 17.52) < 1.0 and abs(res["Lyap"]["b_max"] - 19.11) < 1.0
    assert abs(res["BLF"]["t_sz"] - 3.37) < 0.2
    assert res["BLF"]["a_max"] < res["Lyap"]["a_max"]


def test_literal_paper_law_leaves_position_offset_and_drop_removes_it():
    """The paper's q1 integral int m31^-1 m32 z' dt does not vanish after a climb with a swinging load (finding D4)."""
    lit = C.paper_metrics(C.run_paper_replication(P=0.05, T=20.0, z_coupling="paper"))
    fix = C.paper_metrics(C.run_paper_replication(P=0.05, T=20.0, z_coupling="drop"))
    assert np.abs(lit["x_final_err"][:2]).min() > 1.5
    assert np.abs(fix["x_final_err"][:2]).max() < 0.5
    assert abs(fix["t_sx"] - 15.01) < 2.0                              # paper: 15.01 s


def test_rigid3d_adapter_x2_with_limits():
    """State -> wrench adapter on a full 3D drone + pendulum with X2 mass/inertia and thrust/torque limits."""
    r = C.run_rigid3d(0.05, (10.0, 10.0, 3.0), T=20.0, dt=0.002, params=C.X2_PARAMS, att_w=30.0, wrench_limit=C.X2_LIMITS)
    assert np.isfinite(r["x_Q"]).all()
    assert np.linalg.norm(r["x_Q"][-1] - r["target"]) < 0.2
    assert np.degrees(np.abs(r["ab"]).max()) < 15.0


def test_ramped_20m_move_x2_with_limits():
    """Shaped reference (D5): a 20 m diagonal move converges on the X2-parameter plant and the barrier lowers the swing."""
    kw = dict(T=40.0, dt=0.002, params=C.X2_PARAMS, att_w=30.0, wrench_limit=C.X2_LIMITS, ramp=3.0)
    blf = C.run_rigid3d(0.05, (20.0, 20.0, 5.0), **kw)
    lyap = C.run_rigid3d(0.0, (20.0, 20.0, 5.0), **kw)
    for r in (blf, lyap):
        assert np.isfinite(r["x_Q"]).all() and np.linalg.norm(r["x_Q"][-1] - r["target"]) < 0.2
    assert np.abs(blf["ab"]).max() < np.abs(lyap["ab"]).max()
