#!/usr/bin/env python3
"""论文复现（单文件版）。

Runkun Li, Wenhai Qi, Guangdeng Zong, Shun-Feng Su, Jinde Cao,
"Reinforcement Learning-Based Optimization for Interval Type-2 Fuzzy Unknown
Nonlinear System: An Zero-Sum Control Scheme", IEEE Trans. Cybern., 2026.
DOI: 10.1109/TCYB.2026.3720445

与论文的对应关系：
  第 II 节   区间二型模糊模型，式 (1)(2)(6)
  第 III 节  Algorithm 1：基于模型的策略迭代（Kleinman 形式，式 (20)–(22)）
             Algorithm 2：无模型离策略积分强化学习（式 (25)、(36)–(43)）
  第 IV 节   半车主动悬架（SCASM）算例，图 3–图 8

运行：
  python reproduce_it2fuzzy.py                     # 默认：推导修正模型
  python reproduce_it2fuzzy.py --model paper_a53   # A(5,3) 取论文印刷值 ζ3·S_fwI
  python reproduce_it2fuzzy.py --help

学习器只使用测量数据 z、u、w 的区间积分；A、B、D 只用于（1）仿真产生数据、
（2）独立求解博弈 Riccati 方程用于对照，参考解不传入学习器。
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp

# ============================================================================
# 0. 论文参数（表 II 与第 IV 节）
# ============================================================================
MC, J_THETA = 685.0, 1380.0          # 车身质量 kg，俯仰转动惯量 kg·m²
L1, L2 = 1.35, 1.55                  # 质心到前、后轴距离 m
SF1, SF2 = 17500.0, 21000.0          # 前、后悬架刚度 N/m
SR1, SR2 = 210000.0, 210000.0        # 前、后轮胎刚度 N/m
SW1, SW2 = 1100.0, 1100.0            # 前、后悬架阻尼 N·s/m
MW1, MW2 = 42.0, 45.0                # 前、后非簧载质量 kg

EPS = 0.01 / np.pi
GAMMA = 1.2
Q = 0.1 * np.eye(8)
R = np.eye(2)
Z0 = np.array([2.0, 5.0, 5.0, 2.0, 5.0, 5.0, 8.0, 8.0])
T_WIN = 0.05                         # 积分窗口 T
DELTA = 1e-7                         # 停止阈值 δ（逐元素）
N_X, N_U, N_W = 8, 2, 2
N_P = N_X * (N_X + 1) // 2           # P 的独立元素 36
N_UNKNOWN = N_P + N_U * N_X + N_W * N_X  # 36 + 16 + 16 = 68（论文正文误写为 40）

# 论文第 8 页打印的 K*、N*，仅用于事后对照，不参与学习
PAPER_K = np.array([
    [[-0.0012, 0.0006, -0.0000, -0.0028, -0.0005, -0.0003, 0.0019, -0.0005],
     [0.0013, -0.0005, -0.0001, 0.0028, 0.0007, 0.0002, -0.0038, 0.0]],
    [[0.0, -0.0020, -0.0002, 0.0192, -0.0005, 0.0010, 0.0500, -0.0011],
     [0.0, 0.0022, 0.0001, -0.0194, 0.0007, -0.0011, -0.0524, 0.0]],
])
PAPER_N = np.array([
    [[-0.0293, 0.0843, -0.0071, -0.0090, 0.0526, -0.0044, -8.4129, -0.3090],
     [-0.0305, 0.0013, 0.0031, -0.0803, 0.0807, -0.0074, -0.3089, -8.2897]],
    [[-0.0298, 0.0840, -0.0071, -0.0065, 0.0541, -0.0041, -8.4176, -0.2860],
     [-0.0340, 0.0001, 0.0029, -0.0851, 0.0807, -0.0073, -0.2844, -8.2770]],
])


# ============================================================================
# 1. 半车悬架模型、非线性项与区间二型隶属函数
# ============================================================================
def mechanical_matrices(model="corrected"):
    """式 (44) 的 A、B、D。

    状态 z = [前悬架变形, 前车身速度, 前轮速度, 后悬架变形, 后车身速度,
              后轮速度, 前轮胎变形, 后轮胎变形]；u 为前后作动力，w 为路面速度。
    由第 IV 节四个运动方程推导，相对印刷矩阵的修正：
      ζ2 = 1/Mc − lI·lII/Jθ（印刷为 +）；补 A(2,4) = −ζ2 S_fII、A(3,7) = −S_rI/M_wI、
      A(5,4) = −ζ3 S_fII；A(5,3) = ζ2 S_fwI（印刷为 ζ3 S_fwI）；A(5,5) = −ζ3 S_fwII（印刷为 +）。
    model="paper_a53" 仅把 A(5,3) 恢复为印刷值 ζ3 S_fwI，其余修正保留。
    """
    z1 = 1 / MC + L1**2 / J_THETA
    z2 = 1 / MC - L1 * L2 / J_THETA
    z3 = 1 / MC + L2**2 / J_THETA
    A = np.array([
        [0, 1, -1, 0, 0, 0, 0, 0],
        [-z1 * SF1, -z1 * SW1, z1 * SW1, -z2 * SF2, -z2 * SW2, z2 * SW2, 0, 0],
        [SF1 / MW1, SW1 / MW1, -SW1 / MW1, 0, 0, 0, -SR1 / MW1, 0],
        [0, 0, 0, 0, 1, -1, 0, 0],
        [-z2 * SF1, -z2 * SW1, z2 * SW1, -z3 * SF2, -z3 * SW2, z3 * SW2, 0, 0],
        [0, 0, 0, SF2 / MW2, SW2 / MW2, -SW2 / MW2, 0, -SR2 / MW2],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ], dtype=float)
    if model == "paper_a53":
        A[4, 2] = z3 * SW1
    elif model != "corrected":
        raise ValueError(f"未知模型 {model}")
    B = np.array([[0, 0], [z1, z2], [-1 / MW1, 0], [0, 0],
                  [z2, z3], [0, -1 / MW2], [0, 0], [0, 0]], dtype=float)
    D = np.zeros((8, 2))
    D[6, 0] = D[7, 1] = -1.0
    return A, B, D


def local_matrices(A, B):
    """规则 α 的局部矩阵：A_α = A + 10⁴c_α b1 e1ᵀ + 10⁴ b2 e4ᵀ，c1 = 1，c2 = ε。"""
    e1, e4 = np.eye(8)[0], np.eye(8)[3]
    return [A + 1e4 * c * np.outer(B[:, 0], e1) + 1e4 * np.outer(B[:, 1], e4)
            for c in (1.0, EPS)]


def rho(z):
    """非线性项 ρ(z) = 10⁴[s(z1)·z1, z4]，s = 1 − e^{−z1²} + (e^{−z1²} − e^{−2z1²}) sin z1。"""
    z1 = z[..., 0]
    a, b = np.exp(-z1**2), np.exp(-2 * z1**2)
    s = 1 - a + (a - b) * np.sin(z1)
    return 1e4 * np.stack([s * z1, z[..., 3]], axis=-1)


def membership_bounds(z1):
    """第 IV 节的上下隶属度 (h̲1, h̄1, h̲2, h̄2)，e^{−z1²/0.5} = e^{−2z1²}。"""
    a, b = np.exp(-z1**2), np.exp(-2 * z1**2)
    return 1 - a / (1 - EPS), 1 - b / (1 - EPS), b / (1 - EPS), a / (1 - EPS)


def fuzzy_weights(z1):
    """H_α = h̲_α f̲_α + h̄_α f̄_α。H2 的第二项取 h̄2·f̄2（印刷为 h̄2·h̄2），保证 H1 + H2 = 1。"""
    lo1, up1, lo2, up2 = membership_bounds(z1)
    s = np.sin(z1)
    return np.stack([lo1 * (1 - s) + up1 * s, lo2 * s + up2 * (1 - s)])


def check_fuzzy_equivalence(A, B, A_loc, n=2000, seed=0):
    """数值核对 Σ H_α A_α z = A z + B ρ(z) 及 H1 + H2 = 1。"""
    z = np.random.default_rng(seed).uniform(-5, 5, size=(n, 8))
    H = fuzzy_weights(z[:, 0])
    fuzzy = H[0][:, None] * (z @ A_loc[0].T) + H[1][:, None] * (z @ A_loc[1].T)
    exact = z @ A.T + rho(z) @ B.T
    return float(np.abs(fuzzy - exact).max()), float(np.abs(H.sum(0) - 1).max())


# ============================================================================
# 2. Algorithm 1：基于模型的策略迭代与参考解
# ============================================================================
def game_are(A_a, B, D):
    """博弈代数 Riccati 方程 AᵀP + PA + Q − PBR⁻¹BᵀP + γ⁻²PDDᵀP = 0 的稳定解。"""
    P = la.solve_continuous_are(A_a, np.hstack([B, D]), Q,
                                la.block_diag(R, -GAMMA**2 * np.eye(N_W)))
    P = (P + P.T) / 2
    res = A_a.T @ P + P @ A_a + Q - P @ B @ la.solve(R, B.T) @ P + P @ D @ D.T @ P / GAMMA**2
    return P, float(np.abs(res).max())


def gains_from_P(P, B, D):
    """K = R⁻¹BᵀP，N = γ⁻²DᵀP（式 (15) 中 N 的 Bᵀ 应为 Dᵀ，与式 (22) 一致）。"""
    return la.solve(R, B.T @ P), D.T @ P / GAMMA**2


def model_based_pi(A_a, B, D, max_iter=50):
    """Algorithm 1（Kleinman 形式，式 (20)–(22)），从 K⁰ = N⁰ = 0 开始。"""
    K, N = np.zeros((N_U, N_X)), np.zeros((N_W, N_X))
    P_prev, norms = None, []
    for _ in range(max_iter):
        Ai = A_a - B @ K + D @ N
        Qi = Q + K.T @ R @ K - GAMMA**2 * N.T @ N
        P = la.solve_continuous_lyapunov(Ai.T, -Qi)
        norms.append(float(la.norm(P, 2)))
        K, N = gains_from_P(P, B, D)
        if P_prev is not None and np.abs(P - P_prev).max() < DELTA:
            break
        P_prev = P
    return P, norms


# ============================================================================
# 3. Algorithm 2：无模型离策略积分强化学习
# ============================================================================
IU = np.triu_indices(N_X)


def vecs_inv(p):
    """由 P̄ = [P11, 2P12, …, P22, …, Pnn]（式 (28)）还原对称矩阵 P。"""
    M = np.zeros((N_X, N_X))
    M[IU] = p
    return (M + M.T) / 2


def unpack_sym(v):
    """由上三角向量还原对称矩阵（用于 ∫zzᵀ）。"""
    M = np.zeros(v.shape[:-1] + (N_X, N_X))
    M[..., IU[0], IU[1]] = v
    return M + np.swapaxes(M, -1, -2) - M * np.eye(N_X)


def numerical_rank(M, rcond=1e-10):
    """列归一化后的 SVD 数值秩，同时返回最小/最大奇异值比。"""
    norms = np.linalg.norm(M, axis=0)
    s = la.svdvals(M / np.where(norms > 0, norms, 1.0))
    return int((s > rcond * s[0]).sum()), float(s[-1] / s[0])


def paper_probes():
    """论文原探测：E1 = (e^{−0.1t}+0.01)sin10t 作用于 u，E2 = (e^{−0.1t}+0.1)sin10t 作用于 w。

    论文未给出逐通道分配，此处两路 u 同用 E1、两路 w 同用 E2。
    """
    def e_u(t):
        return np.full(N_U, (np.exp(-0.1 * t) + 0.01) * np.sin(10 * t))

    def e_w(t):
        return np.full(N_W, (np.exp(-0.1 * t) + 0.1) * np.sin(10 * t))
    return e_u, e_w


def multisine_probes(amp_u, amp_w, seed):
    """补充探测：四个输入通道各自使用四频正弦和，频率互不相同（论文未给出的实验设定）。"""
    freqs = np.array([[1.1, 3.7, 8.3, 17.9],     # u1
                      [1.9, 5.3, 11.6, 23.1],    # u2
                      [2.6, 6.8, 13.9, 29.3],    # w1
                      [0.7, 4.4, 9.7, 20.6]])    # w2
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, freqs.shape)
    amps = np.array([amp_u, amp_u, amp_w, amp_w]) / np.sqrt(freqs.shape[1])

    def channels(t):
        return amps * np.sin(freqs * t + phases).sum(axis=1)
    return (lambda t: channels(t)[:N_U]), (lambda t: channels(t)[N_U:])


def collect_data(A_a, B, D, e_u, e_w, duration, K0=None, N0=None):
    """沿一条不重置的连续轨迹采集式 (36)–(39) 所需的窗口积分。

    行为策略 u = −K⁰z + E1，w = N⁰z + E2。为得到与状态同等精度的积分，
    把 ∫zzᵀ、∫uzᵀ、∫wzᵀ 并入状态一起积分，再在窗口端点作差。
    """
    K0 = np.zeros((N_U, N_X)) if K0 is None else K0
    N0 = np.zeros((N_W, N_X)) if N0 is None else N0

    def rhs(t, y):
        z = y[:N_X]
        u = -K0 @ z + e_u(t)
        w = N0 @ z + e_w(t)
        return np.concatenate([A_a @ z + B @ u + D @ w, np.outer(z, z)[IU],
                               np.outer(u, z).ravel(), np.outer(w, z).ravel()])

    n_win = int(round(duration / T_WIN))
    t_grid = np.arange(n_win + 1) * T_WIN
    y0 = np.concatenate([Z0, np.zeros(N_P + (N_U + N_W) * N_X)])
    sol = solve_ivp(rhs, (0, t_grid[-1]), y0, method="DOP853", t_eval=t_grid,
                    rtol=1e-10, atol=1e-10, max_step=0.01)
    if not sol.success:
        raise RuntimeError(sol.message)
    Y = sol.y.T
    z = Y[:, :N_X]
    zbar = np.einsum("ki,kj->kij", z, z)[:, IU[0], IU[1]]
    d = np.diff(Y, axis=0)
    o1, o2 = N_X + N_P, N_X + N_P + N_U * N_X
    return {
        "t": t_grid, "z": z,
        "dzbar": np.diff(zbar, axis=0),                       # ϕ^zz
        "Izz": d[:, N_X:o1],                                  # ς^zz（上三角）
        "Szz": unpack_sym(d[:, N_X:o1]),                      # ∫ z zᵀ
        "Suz": d[:, o1:o2].reshape(-1, N_U, N_X),             # ∫ u zᵀ（ς^zu）
        "Swz": d[:, o2:].reshape(-1, N_W, N_X),               # ∫ w zᵀ（ς^zw）
    }


def offpolicy_learning(data, P0=None, max_iter=50, rcond=1e-10):
    """Algorithm 2：在同一批数据上迭代求解式 (40)–(43)。

    第 i 次迭代的回归方程（式 (25)）逐窗口为
      P̄ᵢ·Δz̄ − 2⟨K^{i+1}, R∫(u + Kⁱz)zᵀ⟩ − 2γ²⟨N^{i+1}, ∫(w − Nⁱz)zᵀ⟩ = −⟨Qⁱ, ∫zzᵀ⟩，
      Qⁱ = Q + KⁱᵀRKⁱ − γ²NⁱᵀNⁱ。
    满列秩（=68）时用列缩放最小二乘求解；满足 max|Pⁱ⁺¹ − Pⁱ| < δ 即停止。
    P⁰ 只用于第一次收敛判断（流程图中的输入 P⁰_α），K⁰ = N⁰ = 0。
    """
    n = data["dzbar"].shape[0]
    data_rank, _ = numerical_rank(np.hstack([data["Izz"], data["Suz"].reshape(n, -1),
                                             data["Swz"].reshape(n, -1)]), rcond)
    out = {"windows": n, "data_rank": data_rank, "rank_required": N_UNKNOWN,
           "converged": False, "P_hist": [], "K": None, "N": None, "P": None}
    if data_rank < N_UNKNOWN:
        return out

    K, N = np.zeros((N_U, N_X)), np.zeros((N_W, N_X))
    P_prev = 10.0 * np.eye(N_X) if P0 is None else P0
    out["P_hist"].append(P_prev)
    Szz, Suz, Swz = data["Szz"], data["Suz"], data["Swz"]
    for i in range(max_iter):
        Theta = np.hstack([data["dzbar"],
                           -2 * (R @ (Suz + K @ Szz)).reshape(n, -1),
                           -2 * GAMMA**2 * (Swz - N @ Szz).reshape(n, -1)])
        Qi = Q + K.T @ R @ K - GAMMA**2 * N.T @ N
        rhs = -np.einsum("ij,kij->k", Qi, Szz)
        rank, cond_ratio = numerical_rank(Theta, rcond)
        if rank < N_UNKNOWN:
            out["fail_reason"] = f"第 {i + 1} 次迭代回归矩阵秩 {rank} < {N_UNKNOWN}"
            break
        scale = np.linalg.norm(Theta, axis=0)
        x = np.linalg.lstsq(Theta / scale, rhs, rcond=None)[0] / scale
        P = vecs_inv(x[:N_P])
        K = x[N_P:N_P + N_U * N_X].reshape(N_U, N_X)
        N = x[N_P + N_U * N_X:].reshape(N_W, N_X)
        out["P_hist"].append(P)
        out["fit_residual"] = float(np.linalg.norm(Theta @ x - rhs) / np.linalg.norm(rhs))
        out["theta_sv_ratio"] = cond_ratio
        if np.abs(P - P_prev).max() < DELTA:
            out["converged"] = True
            break
        P_prev = P
    out.update(P=out["P_hist"][-1], K=K, N=N, iterations=len(out["P_hist"]) - 1)
    return out


# ============================================================================
# 4. 闭环仿真（图 5–7）与蒙特卡洛（图 8）
# ============================================================================
def road_profile(t):
    """第 IV 节的路面速度 [q̇I, q̇II]。"""
    return np.array([np.exp(-0.01 * t) * np.sin(100 * t) + 0.01 * np.sin(15 * t),
                     np.exp(-0.01 * t) * np.sin(10 * t) + 0.1 * np.sin(10 * t)])


def blended(z, Ks, Ns):
    """式 (6)：u = −Σ H_α K_α z，w = Σ H_α N_α z。z 可为 (8,) 或 (m, 8)；Ks 可为 (2,2,8) 或 (m,2,2,8)。"""
    H = fuzzy_weights(z[..., 0])
    Kb = H[0][..., None, None] * Ks[..., 0, :, :] + H[1][..., None, None] * Ks[..., 1, :, :]
    Nb = H[0][..., None, None] * Ns[..., 0, :, :] + H[1][..., None, None] * Ns[..., 1, :, :]
    return -np.einsum("...ij,...j->...i", Kb, z), np.einsum("...ij,...j->...i", Nb, z)


def closed_loop(A, B, D, Ks, Ns, t_end, dt, disturbance="worst"):
    """非线性系统 ż = Az + B(u + ρ(z)) + Dw。

    disturbance="worst"：w = Σ H_α N_α z（局部扰动反馈的模糊混合，未证明为全局最坏扰动）；
    disturbance="road"：w 为第 IV 节给出的路面速度。
    """
    def inputs(t, z):
        u, w = blended(z, Ks, Ns)
        if disturbance == "road":
            w = road_profile(t)
        return u, w

    def rhs(t, z):
        u, w = inputs(t, z)
        return A @ z + B @ (u + rho(z)) + D @ w

    t = np.arange(0, t_end + dt / 2, dt)
    sol = solve_ivp(rhs, (0, t_end), Z0, method="DOP853", t_eval=t,
                    rtol=1e-9, atol=1e-9, max_step=0.002 if disturbance == "road" else 0.01)
    if not sol.success:
        raise RuntimeError(sol.message)
    z = sol.y.T
    uw = [inputs(ti, zi) for ti, zi in zip(t, z)]
    return t, z, np.array([v[0] for v in uw]), np.array([v[1] for v in uw])


def monte_carlo(A, B, D, Ks, Ns, n_samples, sigma, horizon, seed):
    """图 8：随机扰动两条规则的控制增益 K（N 固定为学得值），积分累计代价

    J(t) = ∫₀ᵗ (zᵀQz + uᵀRu − γ²wᵀw) dτ，w = Σ H_α N_α z。
    第 0 组为 Algorithm 2 学得的增益，其余为 K + σ·标准正态（逐元素）。
    """
    rng = np.random.default_rng(seed)
    K_all = np.concatenate([Ks[None], Ks[None] + sigma * rng.standard_normal((n_samples,) + Ks.shape)])
    m = K_all.shape[0]
    N_all = np.broadcast_to(Ns, (m,) + Ns.shape)

    def rhs(t, y):
        Y = y.reshape(m, N_X + 1)
        z = Y[:, :N_X]
        u, w = blended(z, K_all, N_all)
        dz = z @ A.T + (u + rho(z)) @ B.T + w @ D.T
        dJ = np.einsum("mi,ij,mj->m", z, Q, z) + np.einsum("mi,ij,mj->m", u, R, u) \
            - GAMMA**2 * np.einsum("mi,mi->m", w, w)
        return np.hstack([dz, dJ[:, None]]).ravel()

    t = np.linspace(0, horizon, int(round(horizon / 0.1)) + 1)
    y0 = np.hstack([np.tile(Z0, (m, 1)), np.zeros((m, 1))]).ravel()
    sol = solve_ivp(rhs, (0, horizon), y0, method="DOP853", t_eval=t,
                    rtol=1e-9, atol=1e-9, max_step=0.01)
    if not sol.success:
        raise RuntimeError(sol.message)
    Y = sol.y.T.reshape(len(t), m, N_X + 1)
    return t, Y[:, :, N_X], np.linalg.norm(Y[-1, :, :N_X], axis=1), K_all


# ============================================================================
# 5. 绘图（图 3–8，坐标范围与论文一致）
# ============================================================================
def plot_fig3(out):
    z1 = np.linspace(-3, 3, 601)
    lo1, up1, lo2, up2 = membership_bounds(z1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(z1, lo1, "b", label=r"lower $h_1(z_1)$")
    ax[0].plot(z1, up1, "r", label=r"upper $\overline{h}_1(z_1)$")
    ax[1].plot(z1, lo2, "b", label=r"lower $h_2(z_1)$")
    ax[1].plot(z1, up2, "r", label=r"upper $\overline{h}_2(z_1)$")
    for a, loc in zip(ax, ["lower left", "upper right"]):
        a.set_xlabel(r"$z_1$")
        a.legend(loc=loc, fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle("Fig. 3  Membership functions")
    fig.tight_layout()
    fig.savefig(out / "fig3_membership.png", dpi=150)
    plt.close(fig)


def plot_fig4(out, learned, P_ref):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, res, Pr, name in zip(ax, learned, P_ref, ["P_1", "P_2"]):
        norms = [la.norm(P, 2) for P in res["P_hist"]]
        a.plot(range(len(norms)), norms, "b-o", ms=3, label=rf"$\|{name}\|$ (Algorithm 2)")
        a.axhline(la.norm(Pr, 2), color="gray", ls="--", lw=1, label="game ARE")
        a.set_xlabel("Number of iterations")
        a.set_ylim(8, 14)
        a.legend(fontsize=8, loc="lower right")
        a.grid(alpha=0.3)
    fig.suptitle(r"Fig. 4  $\|P_1\|$, $\|P_2\|$ by Algorithm 2 (iteration 0 = input $P^0=10I$)")
    fig.tight_layout()
    fig.savefig(out / "fig4_P_convergence.png", dpi=150)
    plt.close(fig)


def plot_fig5_7(out, t, z, u, w, t_show=3.5):
    k = t <= t_show + 1e-9
    colors = ["#a00000", "black", "orange", "blue", "magenta", "#800000", "cyan", "green"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i in range(8):
        ax.plot(t[k], z[k, i], color=colors[i], lw=1, label=rf"$x_{i + 1}$")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("x")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Fig. 5  System state")
    fig.tight_layout()
    fig.savefig(out / "fig5_states.png", dpi=150)
    plt.close(fig)
    for sig, name, fn in [(u, "u", "fig6_control.png"), (w, "w", "fig7_disturbance.png")]:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(t[k], sig[k, 0], "b", lw=1, label=rf"${name}_1$")
        ax.plot(t[k], sig[k, 1], "r", lw=1, label=rf"${name}_2$")
        ax.set_xlabel("Time/s")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title(f"Fig. {6 if name == 'u' else 7}  Trajectories of {name}(t)")
        fig.tight_layout()
        fig.savefig(out / fn, dpi=150)
        plt.close(fig)


def plot_fig8(out, t, J):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})
    ax[0].plot(t, J[:, 1:], color="b", lw=0.6)
    ax[0].plot(t, J[:, 0], color="r", lw=2, label="learned gains (Algorithm 2)")
    ax[0].set_xlabel("Time(s)")
    ax[0].set_ylabel(r"$J(0,t)$")
    ax[0].legend(loc="lower right")
    ax[0].grid(alpha=0.3)
    gap = J[-1, 1:] - J[-1, 0]
    ax[1].hist(gap, bins=15, color="b", alpha=0.7)
    ax[1].axvline(0, color="r", lw=2)
    ax[1].set_xlabel(rf"$J_{{\rm random}}({t[-1]:g}) - J_{{\rm learned}}({t[-1]:g})$")
    ax[1].set_ylabel("count")
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"Fig. 8  Monte Carlo ({J.shape[1] - 1} random control gains)")
    fig.tight_layout()
    fig.savefig(out / "fig8_monte_carlo.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 6. 主流程
# ============================================================================
def fmt(M):
    return np.array2string(np.asarray(M), precision=4, suppress_small=True, max_line_width=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["corrected", "paper_a53"], default="corrected")
    ap.add_argument("--duration", type=float, default=20.0, help="Algorithm 2 数据采集时长 s")
    ap.add_argument("--amp-u", type=float, default=500.0, help="补充探测：控制通道幅度 N")
    ap.add_argument("--amp-w", type=float, default=1.0, help="补充探测：扰动通道幅度 m/s")
    ap.add_argument("--mc-samples", type=int, default=50)
    ap.add_argument("--mc-sigma", type=float, default=0.05, help="图 8 中 K 的逐元素随机扰动标准差")
    ap.add_argument("--mc-horizon", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")
    args = ap.parse_args()
    # 先验证输入再创建输出，避免无效参数触发不完整实验或静默改变窗口时长。
    if not np.isfinite(args.duration) or args.duration <= 0:
        ap.error("--duration 必须为有限正数")
    if not np.isclose(args.duration / T_WIN, round(args.duration / T_WIN), rtol=0, atol=1e-9):
        ap.error(f"--duration 必须是积分窗口 {T_WIN} 秒的整数倍")
    if not np.isfinite(args.mc_horizon) or args.mc_horizon < 0.1:
        ap.error("--mc-horizon 必须为不小于 0.1 的有限秒数")
    if args.mc_samples < 1:
        ap.error("--mc-samples 必须至少为 1")
    for name, value in [("--amp-u", args.amp_u), ("--amp-w", args.amp_w), ("--mc-sigma", args.mc_sigma)]:
        if not np.isfinite(value) or value < 0:
            ap.error(f"{name} 必须为有限非负数")
    out = args.out / args.model
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    summary = {"model": args.model, "args": {k: str(v) for k, v in vars(args).items()}}
    # 默认输出记录为项目相对路径，避免结果文件绑定本机目录。
    if args.out == Path(__file__).resolve().parent / "output":
        summary["args"]["out"] = "output"
    # 先标记未完成；中途失败时旧报告不能继续表示本次成功。
    summary["execution_status"] = "incomplete"
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- 模型 ----------
    A, B, D = mechanical_matrices(args.model)
    A_loc = local_matrices(A, B)
    eq_err, sum_err = check_fuzzy_equivalence(A, B, A_loc)
    print(f"[模型] {args.model}；模糊混合与非线性模型最大偏差 {eq_err:.2e}，|H1+H2−1| ≤ {sum_err:.1e}")
    print(f"       局部开环最大实部特征值：{[round(float(la.eigvals(Aa).real.max()), 4) for Aa in A_loc]}")
    summary["fuzzy_equivalence_error"] = eq_err

    # ---------- Algorithm 1 与参考解 ----------
    P_ref, pi_norms = [], []
    for a, Aa in enumerate(A_loc):
        P_star, res = game_are(Aa, B, D)
        P_pi, norms = model_based_pi(Aa, B, D)
        P_ref.append(P_star)
        pi_norms.append(norms)
        print(f"[Algorithm 1] 规则{a + 1}：{len(norms)} 次迭代，‖P‖₂ 序列 {np.round(norms, 4).tolist()}，"
              f"与 ARE 解差 {np.abs(P_pi - P_star).max():.1e}（ARE 残差 {res:.1e}）")
    summary["algorithm1_norms"] = pi_norms

    # ---------- Algorithm 2：论文原探测，仅做秩检验 ----------
    e_u, e_w = paper_probes()
    summary["paper_probe_rank"] = []
    for a, Aa in enumerate(A_loc):
        res = offpolicy_learning(collect_data(Aa, B, D, e_u, e_w, args.duration))
        summary["paper_probe_rank"].append(res["data_rank"])
        print(f"[Algorithm 2 / 论文探测] 规则{a + 1}：数据秩 {res['data_rank']}/{N_UNKNOWN}"
              f" → {'可学习' if res['data_rank'] == N_UNKNOWN else '不满足满列秩，无法辨识'}")

    # ---------- Algorithm 2：独立多频探测，学习 P、K、N ----------
    e_u, e_w = multisine_probes(args.amp_u, args.amp_w, args.seed)
    learned = []
    for a, Aa in enumerate(A_loc):
        res = offpolicy_learning(collect_data(Aa, B, D, e_u, e_w, args.duration))
        if not res["converged"]:
            raise SystemExit(f"规则{a + 1} 学习失败：{res.get('fail_reason', '未收敛')}，可增大 --duration")
        err = np.abs(res["P"] - P_ref[a]).max()
        res["P_err"] = float(err)
        learned.append(res)
        print(f"[Algorithm 2 / 多频探测] 规则{a + 1}：{res['windows']} 个窗口，数据秩 {res['data_rank']}/{N_UNKNOWN}，"
              f"{res['iterations']} 次迭代收敛；‖P‖₂ 序列 {np.round([la.norm(P, 2) for P in res['P_hist']], 4).tolist()}；"
              f"与 ARE 解最大误差 {err:.2e}，相对拟合残差 {res['fit_residual']:.1e}")
    Ks = np.array([r["K"] for r in learned])
    Ns = np.array([r["N"] for r in learned])
    for a in range(2):
        print(f"\n规则{a + 1} 学得 K*：\n{fmt(Ks[a])}\n论文打印 K*：\n{fmt(PAPER_K[a])}")
        print(f"规则{a + 1} 学得 N*：\n{fmt(Ns[a])}\n论文打印 N*：\n{fmt(PAPER_N[a])}")
    dK = float(np.abs(Ks - PAPER_K).max())
    dN = float(np.abs(Ns - PAPER_N).max())
    print(f"\n与论文打印增益的最大绝对差：K {dK:.4f}，N {dN:.4f}")
    summary["algorithm2"] = [{
        "windows": r["windows"], "data_rank": r["data_rank"], "iterations": r["iterations"],
        "P_norm_history": [float(la.norm(P, 2)) for P in r["P_hist"]],
        "max_abs_error_vs_ARE": r["P_err"], "fit_residual": r["fit_residual"],
        "K": r["K"].tolist(), "N": r["N"].tolist()} for r in learned]
    summary["max_abs_diff_vs_paper"] = {"K": dK, "N": dN}
    for a in range(2):
        cond26 = Q + Ks[a].T @ R @ Ks[a] - GAMMA**2 * Ns[a].T @ Ns[a]
        summary.setdefault("theorem3_cond26_min_eig", []).append(float(la.eigvalsh(cond26).min()))
    np.savez(out / "learned_gains.npz", K=Ks, N=Ns, P=np.array([r["P"] for r in learned]),
             P_ref=np.array(P_ref), A_loc=np.array(A_loc), B=B, D=D)

    # ---------- 闭环仿真（图 5–7） ----------
    t, z, u, w = closed_loop(A, B, D, Ks, Ns, t_end=10.0, dt=0.001, disturbance="worst")
    print(f"\n[闭环] 局部扰动反馈 w=ΣH N z：‖z(0)‖={la.norm(z[0]):.3f}，‖z(3.5)‖={la.norm(z[t <= 3.5][-1]):.3e}，"
          f"‖z(10)‖={la.norm(z[-1]):.3e}；|u|max={np.abs(u).max():.4f}，|w|max={np.abs(w).max():.2f}")
    summary["closed_loop"] = {"z_norm_end": float(la.norm(z[-1])), "u_absmax": float(np.abs(u).max()),
                              "w_absmax": float(np.abs(w).max())}

    # ---------- 蒙特卡洛（图 8） ----------
    tm, J, z_end, _ = monte_carlo(A, B, D, Ks, Ns, args.mc_samples, args.mc_sigma, args.mc_horizon, args.seed)
    gap = J[-1, 1:] - J[-1, 0]
    print(f"[蒙特卡洛] {args.mc_samples} 组随机 K（σ={args.mc_sigma}）：J_learned({tm[-1]:g})={J[-1, 0]:.4f}；"
          f"随机组与之差 min {gap.min():.4f}、max {gap.max():.4f}；比学得增益代价更低的组数 {(gap < 0).sum()}；"
          f"末端最大 ‖z‖ {z_end.max():.1e}")
    summary["monte_carlo"] = {"J_learned": float(J[-1, 0]), "gap_min": float(gap.min()),
                              "gap_max": float(gap.max()), "n_better_than_learned": int((gap < 0).sum())}

    # ---------- 图 ----------
    plot_fig3(out)
    plot_fig4(out, learned, P_ref)
    plot_fig5_7(out, t, z, u, w)
    plot_fig8(out, tm, J)
    summary["runtime_s"] = round(time.time() - t_start, 1)
    summary["execution_status"] = "completed"
    temporary = out / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(out / "summary.json")
    print(f"\n完成，用时 {summary['runtime_s']} s。图与数据保存在 {out}")


if __name__ == "__main__":
    main()
