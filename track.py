# -*- coding: utf-8 -*-
"""
闭环赛道生成器（Closed Track）
==============================

用参数方程生成一条平滑的闭环赛道中心线：
    r(theta) = R0 * (1 + a * cos(m * theta))

其中 R0 是基准半径，a 是"起伏幅度"，m 是"花瓣数"。a > 0 时赛道有
弯曲变化的路段（有弯道、有大弯），用来考验 MPC 的跟踪能力；a = 0 时
退化为圆形赛道（恒定曲率），是最简单的测试用例。

由于 theta 均匀采样时线速度不均匀，这里按弧长重新均匀采样，
保证相邻路点间距一致（spacing 米一个点），并计算每个路点的航向角 psi。

查询接口：
    nearest(p, hint=None)  —— 最近路点（可选传入上一步进度做局部窗口搜索）
    lateral_error(p, ...)  —— 相对中心线的横向误差（带符号，左正右负）
"""

import numpy as np


class Track:
    """一条闭环赛道：中心线路点 + 每个路点的航向 + 常用查询方法。"""

    def __init__(self, P, psi, spacing):
        # P: (M, 2) 中心线路点坐标（已按弧长均匀重采样，闭环）
        # psi: (M,)   每个路点的航向角（车头切向方向，已 unwrap 保证连续）
        # spacing: 相邻路点的弧长间隔 [m]
        self.P = P
        self.psi = psi
        self.spacing = spacing
        self.n = len(P)

        # 切线方向的单位向量（用于算侧向偏移 / 画赛道边界）
        tangents = np.roll(P, -1, axis=0) - P          # 每个点指向下一个点
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        self.tangent = tangents / np.maximum(norms, 1e-9)

        self.length = self.n * spacing                 # 赛道总长 [m]

    # ------------------------------------------------------------------
    def nearest(self, p, hint=None, window=60):
        """
        返回赛道中心线上离点 p 最近的路点下标。

        传入 hint（上一步的进度下标）时只在 [hint-window, hint+window]
        的环形窗口内搜索，把每步 O(M) 扫描降到 O(window)；
        若最优解落在窗口边界上（说明车已跑出窗口，例如大偏差恢复场景），
        自动退化为全量搜索以保证正确性。
        """
        p = np.asarray(p, dtype=float)
        n = self.n

        if hint is None:
            d = self.P - p
            return int(np.argmin(np.sum(d * d, axis=1)))

        idx = (np.arange(hint - window, hint + window + 1)) % n
        d = self.P[idx] - p
        j = int(np.argmin(np.sum(d * d, axis=1)))
        if j == 0 or j == idx.size - 1:      # 命中窗口边界 → 回退全量搜索
            d_all = self.P - p
            return int(np.argmin(np.sum(d_all * d_all, axis=1)))
        return int(idx[j])

    def lateral_error(self, p, idx=None, hint=None):
        """
        相对中心线的横向误差（左正右负）与所用路点下标。

        e_lat = n·(p - P_idx)，其中 n 是该路点的单位法向（左法向）。
        """
        if idx is None:
            idx = self.nearest(p, hint=hint)
        t = self.tangent[idx]                      # 单位切向
        nvec = np.array([-t[1], t[0]])             # 左法向
        e_lat = float(np.dot(nvec, np.asarray(p, dtype=float) - self.P[idx]))
        return e_lat, idx

    # ------------------------------------------------------------------
    def lane_boundaries(self, lane_width):
        """返回左右边界（中心线沿法向平移 lane_width/2）。"""
        nx = -self.tangent[:, 1]
        ny = self.tangent[:, 0]
        half = lane_width / 2.0
        left = self.P + half * np.column_stack([nx, ny])
        right = self.P - half * np.column_stack([nx, ny])
        return left, right


def make_track(R0=32.0, ripple=0.18, harmonic=3,
               n_raw=3000, spacing=0.5):
    """
    生成闭环赛道。

    参数:
        R0      : 基准半径 [m]
        ripple  : 半径起伏幅度（0~0.3 左右，越大弯道越多越急）
        harmonic: 起伏的"花瓣数"（3 表示 3 个急弯 3 个缓弯交替）
        n_raw   : 参数采样点数（越多越平滑）
        spacing : 重采样后相邻路点的弧长间隔 [m]

    返回:
        Track 对象
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_raw, endpoint=False)
    r = R0 * (1.0 + ripple * np.cos(harmonic * theta))
    x_raw = r * np.cos(theta)
    y_raw = r * np.sin(theta)

    # ---- 先把它当作一条"打开的多段线"算累积弧长 ----
    loop = np.column_stack([x_raw, y_raw])
    loop_closed = np.vstack([loop, loop[0]])
    seg_len = np.linalg.norm(np.diff(loop_closed, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])   # (n_raw+1,)

    # ---- 按固定弧长间隔重采样（保证路点间距一致）----
    m = int(np.floor(s_cum[-1] / spacing))
    s_target = np.arange(m, dtype=float) * spacing
    P = np.column_stack([
        np.interp(s_target, s_cum, loop_closed[:, 0]),
        np.interp(s_target, s_cum, loop_closed[:, 1]),
    ])

    # ---- 每个路点的航向：切向方向角，unwrap 保证连续 ----
    seg = np.roll(P, -1, axis=0) - P
    psi_raw = np.arctan2(seg[:, 1], seg[:, 0])
    psi = np.unwrap(psi_raw)

    return Track(P, psi, spacing)
