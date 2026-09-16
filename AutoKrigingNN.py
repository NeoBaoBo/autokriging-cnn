"""AutoKrigingNN：3D CNN 模型 + 张量拟合工具（预测 64 个方向的 range）

- 模型：四层 3D CNN，输入 [2, 32, 32, 32]（IDW 体素 + 掩膜），输出 N_DIR 个方向的 range
- 张量拟合：64 方向 range -> 正定张量 -> 特征分解 -> 三轴长 + 方向
"""
import numpy as np

import torch
import torch.nn as nn

N_DIR = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fibonacci_sphere(n):
    """斐波那契球面采样：n 个近似均匀分布的 3D 方向向量"""
    pts = []
    phi = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z * z)
        theta = phi * i
        pts.append([r * np.cos(theta), r * np.sin(theta), z])
    return np.array(pts, dtype=np.float64)


class AutoKrigingNN(nn.Module):
    """四层 3D CNN，输出 N_DIR 个方向的变程 range"""

    def __init__(self, out_dim=N_DIR):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.head(self.features(x))


def fit_tensor_psd(directions, ranges):
    """正定约束张量拟合：D = L L^T（Cholesky 参数化），保证特征值恒正。

    替代无约束最小二乘：当预测 range 噪声大、不构成合法椭球时，
    无约束最小二乘会得到负特征值（对应轴长虚数），本函数强制正定。
    """
    from scipy.optimize import least_squares

    U = np.asarray(directions, dtype=np.float64)
    b = 1.0 / (np.asarray(ranges, dtype=np.float64) ** 2)

    # 用无约束 lstsq 解做初值
    A = np.array([[ux*ux, uy*uy, uz*uz, 2*ux*uy, 2*ux*uz, 2*uy*uz] for ux, uy, uz in U])
    d = np.linalg.lstsq(A, b, rcond=None)[0]
    D0 = np.array([[d[0], d[3], d[4]], [d[3], d[1], d[5]], [d[4], d[5], d[2]]])
    w, V = np.linalg.eigh(D0)
    w = np.abs(w) + 1e-6
    D_psd = (V * w) @ V.T
    L0 = np.linalg.cholesky(D_psd + np.eye(3) * 1e-8)

    p0 = np.array([
        np.log(max(L0[0, 0], 1e-8)), L0[1, 0], np.log(max(L0[1, 1], 1e-8)),
        L0[2, 0], L0[2, 1], np.log(max(L0[2, 2], 1e-8)),
    ])

    def unpack(p):
        return np.array([[np.exp(p[0]), 0, 0],
                         [p[1], np.exp(p[2]), 0],
                         [p[3], p[4], np.exp(p[5])]])

    def residual(p):
        L = unpack(p)
        LtU = U @ L  # 每行 = L^T u
        quad = np.einsum('ij,ij->i', LtU, LtU)
        return quad - b

    res = least_squares(residual, p0, method='lm', max_nfev=20000)
    L = unpack(res.x)
    return L @ L.T


def tensor_to_axes_angles(D):
    """从正定张量取轴长与方向（特征值恒正，无需裁剪）"""
    eigvals, eigvecs = np.linalg.eigh(D)
    order = np.argsort(eigvals)  # 从小到大 = 轴长从大到小
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    axes = 1.0 / np.sqrt(np.maximum(eigvals, 1e-16))
    return axes, eigvecs


def angle_between(v1, v2):
    """两个方向向量间的夹角（度，主轴无向，取绝对值）"""
    cos = np.clip(np.abs(np.dot(v1, v2)), 0, 1)
    return np.degrees(np.arccos(cos))
