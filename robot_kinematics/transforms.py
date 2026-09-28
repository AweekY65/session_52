"""齐次变换与旋转数学工具（NumPy 实现，核心计算全部在工程内完成）。

约定
----
- 旋转矩阵 R 为 3x3 正交矩阵，满足 det(R) = +1。
- 齐次变换 T 为 4x4 矩阵 ``[R p; 0 1]``，表示「子坐标系相对于父坐标系」的位姿：
  父系下点 ``p_parent = R @ p_child + t``。
- URDF 的 ``origin rpy="r p y"`` 使用固定轴（外旋）XYZ 顺序，
  即 ``R = Rz(y) @ Ry(p) @ Rx(r)``。
- 旋转的最小表示使用旋转向量（axis-angle，3 维），
  ``theta = |v|``，轴为 ``v / theta``。
"""

from __future__ import annotations

import numpy as np


def _as_3vec(value, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float).reshape(-1)
    if vec.size != 3:
        raise ValueError(f"{name} 必须是长度为 3 的向量，实际长度 {vec.size}")
    return vec


def rx_matrix(angle: float) -> np.ndarray:
    """绕固定 X 轴旋转 ``angle`` 的旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ry_matrix(angle: float) -> np.ndarray:
    """绕固定 Y 轴旋转 ``angle`` 的旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rz_matrix(angle: float) -> np.ndarray:
    """绕固定 Z 轴旋转 ``angle`` 的旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF 固定轴 RPY（外旋 XYZ）转旋转矩阵：``R = Rz(yaw) Ry(pitch) Rx(roll)``。"""
    return rz_matrix(yaw) @ ry_matrix(pitch) @ rx_matrix(roll)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """由 3x3 旋转与 3 维平移组装 4x4 齐次变换。"""
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3):
        raise ValueError(f"旋转矩阵必须是 3x3，实际形状 {r.shape}")
    t = _as_3vec(translation, "translation")
    tform = np.eye(4)
    tform[:3, :3] = r
    tform[:3, 3] = t
    return tform


def invert_transform(tform: np.ndarray) -> np.ndarray:
    """齐次变换求逆：刚体变换的逆 ``[R^T, -R^T t; 0 1]``。"""
    t = np.asarray(tform, dtype=float)
    if t.shape != (4, 4):
        raise ValueError(f"齐次变换必须是 4x4，实际形状 {t.shape}")
    r = t[:3, :3]
    p = t[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ p
    return inv


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """旋转向量 -> 旋转矩阵（Rodrigues 公式）；零向量返回单位阵。"""
    v = _as_3vec(rotvec, "rotvec")
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    axis = v / angle
    x, y, z = axis
    c, s, c1 = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ]
    )


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 旋转向量（SO(3) 对数映射）。

    返回的向量模长在 ``[0, pi]``，是两个姿态间「最短路径」的最小表示。
    数值上通过对角度截断到 arccos 的定义域 [-1, 1] 保证稳定性。
    """
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3):
        raise ValueError(f"旋转矩阵必须是 3x3，实际形状 {r.shape}")
    cos_theta = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_theta))
    if angle < 1e-10:
        return np.zeros(3)
    if np.pi - angle < 1e-10:
        # 接近 180°：用 R+I 的列恢复转轴，避免 sin(angle) -> 0。
        rr = r + np.eye(3)
        # 取范数最大的列。
        col = rr[:, int(np.argmax(np.linalg.norm(rr, axis=0)))]
        axis = col / np.linalg.norm(col)
        return axis * angle
    axis = np.array(
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]
    ) / (2.0 * np.sin(angle))
    return axis * angle


def rotation_error(rotation_target: np.ndarray, rotation_current: np.ndarray) -> np.ndarray:
    """从当前姿态到目标姿态的旋转误差向量（在世界系下表示）。

    ``R_err = R_target @ R_current^T``，其旋转向量就是姿态误差，
    与几何雅可比的角速度部分（世界系）配套使用。
    """
    return matrix_to_rotvec(rotation_target @ rotation_current.T)


def is_valid_rotation(rotation: np.ndarray, tol: float = 1e-6) -> bool:
    """检查 3x3 矩阵是否为合法旋转矩阵。"""
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3):
        return False
    if abs(np.linalg.det(r) - 1.0) > tol:
        return False
    if not np.allclose(r @ r.T, np.eye(3), atol=tol):
        return False
    return True
