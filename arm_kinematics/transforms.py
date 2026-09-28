"""三维旋转与齐次变换的基础数学工具（核心运动学计算均在工程内完成）。

约定：
    - 旋转矩阵为 3x3 正交矩阵，齐次变换为 4x4 矩阵 ``[R p; 0 0 0 1]``。
    - 固定角 RPY 与旋转向量均采用 URDF/ROS 约定（见 README「坐标系与变换顺序」）。
"""

from __future__ import annotations

import numpy as np


def _as_vec3(value, name: str) -> np.ndarray:
    """把输入转换成长度为 3 的 float ndarray。"""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} 必须是长度 3 的序列，实际形状为 {arr.shape}")
    return arr


def rpy_to_matrix(rpy) -> np.ndarray:
    """固定角 (roll, pitch, yaw) 转 3x3 旋转矩阵。

    采用 URDF 约定 ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``，
    即先绕固定系 x 轴转 roll，再绕固定系 y 轴转 pitch，最后绕固定系 z 轴转 yaw。
    """
    r, p, y = _as_vec3(rpy, "rpy")
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def make_transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    """由平移 xyz 与固定角 rpy 构造 4x4 齐次变换。"""
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(rpy)
    t[:3, 3] = _as_vec3(xyz, "xyz")
    return t


def invert_transform(t: np.ndarray) -> np.ndarray:
    """齐次变换的逆 ``T^-1``（利用正交性，无需通用求逆）。"""
    t = np.asarray(t, dtype=float)
    r = t[:3, :3]
    p = t[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ p
    return inv


def rotation_to_axis_angle(r: np.ndarray) -> np.ndarray:
    """旋转矩阵转旋转向量（轴 × 角度），角度范围 ``(-pi, pi]``。

    使用稳健公式 ``sin(theta) 由反对称部分提取``，并在 theta≈pi 时
    由 ``(R+I)/2`` 的列向量恢复转轴，避免近 pi 处数值发散。
    """
    r = np.asarray(r, dtype=float)
    cos_theta = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # 小角度：直接用反对称部分提取轴，避免除以接近零的 sin(theta)
    if theta < 1e-9:
        return 0.5 * np.array([r[2, 1] - r[1, 2],
                               r[0, 2] - r[2, 0],
                               r[1, 0] - r[0, 1]])

    if np.pi - theta < 1e-7:
        # theta ≈ pi：轴向量与 R+I 的非零列成正比（其长度为 2|axis_component|）
        m = (r + np.eye(3)) / 2.0
        col_norms = np.linalg.norm(m, axis=0)
        axis = m[:, int(np.argmax(col_norms))]
        n = np.linalg.norm(axis)
        axis = axis / n if n > 0 else np.array([1.0, 0.0, 0.0])
        # 用 R @ axis == axis 选择符号，使轴向满足右手定则下的一致约定
        if axis[0] < 0 or (axis[0] == 0 and axis[1] < 0) or \
           (axis[0] == 0 and axis[1] == 0 and axis[2] < 0):
            axis = -axis
        return axis * theta

    axis = np.array([r[2, 1] - r[1, 2],
                     r[0, 2] - r[2, 0],
                     r[1, 0] - r[0, 1]]) / (2.0 * np.sin(theta))
    return axis * theta


def rotation_error(r_current: np.ndarray, r_desired: np.ndarray) -> np.ndarray:
    """旋转误差向量。

    返回 ``R_err = R_desired @ R_current^T``（世界/基座系下「当前姿态转到目标姿态」
    的旋转）对应的旋转向量，与基座系几何雅可比的角速度部分配套使用。
    """
    r_err = r_desired @ np.asarray(r_current, dtype=float).T
    return rotation_to_axis_angle(r_err)


def pose_error(t_current: np.ndarray, t_desired: np.ndarray) -> np.ndarray:
    """6 维位姿误差 ``[位置误差(3); 旋转误差(3)]``，均在基座系下表示。"""
    e_pos = np.asarray(t_desired[:3, 3], dtype=float) - \
        np.asarray(t_current[:3, 3], dtype=float)
    e_rot = rotation_error(t_current[:3, :3], t_desired[:3, :3])
    return np.concatenate([e_pos, e_rot])
