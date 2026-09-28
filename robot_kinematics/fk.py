"""正向运动学（Forward Kinematics）。

给定可动关节值 ``q``（顺序与 ``chain.movable_joints`` 一致，fixed joint 不出现），
计算链上每个 link 以及末端执行器（tip link）相对 base 的齐次变换矩阵。

变换沿链按 ``T_base->tip = T_1 T_2 ... T_n`` 顺序相乘，
其中 ``T_i = origin_i @ Motion_i(axis_i, q_i)``。
"""

from __future__ import annotations

import numpy as np

from .chain import KinematicChain


def fk_all(chain: KinematicChain, q) -> dict[str, np.ndarray]:
    """返回 ``{link_name: 4x4 位姿}``，包含 base link（单位阵）到 tip link。"""
    link_poses, _, _ = chain.accumulate_transforms(q)
    return {name: pose for name, pose in zip(chain.link_names, link_poses)}


def forward_kinematics(chain: KinematicChain, q, link_name: str | None = None) -> np.ndarray:
    """计算指定 link（默认末端 tip link）相对 base 的 4x4 齐次变换。"""
    link_poses, _, _ = chain.accumulate_transforms(q)
    target = chain.tip_link if link_name is None else link_name
    if target not in chain.link_names:
        raise KeyError(f"link '{target}' 不在运动链 {chain.base_link}->{chain.tip_link} 中")
    return link_poses[chain.link_names.index(target)].copy()
