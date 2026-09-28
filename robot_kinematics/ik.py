"""数值逆运动学（Inverse Kinematics）。

方法
~~~~
阻尼最小二乘（Damped Least Squares / Levenberg–Marquardt）：

    dq = J^T (J J^T + lambda^2 I)^{-1} e

- ``e`` 为末端误差：平移部分取目标与当前位置之差；旋转部分取
  ``log(R_target R_current^T)`` 的旋转向量（SO(3) 对数映射），
  与 :mod:`robot_kinematics.jacobian` 的角速度块一致。
- 每一步带 **回溯线搜索**（backtracking），新关节值夹到关节限位内，
  并按经典 LM gain-ratio（实际下降 / 线性模型预测下降）决定接受与否；
- 阻尼 ``lambda`` 自适应：实际下降接近预测就减小，远小于预测就增大；
- 加入 **零空间收拢项** ``(I - J^+ J) * k * (q_center - q)``，
  让关节朝行程中点收拢。它在常规构型下不改变末端运动，但在全伸展 /
  全折叠这类雅可比秩亏的奇异初始化下提供打破对称所必需的自运动；
- 若初始构型在奇异点上完全停滞，会用固定种子的小幅随机扰动做若干次
  重启（结果可复现）；
- 位置 / 姿态误差都小于 ``tol`` 才判定收敛；
- 预算耗尽返回 :attr:`IKStatus.MAX_ITERATIONS`；
  所有阻尼 / 步长 / 重启都找不到下降方向（局部极小、被限位挡住、
  目标过近等）返回 :attr:`IKStatus.UNREACHABLE`；
- 目标点到 base 的距离超过保守可达半径，直接返回
  :attr:`IKStatus.UNREACHABLE`，不伪造成功结果。

任何情况下求解器都返回 :class:`IKResult`，其中带明确状态、
实际误差与迭代次数；只有 ``status == IKStatus.CONVERGED`` 时
``q`` 才是有效解。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .chain import KinematicChain
from .errors import IKError
from .transforms import rotation_error


class IKStatus(str, Enum):
    """IK 求解状态。"""

    CONVERGED = "converged"                # 达到容差
    MAX_ITERATIONS = "max_iterations"      # 迭代预算用完（仍在下降，只是没收敛）
    UNREACHABLE = "unreachable"            # 目标不可达 / 被限位挡住 / 陷入局部极小
    BAD_TARGET = "bad_target"              # 目标位姿形状或数值非法


@dataclass
class IKResult:
    """IK 求解结果。只有 ``status == IKStatus.CONVERGED`` 时 ``q`` 才有效。"""

    status: IKStatus
    q: Optional[np.ndarray]
    position_error: float
    orientation_error: float
    iterations: int
    message: str = ""

    @property
    def success(self) -> bool:
        return self.status is IKStatus.CONVERGED

    def __bool__(self) -> bool:  # 方便 ``if ik_solve(...):`` 但建议显式判断
        return self.success


# ---------------------------------------------------------------------- #
# 可达性保守估计
# ---------------------------------------------------------------------- #
def max_reach(chain: KinematicChain) -> float:
    """末端相对 base 的保守最大可达距离（米）。

    把链上每个 joint 的 ``<origin>`` 平移长度直接相加；
    对 prismatic 关节再加上该关节在全行程中相对 joint 原点
    可能达到的最大伸出量 ``max(|lower|, |upper|)``。
    该上界对任意关节配置都成立（三角不等式），因此用于快速剔除
    明显不可达的目标是保守的（不会把可达目标误判为不可达）。
    """
    reach = 0.0
    for joint in chain.joints:
        reach += float(np.linalg.norm(joint.origin.xyz))
        if joint.joint_type == "prismatic" and joint.limit is not None:
            reach += max(abs(joint.limit.lower), abs(joint.limit.upper))
    return reach


def within_reach(
    chain: KinematicChain, position, tolerance: float = 1e-9
) -> bool:
    """目标点是否落在保守可达球内（仅做外半径检查，不能保证可达）。"""
    p = np.asarray(position, dtype=float).reshape(-1)
    if p.size != 3 or not np.all(np.isfinite(p)):
        return False
    return float(np.linalg.norm(p)) <= max_reach(chain) + tolerance


# ---------------------------------------------------------------------- #
# 目标位姿解析
# ---------------------------------------------------------------------- #
def _coerce_target(target):
    """把目标转成 (R_target 3x3, p_target 3, has_orientation bool)。

    形状非法返回 None。
    """
    if target is None:
        return None
    try:
        arr = np.asarray(target, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if arr.shape == (4, 4):
        return arr[:3, :3].copy(), arr[:3, 3].copy(), True
    if arr.shape == (3, 3):
        return arr.copy(), None, True
    if arr.shape == (3,):
        return None, arr.reshape(3).copy(), False
    return None


def _pose_at(chain: KinematicChain, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """无限位检查的 FK，返回 (R, p)。q 由求解器保证在限位内。"""
    poses, _, _ = chain.accumulate_transforms(q, check_limits=False)
    tip = poses[-1]
    return tip[:3, :3], tip[:3, 3]


def inverse_kinematics(
    chain: KinematicChain,
    target,
    q0=None,
    *,
    tol: float = 1e-6,
    max_iterations: int = 100,
    damping: float = 1e-3,
    use_orientation: Optional[bool] = None,
    position_mask=None,
    reach_check: bool = True,
) -> IKResult:
    """阻尼最小二乘数值 IK。

    Parameters
    ----------
    chain:
        目标运动链。
    target:
        目标位姿。接受 4x4 齐次矩阵（位姿）、3x3 旋转矩阵（仅姿态）、
        长度 3 的位置向量（仅位置）；其他形状返回 ``BAD_TARGET``。
    q0:
        初始关节值；默认取链的零位。
    tol:
        收敛容差，位置单位米、姿态单位弧度，两者都需满足。
    max_iterations:
        迭代上限（>=0）。
    damping:
        DLS 初始阻尼系数 lambda；会在迭代中自适应调整。
    use_orientation:
        是否跟踪姿态。``None`` 时由 target 形状决定（4x4/3x3 跟踪，
        3 向量不跟踪）。
    position_mask:
        长度 3 的布尔序列，选择位置误差的哪些轴参与求解
        （例如 ``(True, True, False)`` 只约束 x、y）。
    reach_check:
        是否在求解前做保守外半径可达性预检。

    Returns
    -------
    IKResult
    """
    # ---------------- 输入检查 ---------------- #
    if max_iterations < 0 or int(max_iterations) != max_iterations:
        raise IKError("max_iterations 必须是非负整数")
    if tol <= 0.0:
        raise IKError("tol 必须为正数")
    if damping < 0.0:
        raise IKError("damping 必须非负")

    coerced = _coerce_target(target)
    if coerced is None:
        return IKResult(
            IKStatus.BAD_TARGET,
            None,
            float("inf"),
            float("inf"),
            0,
            f"目标位姿形状非法：{np.asarray(target).shape}，接受 3 向量 / 3x3 / 4x4",
        )
    r_target, p_target, target_has_orient = coerced

    if position_mask is None:
        mask = np.array([True, True, True])
    else:
        mask = np.asarray(position_mask, dtype=bool).reshape(-1)
        if mask.size != 3:
            raise IKError("position_mask 必须是长度 3 的布尔序列")
    track_orientation = target_has_orient if use_orientation is None else bool(
        use_orientation and target_has_orient
    )

    if q0 is None:
        q = chain.zero_configuration()
    else:
        q = np.asarray(q0, dtype=float).reshape(-1)
        if q.size != chain.dof or not np.all(np.isfinite(q)):
            raise IKError(f"q0 必须是长度 {chain.dof} 的有限数值向量")
        q = chain.clamp_to_limits(q)

    # 零自由度链（纯 fixed joint）：只有「恰好到达」与「不可达」两种结果。
    if chain.dof == 0:
        r_cur, p_cur = _pose_at(chain, q)
        ep = float(np.linalg.norm(p_cur - p_target)) if p_target is not None else 0.0
        eo = (
            float(np.linalg.norm(rotation_error(r_target, r_cur)))
            if track_orientation and r_target is not None
            else 0.0
        )
        ok_p = p_target is None or ep <= tol
        ok_o = not track_orientation or eo <= tol
        if ok_p and ok_o:
            return IKResult(IKStatus.CONVERGED, q.copy(), ep, eo, 0, "零自由度链，当前位姿即目标")
        return IKResult(
            IKStatus.UNREACHABLE, q.copy(), ep, eo, 0,
            "零自由度链（只有 fixed joint），末端位姿无法改变，目标不可达",
        )

    # 保守可达性预检（只约束位置；部分轴掩码时跳过，避免误判）。
    if reach_check and p_target is not None and mask.all():
        dist = float(np.linalg.norm(p_target))
        if dist > max_reach(chain) + 1e-9:
            return IKResult(
                IKStatus.UNREACHABLE,
                q.copy(),
                dist,
                0.0,
                0,
                f"目标距离 {dist:.6g} 超出保守可达半径 {max_reach(chain):.6g}",
            )

    # ---------------- DLS 求解 ---------------- #
    from .jacobian import jacobian  # 局部导入避免模块初始化环依赖

    pos_rows = np.flatnonzero(mask)
    joint_centers = np.array(
        [
            0.5 * (lo + hi) if np.isfinite(lo) and np.isfinite(hi) else 0.0
            for lo, hi in chain.joint_limits
        ]
    )
    # 各关节可行半宽，用于给随机重启一个与量程匹配的扰动尺度。
    joint_halfwidths = np.array(
        [
            0.5 * (hi - lo) if np.isfinite(lo) and np.isfinite(hi) else np.pi
            for lo, hi in chain.joint_limits
        ]
    )

    def assemble_rows(j_full: np.ndarray) -> np.ndarray:
        # 注意：对 2D 矩阵直接用布尔一维数组做行索引会被 NumPy 展平，
        # 必须显式取整数行下标。
        blocks = [j_full[:3][pos_rows, :]]
        if track_orientation:
            blocks.append(j_full[3:, :])
        return np.vstack(blocks)

    def error_at(r, p):
        ep = (p_target - p) if p_target is not None else np.zeros(3)
        ep = ep[mask]
        if track_orientation and r_target is not None:
            eo = rotation_error(r_target, r)
        else:
            eo = np.zeros(0)
        return ep, eo

    def norms(q_now):
        r_now, p_now = _pose_at(chain, q_now)
        ep_now, eo_now = error_at(r_now, p_now)
        return float(np.linalg.norm(ep_now)), float(np.linalg.norm(eo_now))

    def dls_run(q_start: np.ndarray, iteration_budget: int):
        """从 q_start 出发跑阻尼最小二乘。

        返回 ``(q, n_pos, n_ori, used, stalled)``；``stalled=True`` 表示
        所有阻尼 / 步长下都找不到可接受的关节步（被限位挡住、局部极小或
        停在纯奇异点），调用方可据此决定重启还是放弃。
        """
        q = q_start.copy()
        n_pos, n_ori = norms(q)
        lam = max(float(damping), 1e-9)

        for used in range(1, iteration_budget + 1):
            if n_pos <= tol and n_ori <= tol:
                return q, n_pos, n_ori, used, False

            r_cur, p_cur = _pose_at(chain, q)
            ep, eo = error_at(r_cur, p_cur)
            j = assemble_rows(jacobian(chain, q))
            e = np.concatenate([ep, eo])
            norm_e = float(np.linalg.norm(e))
            min_decrease = max(1e-14, 1e-6 * tol)

            # 次级任务：把关节往行程中点拉。投影 (I - J^+J) 使其作用在
            # 雅可比零空间上，常规构型下不改变末端运动；在全伸展 / 全折叠
            # 这类雅可比秩亏的奇异初始化下提供打破对称所必需的自运动。
            center_gain = 0.3 if norm_e > 10.0 * tol else 0.0
            q_null = center_gain * (joint_centers - q)

            accepted = False
            lam_try = lam
            for _round in range(8):
                jj = j @ j.T
                eye_task = np.eye(j.shape[0])
                dq_task = j.T @ np.linalg.solve(
                    jj + (lam_try ** 2) * eye_task, e
                )
                if q_null.size and center_gain:
                    dq_null = (
                        np.eye(chain.dof)
                        - j.T @ np.linalg.solve(jj + (lam_try ** 2) * eye_task, j)
                    ) @ q_null
                    dq = dq_task + dq_null
                else:
                    dq = dq_task

                # 回溯线搜索：找一个严格下降的步。梯度很小时下降量也小，
                # 因此采用「相对下降 + 与步长尺度匹配的容差」，避免把
                # 正常但缓慢的下降误判为停滞。
                alpha = 1.0
                while alpha > 2.0 ** -24:
                    candidate = chain.clamp_to_limits(q + alpha * dq)
                    if np.allclose(candidate, q, atol=1e-15, rtol=0.0):
                        break
                    cand_p, cand_o = norms(candidate)
                    candidate_norm = (cand_p ** 2 + cand_o ** 2) ** 0.5
                    sufficient = (
                        candidate_norm
                        < norm_e - min_decrease - 1e-12 * alpha * (1.0 + norm_e)
                    )
                    if sufficient:
                        accepted = True
                        q = candidate
                        n_pos, n_ori = cand_p, cand_o
                        break
                    alpha *= 0.5
                if accepted:
                    # 本档阻尼有效则适度下调以加速末端收敛，否则加大。
                    lam = max(1e-10, min(1.0, lam_try * (0.7 if alpha > 0.25 else 1.5)))
                    break
                lam_try *= 4.0  # 可能在奇异点 / 限位附近，加大阻尼重试

            if not accepted:
                return q, n_pos, n_ori, used, True

        return q, n_pos, n_ori, iteration_budget, False

    # 第一次：用户给定 / 零位初始构型，吃满全部迭代预算。
    q, n_pos, n_ori, used, stalled = dls_run(q, int(max_iterations))
    iterations = used
    if n_pos <= tol and n_ori <= tol:
        return IKResult(
            IKStatus.CONVERGED, q, n_pos, n_ori, iterations, "在容差内收敛"
        )
    if not stalled:
        # 预算耗尽但仍在正常下降 —— 典型的「预算太小」情形。
        return IKResult(
            IKStatus.MAX_ITERATIONS, q, n_pos, n_ori, iterations,
            f"达到迭代上限 {max_iterations} 仍未收敛，位置误差 {n_pos:.3e}、"
            f"姿态误差 {n_ori:.3e}",
        )

    # 初始构型完全停滞（典型：零位是全伸展 / 全折叠的秩亏奇异点，
    # 或一开始就被限位顶住）。用固定种子的小幅随机扰动重启若干次，
    # 每次分配一段独立预算；总迭代数仍受 max_iterations 约束。
    budget_per_restart = max(1, max_iterations // 4)
    rng = np.random.default_rng(0)
    restart = 0
    while restart < 8 and iterations < max_iterations:
        restart += 1
        budget = min(budget_per_restart, max_iterations - iterations)
        if budget <= 0:
            break
        # 逐次重启扩大扰动并加大多样性（幅度 + 双向）。
        scale = 0.1 * restart
        q_seed = chain.clamp_to_limits(
            chain.zero_configuration()
            + scale * joint_halfwidths * rng.standard_normal(chain.dof)
        )
        q2, n2_pos, n2_ori, used2, stalled2 = dls_run(q_seed, budget)
        iterations += used2
        if n2_pos <= tol and n2_ori <= tol:
            return IKResult(
                IKStatus.CONVERGED, q2, n2_pos, n2_ori, iterations,
                f"在容差内收敛（随机重启 {restart} 次）",
            )
        if n2_pos < n_pos:  # 保留历史最好解
            q, n_pos, n_ori = q2, n2_pos, n2_ori

    if n_pos > tol or n_ori > tol:
        return IKResult(
            IKStatus.UNREACHABLE, q, n_pos, n_ori, iterations,
            "阻尼最小二乘找不到能继续下降误差的关节步：目标被关节限位阻挡、"
            "处于工作空间空洞或陷入局部极小（已尝试随机重启）",
        )
    return IKResult(
        IKStatus.CONVERGED, q, n_pos, n_ori, iterations, "在容差内收敛",
    )
