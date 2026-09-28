"""数值逆运动学（自适应阻尼最小二乘 / Damped Least Squares + 随机重启）。

目标：找到关节角 ``q`` 使末端位姿 ``FK(q)`` 接近目标 4x4 齐次变换。

每轮迭代::

    e = pose_error(FK(q), T_target)              # 6 维，基座系
    J = geometric_jacobian(robot, q)             # 6×nq
    Δq = J^T (J J^T + λ²I)^-1 e                  # 阻尼最小二乘步长

阻尼系数是**自适应**的：以可操作度指标 ``sigma_min(J)``（最小奇异值）衡量与
奇异构型的距离，越接近奇异 λ 越大，避免步长在病态方向发散：
``λ² = λ0² + max(0, 1 − (σ_min/σ0)²)·λ_max²``（仅当 σ_min<σ0 时触发）。
随后用回退（backtracking）线搜索保证收敛尺度化后的误差
（``‖e_pos‖/tol_pos + ‖e_rot‖/tol_rot``）单调下降——直接把米与弧度相加会因
量纲尺度不同而错误拒绝「以极小姿态代价换取大幅位置下降」的步长——并把结果
钳制到关节限位内。

由于 IK 是非凸问题，欠驱动臂（nq<6）在工作空间内部就可能存在完整位姿不可达
的非零驻点，满秩臂从远处初值也可能落入局部极小。线搜索彻底无法下降时，
求解器会在关节限位内进行至多 ``restarts`` 次均匀随机重启；全部尝试失败后返回
明确的失败状态，绝不伪造成功。

满足 ``‖e_pos‖ <= tol_pos`` 且 ``‖e_rot‖ <= tol_rot`` 即返回 ``CONVERGED``；
迭代上限用尽返回 ``MAX_ITERATIONS``；线搜索无法继续下降且重启次数耗尽返回
``STALLED``；目标在保守可达半径之外返回 ``UNREACHABLE``（见 :func:`max_reach`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .kinematics import coerce_q, forward_kinematics, jacobian
from .transforms import pose_error
from .urdf_loader import JointType, RobotModel


class IKStatus(str, Enum):
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    STALLED = "stalled"
    UNREACHABLE = "unreachable"


@dataclass
class IKResult:
    """IK 求解结果。

    status 不为 CONVERGED 时，success 恒为 False，调用方不得把 q 当作可行解使用。
    """

    success: bool
    status: IKStatus
    q: np.ndarray | None
    iterations: int
    pos_error: float
    rot_error: float
    message: str
    history: list[float] = field(default_factory=list)
    restarts_used: int = 0


def max_reach(robot: RobotModel) -> float:
    """保守的可达半径上界（单位米）。

    对链上每个活动/固定关节的 <origin xyz> 平移长度以及 prismatic 关节的
    行程长度求和：沿链的折线总长是末端可达距离的上界（三角形不等式），
    因此目标距离超过该值时必定不可达；不超过也不代表一定可达。
    """
    reach = 0.0
    for joint in robot.chain:
        reach += float(np.linalg.norm(joint.origin.xyz))
        if joint.joint_type is JointType.PRISMATIC:
            reach += max(abs(joint.upper), abs(joint.lower))
    return reach


def _clamp_to_limits(robot: RobotModel, q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for i, joint in enumerate(robot.movable_joints):
        if joint.joint_type is JointType.CONTINUOUS:
            continue
        out[i] = np.clip(out[i], joint.lower, joint.upper)
    return out


def _errors(t: np.ndarray, target: np.ndarray) -> tuple[float, float, np.ndarray]:
    e = pose_error(t, target)
    return float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])), e


def _random_q(robot: RobotModel, rng: np.random.Generator) -> np.ndarray:
    q = np.zeros(robot.nq)
    for i, joint in enumerate(robot.movable_joints):
        if joint.joint_type is JointType.CONTINUOUS:
            q[i] = rng.uniform(-np.pi, np.pi)
        else:
            q[i] = rng.uniform(joint.lower, joint.upper)
    return q


def inverse_kinematics(
    robot: RobotModel,
    target_pose: np.ndarray,
    q_init=None,
    *,
    tol_pos: float = 1e-5,
    tol_rot: float = 1e-5,
    max_iterations: int = 200,
    damping: float = 1e-4,
    singularity_threshold: float = 0.005,
    max_damping: float = 1e-3,
    max_step: float = 0.5,
    use_limits: bool = True,
    restarts: int = 5,
    seed: int | None = None,
) -> IKResult:
    """带随机重启的自适应阻尼最小二乘逆运动学。

    Args:
        target_pose: 目标 4x4 齐次变换（末端相对基座）
        q_init: 初始关节角；缺省为各关节限位中点（continuous 为 0）。
            线搜索停滞时忽略初值，在限位内均匀随机重启。
        tol_pos / tol_rot: 位置（米）与姿态（弧度）收敛阈值
        max_iterations: 单次尝试的迭代上限
        damping: 基础阻尼 λ0；远离奇异时使用，保证极小
        singularity_threshold: 最小奇异值 σ0：σ_min < σ0 视为接近奇异，
            阻尼按 ``λ0² + (1 − (σ_min/σ0)²)·max_damping²`` 增大
        max_damping: 奇异点处阻尼平方项的上界
        max_step: 单关节单步变化量上限（弧度/米）；过大的 Gauss-Newton
            步会越过最优点，配合线搜索使用更稳
        use_limits: 为 True 时每步把解钳制在 revolute/prismatic 限位内
        restarts: 线搜索停滞后的随机重启次数（0 表示只用给定初值尝试一次）
        seed: 随机重启使用的随机种子，None 表示非确定性

    Returns:
        IKResult。收敛失败时 success=False 且给出明确 status，绝不会返回伪造成功。
    """
    target_pose = np.asarray(target_pose, dtype=float)
    if target_pose.shape != (4, 4):
        raise ValueError("target_pose 必须是 4x4 齐次变换矩阵")
    if not np.all(np.isfinite(target_pose)):
        raise ValueError("target_pose 包含非有限值")
    if tol_pos <= 0 or tol_rot <= 0:
        raise ValueError("tol_pos 与 tol_rot 必须为正数")
    if max_iterations < 1:
        raise ValueError("max_iterations 必须 >= 1")
    if restarts < 0:
        raise ValueError("restarts 必须 >= 0")

    movable = robot.movable_joints

    # 零自由度：只能检查零位正运动学是否已等于目标
    if not movable:
        t0 = forward_kinematics(robot, np.zeros(0)).end_transform
        pe, re, _ = _errors(t0, target_pose)
        if pe <= tol_pos and re <= tol_rot:
            return IKResult(True, IKStatus.CONVERGED, np.zeros(0), 0, pe, re,
                            "固定关节链，零位即满足目标")
        return IKResult(False, IKStatus.UNREACHABLE, None, 0, pe, re,
                        "机器人没有活动关节，无法运动到目标位姿")

    # 初始关节角
    if q_init is None:
        q0 = []
        for j in movable:
            if j.joint_type is JointType.CONTINUOUS:
                q0.append(0.0)
            else:
                q0.append(0.5 * (j.lower + j.upper))
        initial_q = np.array(q0, dtype=float)
    else:
        initial_q = coerce_q(robot, q_init)
    if use_limits:
        initial_q = _clamp_to_limits(robot, initial_q)

    t = forward_kinematics(robot, initial_q).end_transform
    pe0, re0, _ = _errors(t, target_pose)

    # 保守可达性预检（只按基座原点到目标的直线距离判断，不产生误杀）
    target_dist = float(np.linalg.norm(target_pose[:3, 3]))
    if target_dist > max_reach(robot) + tol_pos:
        return IKResult(
            False, IKStatus.UNREACHABLE, initial_q, 0, pe0, re0,
            f"目标距离基座 {target_dist:.4f} m，超过保守可达半径 "
            f"{max_reach(robot):.4f} m，目标不可达",
        )

    rng = np.random.default_rng(seed)
    lam0 = damping
    identity6 = np.eye(6)

    def merit(pos_err: float, rot_err: float) -> float:
        """收敛尺度化的误差标量：位置与姿态分别按各自容差归一化后相加。"""
        return pos_err / tol_pos + rot_err / tol_rot

    def attempt(q_start: np.ndarray):
        """从给定初值做一次完整 DLS 求解，返回 IKResult（不含 unreachable）。"""
        q = q_start.copy()
        t_cur = forward_kinematics(robot, q).end_transform
        pe, re, e = _errors(t_cur, target_pose)
        history = [pe]

        for iteration in range(1, max_iterations + 1):
            if pe <= tol_pos and re <= tol_rot:
                return IKResult(
                    True, IKStatus.CONVERGED, q, iteration - 1, pe, re,
                    f"第 {iteration - 1} 次迭代收敛", history=history,
                )

            j = jacobian(robot, q)
            # 自适应阻尼：用 J J^T 的最小特征值（等价于 σ_min²）衡量可操作度，
            # 比 6×nq 雅可比的完整 SVD 便宜；接近奇异构型时增大阻尼。
            jj = j @ j.T
            try:
                eigvals = np.linalg.eigvalsh(jj)
            except np.linalg.LinAlgError:
                break
            eig_floor = max(float(eigvals[0]), 0.0)
            near_singular = eig_floor < singularity_threshold ** 2
            if near_singular:
                # σ_min = sqrt(eig_min)，按同一二次型比例计算阻尼增量
                ratio = np.sqrt(eig_floor) / singularity_threshold
                lam2 = (
                    lam0 * lam0
                    + (1.0 - ratio * ratio) * max_damping * max_damping
                )
            else:
                lam2 = lam0 * lam0

            # DLS: J^T (J J^T + λ²I)^-1 e  —— 非方雅可比也适用。
            try:
                delta = j.T @ np.linalg.solve(j @ j.T + lam2 * identity6, e)
            except np.linalg.LinAlgError:
                break

            if max_step is not None:
                worst = float(np.max(np.abs(delta))) if delta.size else 0.0
                if worst > max_step:
                    delta *= max_step / worst

            # 回退线搜索：只接受使尺度化综合误差相对下降的步长；限位内求解。
            # 改善阈值取绝对值（容差很紧时仍可继续向零推进）与相对值
            # （远离收敛时避免浮点噪声假停滞）的较大者。
            merit_old = merit(pe, re)
            min_improvement = max(1e-10 * merit_old, 1e-11)
            scale = 1.0
            accepted = False
            q_new = q
            pe_new, re_new = pe, re
            delta_damped = True
            for _ in range(50):
                candidate = q + scale * delta
                if use_limits:
                    candidate = _clamp_to_limits(robot, candidate)
                t_new = forward_kinematics(robot, candidate).end_transform
                pe_c, re_c, _ = _errors(t_new, target_pose)
                if merit_old - merit(pe_c, re_c) > min_improvement:
                    accepted = True
                    q_new, pe_new, re_new = candidate, pe_c, re_c
                    break
                scale *= 0.5
                # 阻尼步在某些近奇异构型下会偏向错误的零空间方向；
                # 退到基础阻尼重新给一次完整步长的机会。
                if near_singular and delta_damped and scale < 1e-3:
                    try:
                        delta = j.T @ np.linalg.solve(
                            j @ j.T + lam0 * lam0 * identity6, e
                        )
                    except np.linalg.LinAlgError:
                        break
                    if max_step is not None:
                        worst = float(np.max(np.abs(delta)))
                        if worst > max_step:
                            delta *= max_step / worst
                    delta_damped = False
                    scale = 1.0
            if not accepted:
                return IKResult(
                    False, IKStatus.STALLED, q, iteration - 1, pe, re,
                    "线搜索无法继续下降，求解停滞"
                    f"（剩余位置误差 {pe:.3e} m，姿态误差 {re:.3e} rad）",
                    history=history,
                )

            q = q_new
            pe, re = pe_new, re_new
            t_cur = forward_kinematics(robot, q).end_transform
            _, _, e = _errors(t_cur, target_pose)
            history.append(pe)

        if pe <= tol_pos and re <= tol_rot:
            return IKResult(True, IKStatus.CONVERGED, q, max_iterations, pe, re,
                            "达到迭代上限时恰好收敛", history=history)
        return IKResult(
            False, IKStatus.MAX_ITERATIONS, q, max_iterations, pe, re,
            f"达到迭代上限 {max_iterations} 仍未收敛"
            f"（位置误差 {pe:.3e} m，姿态误差 {re:.3e} rad）",
            history=history,
        )

    # ---- 初值尝试 + 随机重启 ------------------------------------------------
    best = attempt(initial_q)
    if best.success:
        return best

    total_iterations = best.iterations
    for restart in range(restarts):
        q_rand = _random_q(robot, rng)
        candidate = attempt(q_rand)
        total_iterations += candidate.iterations
        if candidate.success:
            candidate.iterations = total_iterations
            candidate.restarts_used = restart + 1
            candidate.message = (
                f"初值未收敛，第 {restart + 1} 次随机重启后"
                f"{candidate.message}"
            )
            return candidate
        # 保留误差最小的失败结果作为最终报告
        if (candidate.pos_error / tol_pos + candidate.rot_error / tol_rot
                < best.pos_error / tol_pos + best.rot_error / tol_rot):
            best = candidate

    if best.status is IKStatus.STALLED and restarts > 0:
        best.message = (
            f"初值与 {restarts} 次随机重启均停滞"
            f"（最佳剩余位置误差 {best.pos_error:.3e} m，"
            f"姿态误差 {best.rot_error:.3e} rad）；目标可能不可达"
        )
    best.iterations = total_iterations
    best.restarts_used = restarts
    return best
