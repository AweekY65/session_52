"""逆运动学测试：收敛、容差、限位、不可达目标、零自由度、位置掩码。"""

import numpy as np
import pytest

from robot_kinematics import (
    IKStatus,
    KinematicChain,
    forward_kinematics,
    inverse_kinematics,
    max_reach,
    parse_urdf,
    within_reach,
)

from .urdfs import (
    arm_6r_urdf,
    fixed_only_urdf,
    limited_2r_urdf,
    planar_arm_urdf,
    prismatic_urdf,
)


def make_chain(urdf_text: str) -> KinematicChain:
    return KinematicChain(parse_urdf(urdf_text))


def analytic_planar_tip(q, lengths):
    q = np.asarray(q, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    angle = np.cumsum(q)
    return np.array(
        [
            float(np.sum(lengths * np.cos(angle))),
            float(np.sum(lengths * np.sin(angle))),
            0.0,
        ]
    )


# ---------------------------------------------------------------------- #
# 可达目标的收敛性
# ---------------------------------------------------------------------- #
def test_planar_2r_reaches_analytic_points():
    lengths = np.array([1.0, 0.7])
    chain = make_chain(planar_arm_urdf(2, lengths))
    rng = np.random.default_rng(7)
    for _ in range(6):
        q_star = rng.uniform(-2.0, 2.0, 2)
        target = analytic_planar_tip(q_star, lengths)
        result = inverse_kinematics(chain, target, q0=[0.3, 0.3], tol=1e-8)
        assert result.status is IKStatus.CONVERGED, result.message
        assert result.position_error <= 1e-8
        # 解必须真的到达目标（不伪造成功）
        tip = forward_kinematics(chain, result.q)
        np.testing.assert_allclose(tip[:3, 3], target, atol=1e-6)


def test_fk_ik_roundtrip_planar3():
    lengths = np.array([1.0, 0.8, 0.6])
    chain = make_chain(planar_arm_urdf(3, lengths))
    rng = np.random.default_rng(11)
    for _ in range(5):
        q_star = rng.uniform(-1.5, 1.5, 3)
        target = forward_kinematics(chain, q_star)
        result = inverse_kinematics(chain, target, q0=[0.2, 0.2, 0.2], tol=1e-8)
        assert result.status is IKStatus.CONVERGED, result.message
        tip = forward_kinematics(chain, result.q)
        np.testing.assert_allclose(tip[:3, 3], target[:3, 3], atol=1e-6)
        np.testing.assert_allclose(tip[:3, :3], target[:3, :3], atol=1e-6)


def test_fk_ik_roundtrip_6dof_full_pose():
    chain = make_chain(arm_6r_urdf())
    rng = np.random.default_rng(3)
    for _ in range(3):
        q_star = rng.uniform(-1.2, 1.2, 6)
        target = forward_kinematics(chain, q_star)
        result = inverse_kinematics(
            chain, target, q0=np.zeros(6), tol=1e-7, max_iterations=300
        )
        assert result.status is IKStatus.CONVERGED, result.message
        assert result.position_error <= 1e-7
        assert result.orientation_error <= 1e-7
        tip = forward_kinematics(chain, result.q)
        np.testing.assert_allclose(tip[:3, 3], target[:3, 3], atol=1e-5)
        np.testing.assert_allclose(tip[:3, :3], target[:3, :3], atol=1e-5)


def test_solution_respects_tight_tolerance():
    chain = make_chain(planar_arm_urdf(2, [1.0, 0.7]))
    target = analytic_planar_tip([0.6, -0.8], [1.0, 0.7])
    result = inverse_kinematics(chain, target, q0=[0.1, 0.1], tol=1e-10)
    assert result.status is IKStatus.CONVERGED
    assert result.position_error <= 1e-10
    tip = forward_kinematics(chain, result.q)
    assert np.linalg.norm(tip[:3, 3] - target) <= 1e-9


def test_prismatic_ik():
    chain = make_chain(prismatic_urdf())
    result = inverse_kinematics(chain, np.array([0.6, 0.0, 0.0]))
    assert result.status is IKStatus.CONVERGED
    assert result.q[0] == pytest.approx(0.6, abs=1e-6)


def test_prismatic_ik_limit_boundary():
    chain = make_chain(prismatic_urdf(lower=-0.5, upper=1.0))
    # 恰好在上限：可达
    at_limit = inverse_kinematics(chain, np.array([1.0, 0.0, 0.0]))
    assert at_limit.status is IKStatus.CONVERGED
    assert at_limit.q[0] == pytest.approx(1.0, abs=1e-6)
    # 略超上限：不可达，且解必须停在限位内而不是越界
    beyond = inverse_kinematics(chain, np.array([1.2, 0.0, 0.0]))
    assert beyond.status is IKStatus.UNREACHABLE
    assert beyond.q[0] <= 1.0 + 1e-9


def test_mask_position_only():
    # 只约束 x：prismatic 沿 x，目标 y 分量离谱也应收敛
    chain = make_chain(prismatic_urdf())
    result = inverse_kinematics(
        chain,
        np.array([0.3, 5.0, -2.0]),
        position_mask=(True, False, False),
    )
    assert result.status is IKStatus.CONVERGED
    assert result.q[0] == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------- #
# 不可达 / 失败状态
# ---------------------------------------------------------------------- #
def test_unreachable_far_target_returns_unreachable():
    chain = make_chain(planar_arm_urdf(2, [1.0, 0.7]))
    result = inverse_kinematics(chain, np.array([10.0, 0.0, 0.0]))
    assert result.status is IKStatus.UNREACHABLE
    assert not result.success
    # 明确状态 + 非零误差，绝不伪造成功
    assert result.position_error > 1.0


def test_unreachable_outside_workspace_3r():
    # 长度 [2.0, 0.5, 0.5]：内半径 1.0、外半径 3.0 的环形工作空间
    chain = make_chain(planar_arm_urdf(3, [2.0, 0.5, 0.5]))
    inner = inverse_kinematics(chain, np.array([0.3, 0.0, 0.0]), max_iterations=300)
    assert inner.status is IKStatus.UNREACHABLE
    outer = inverse_kinematics(chain, np.array([3.5, 0.0, 0.0]))
    assert outer.status is IKStatus.UNREACHABLE


def test_target_blocked_by_joint_limits():
    # j2 限位 ±0.1 rad：两连杆几乎只能保持共线，末端到 base 的距离被
    # 限制在约 [1.9975, 2.0]。目标 (1, 0) 距离 1.0，几何上需要 j2≈pi
    # 的折叠构型，被限位挡住。
    chain = make_chain(limited_2r_urdf(j2_limit=0.1))
    result = inverse_kinematics(chain, np.array([1.0, 0.0, 0.0]), max_iterations=300)
    assert result.status is IKStatus.UNREACHABLE
    assert result.position_error > 0.5
    # 返回的关节值必须仍在限位内
    lo, hi = chain.joint_limits[1]
    assert lo - 1e-9 <= result.q[1] <= hi + 1e-9


def test_max_iterations_status_when_budget_too_small():
    chain = make_chain(planar_arm_urdf(3, [1.0, 0.8, 0.6]))
    target = analytic_planar_tip([1.0, -0.5, 0.8], [1.0, 0.8, 0.6])
    # 非奇异初始构型 + 1 次迭代预算：来不及收敛
    result = inverse_kinematics(
        chain, target, q0=[0.4, 0.4, 0.4], tol=1e-12, max_iterations=1
    )
    assert result.status is IKStatus.MAX_ITERATIONS
    assert result.iterations <= 1
    # 预算为 0 同样返回 MAX_ITERATIONS
    zero_budget = inverse_kinematics(chain, target, q0=[0.4, 0.4, 0.4], max_iterations=0)
    assert zero_budget.status is IKStatus.MAX_ITERATIONS


def test_bad_target_shape_returns_bad_target():
    chain = make_chain(planar_arm_urdf(2, [1.0, 0.7]))
    for bad in (np.zeros(5), np.zeros((2, 2)), np.zeros((4, 4, 4)), None,
                np.array([np.nan, 0.0, 0.0]), "not a pose"):
        result = inverse_kinematics(chain, bad)
        assert result.status is IKStatus.BAD_TARGET
        assert result.q is None


def test_zero_dof_chain_ik():
    chain = make_chain(fixed_only_urdf())
    assert chain.dof == 0
    # 目标就是当前位姿：收敛
    tip = forward_kinematics(chain, [])
    ok = inverse_kinematics(chain, tip)
    assert ok.status is IKStatus.CONVERGED
    # 目标不同：不可达（零自由度无法改变位姿）
    miss = inverse_kinematics(chain, np.array([1.0, 1.0, 0.0]))
    assert miss.status is IKStatus.UNREACHABLE


def test_reachability_checker_conservative():
    lengths = np.array([1.0, 0.8, 0.6])
    chain = make_chain(planar_arm_urdf(3, lengths))
    # 保守性：所有真实可达的末端位置都必须落在估计半径内（不许误杀）
    rng = np.random.default_rng(5)
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, 3)
        tip = forward_kinematics(chain, q)
        assert within_reach(chain, tip[:3, 3])
    # 明显超出最大伸展的目标必须被识别为不可达
    assert not within_reach(chain, [lengths.sum() + 0.5, 0.0, 0.0])
    assert max_reach(chain) == pytest.approx(lengths.sum())
    # IK 对超远目标快速返回 UNREACHABLE（迭代数为 0，预检拦截）
    far = inverse_kinematics(chain, np.array([10.0, 0.0, 0.0]))
    assert far.status is IKStatus.UNREACHABLE
    assert far.iterations == 0


def test_result_reports_errors_and_iterations():
    chain = make_chain(planar_arm_urdf(2, [1.0, 0.7]))
    target = analytic_planar_tip([0.5, 0.5], [1.0, 0.7])
    result = inverse_kinematics(chain, target, q0=[0.0, 0.0])
    assert result.status is IKStatus.CONVERGED
    assert result.iterations >= 1
    assert result.position_error <= 1e-6
    assert result.orientation_error <= 1e-6
    assert result.message
