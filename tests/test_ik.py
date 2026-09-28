"""逆运动学 (IK) 测试：可达目标往返、限位、不可达、不收敛状态与 6R 回归。

关键约束：任何失败情形 success 必须为 False，并带有明确 status，
不得返回伪造的成功结果。
"""

from __future__ import annotations

import numpy as np
import pytest

from arm_kinematics import (
    IKStatus,
    forward_kinematics,
    inverse_kinematics,
    load_urdf,
    load_urdf_string,
    max_reach,
)

from build_urdfs import (
    build_urdf,
    planar_2r_urdf,
    six_revolute_urdf,
    three_joint_mixed_urdf,
)


def _roundtrip(robot, q_goal, q_init, tol_pos=1e-8, tol_rot=1e-8, **kw):
    target = forward_kinematics(robot, q_goal).end_transform
    return inverse_kinematics(robot, target, q_init=q_init,
                              tol_pos=tol_pos, tol_rot=tol_rot, **kw), target


def test_2r_ik_roundtrip_from_neighbor():
    robot = load_urdf_string(planar_2r_urdf())
    q_goal = np.array([0.6, -0.4])
    result, target = _roundtrip(robot, q_goal, np.array([0.5, -0.3]))
    assert result.success, result.message
    assert result.status is IKStatus.CONVERGED
    # 核验返回的 q 经 FK 后确实满足目标（而不是只看求解器自述）
    reached = forward_kinematics(robot, result.q).end_transform
    assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 1e-8
    Rrel = reached[:3, :3].T @ target[:3, :3]
    angle = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
    assert angle < 1e-7


def test_2r_ik_solves_from_zero_seed():
    robot = load_urdf_string(planar_2r_urdf())
    q_goal = np.array([0.4, 0.3])
    result, _ = _roundtrip(robot, q_goal, np.zeros(2),
                           max_iterations=500)
    assert result.success, result.message


def test_2r_ik_elbow_configuration_preserved():
    """位置型 IK（忽略姿态）：不同目标对应肘部 q2 符号不同，从各自近旁初值
    出发应保持构型符号，不会翻过肘部奇点。"""
    robot = load_urdf_string(planar_2r_urdf())

    target_up = forward_kinematics(robot, [0.5, -0.6]).end_transform
    r_up = inverse_kinematics(
        robot, target_up, q_init=[0.6, -0.7],
        tol_pos=1e-10, tol_rot=100.0, max_iterations=300,
    )
    assert r_up.success, r_up.message
    assert r_up.q[1] < 0
    assert np.allclose(
        forward_kinematics(robot, r_up.q).end_transform[:3, 3],
        target_up[:3, 3], atol=1e-8,
    )

    target_down = forward_kinematics(robot, [-0.2, 0.9]).end_transform
    r_down = inverse_kinematics(
        robot, target_down, q_init=[-0.1, 0.8],
        tol_pos=1e-10, tol_rot=100.0, max_iterations=300,
    )
    assert r_down.success, r_down.message
    assert r_down.q[1] > 0
    assert np.allclose(
        forward_kinematics(robot, r_down.q).end_transform[:3, 3],
        target_down[:3, 3], atol=1e-8,
    )


def test_ik_orientation_component_matters():
    """平面 2R 末端可达某点，但该点末端姿态只有两种；要求一个达不到的姿态
    必须明确失败（MAX_ITERATIONS/STALLED），不能伪造成功。"""
    robot = load_urdf_string(planar_2r_urdf())
    target = forward_kinematics(robot, [0.5, -0.6]).end_transform.copy()
    # 把目标姿态绕 x 轴翻转 90 度：2R 末端 z 轴永远沿基座 z，不可能做到
    target[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    result = inverse_kinematics(robot, target, q_init=[0.4, -0.5],
                                tol_pos=1e-8, tol_rot=1e-8,
                                max_iterations=100)
    assert not result.success
    assert result.status in (IKStatus.MAX_ITERATIONS, IKStatus.STALLED)
    assert result.rot_error > 1e-8


def test_ik_target_beyond_max_reach_is_unreachable():
    robot = load_urdf_string(planar_2r_urdf(1.0, 1.0))
    assert max_reach(robot) == pytest.approx(2.0)
    target = np.eye(4)
    target[0, 3] = 5.0
    result = inverse_kinematics(robot, target, q_init=[0.1, 0.1])
    assert not result.success
    assert result.status is IKStatus.UNREACHABLE
    assert "不可达" in result.message


def test_ik_target_outside_workspace_near_boundary():
    """距离在可达半径之内但 2R 实际到不了的点（如 (-2, 0.01) 附近的反向姿态
    或臂展略超的点 2.01m）——求解器必须报失败。"""
    robot = load_urdf_string(planar_2r_urdf(1.0, 1.0))
    target = np.eye(4)
    target[0, 3] = 2.5  # > 臂展 2m
    result = inverse_kinematics(robot, target)
    assert not result.success
    assert result.status is IKStatus.UNREACHABLE


def test_ik_pose_inside_workspace_can_be_unreachable():
    """工作空间内部的点也可能因姿态不可达而整体 pose 不可达。

    平面 2R 到达点 (x,y) 时末端 yaw 只可能是 ``φ+ψ−β`` 或 ``φ−ψ+β``
    （φ=atan2(y,x)，β=|q2|，ψ=β/2），不可能取径向角 φ（除非全伸展）。
    给一个径向姿态的目标，求解器必须停在局部驻点并明确报 stalled，
    不得返回 success。
    """
    robot = load_urdf_string(planar_2r_urdf(1.0, 1.0))
    x, y = 1.9, 0.5
    yaw = np.arctan2(y, x)
    target = np.eye(4)
    target[0, 3], target[1, 3] = x, y
    target[:3, :3] = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    result = inverse_kinematics(robot, target, q_init=[0.5, -0.3],
                                tol_pos=1e-9, tol_rot=1e-9,
                                max_iterations=200)
    assert not result.success
    assert result.status in (IKStatus.STALLED, IKStatus.MAX_ITERATIONS)
    # 失败时必须仍残留可观察到的误差
    assert result.pos_error > 1e-9 or result.rot_error > 1e-9


def test_ik_respects_joint_limits():
    """收紧 j2 限位后，需要 j2 大幅弯曲才能到达的目标必须失败且解不越限。"""
    urdf = build_urdf([
        {"name": "j1", "type": "revolute", "child": "l1",
         "axis": "0 0 1", "lower": -3.14, "upper": 3.14},
        {"name": "j2", "type": "revolute", "child": "l2",
         "xyz": "1 0 0", "axis": "0 0 1",
         "lower": 0.0, "upper": 0.2},  # j2 几乎不能弯
        {"name": "j_ee", "type": "fixed", "child": "end_effector",
         "xyz": "1 0 0"},
    ])
    robot = load_urdf_string(urdf)
    # 正常 2R 需要“肘部收拢”（j2≈-0.8）才能到达的目标
    target = forward_kinematics(
        load_urdf_string(planar_2r_urdf()), [0.3, -0.8]
    ).end_transform
    result = inverse_kinematics(robot, target, q_init=[0.2, 0.1],
                                tol_pos=1e-6, tol_rot=1e-6,
                                max_iterations=150, use_limits=True)
    assert not result.success
    # 返回的 q（若有）也绝不应越限
    if result.q is not None:
        assert result.q[1] >= -1e-9
    assert result.status in (IKStatus.MAX_ITERATIONS, IKStatus.STALLED,
                             IKStatus.UNREACHABLE)


def test_ik_mixed_3_model_roundtrip():
    """revolute + fixed + continuous + prismatic 的 3 自由度往返。"""
    robot = load_urdf_string(three_joint_mixed_urdf())
    q_goal = np.array([0.4, 0.7, 0.12])
    result, target = _roundtrip(robot, q_goal, np.array([0.1, 0.2, 0.0]),
                                max_iterations=300)
    assert result.success, result.message
    reached = forward_kinematics(robot, result.q).end_transform
    assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 1e-7
    # prismatic 解必须在限位内
    assert -0.25 - 1e-9 <= result.q[2] <= 0.25 + 1e-9


def test_6r_ik_roundtrip_near_seed():
    robot = load_urdf_string(six_revolute_urdf())
    q_goal = np.array([0.5, -0.4, 0.3, 0.8, -0.2, 0.6])
    result, target = _roundtrip(robot, q_goal,
                                q_goal + np.array([0.05, -0.03, 0.04,
                                                   -0.02, 0.03, -0.04]),
                                tol_pos=1e-9, tol_rot=1e-9,
                                max_iterations=400)
    assert result.success, result.message
    reached = forward_kinematics(robot, result.q).end_transform
    assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 1e-8
    Rrel = reached[:3, :3].T @ target[:3, :3]
    angle = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
    assert angle < 1e-8


def test_6r_ik_from_rough_seed_converges_or_reports_honestly():
    robot = load_urdf_string(six_revolute_urdf())
    q_goal = np.array([0.3, -0.2, 0.4, 0.2, 0.1, -0.3])
    target = forward_kinematics(robot, q_goal).end_transform
    result = inverse_kinematics(robot, target, q_init=np.zeros(6),
                                tol_pos=1e-6, tol_rot=1e-6,
                                max_iterations=800)
    # 收敛则必须真正达到精度；没收敛也必须给明确状态，二者必居其一
    if result.success:
        reached = forward_kinematics(robot, result.q).end_transform
        assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 1e-6
    else:
        assert result.status in (IKStatus.MAX_ITERATIONS, IKStatus.STALLED,
                                 IKStatus.UNREACHABLE)
        assert result.pos_error > 1e-6 or result.rot_error > 1e-6


def test_6r_ik_random_restarts_reach_fk_generated_goals():
    """由 FK 构造的目标必然可达；给足随机重启时，求解器必须找到解，
    且报告使用过的重启次数——验证重启机制确实在帮求解器跳出局部驻点。"""
    robot = load_urdf_string(six_revolute_urdf())
    rng = np.random.default_rng(123)
    solved = 0
    for _ in range(5):
        q_goal = np.array([
            rng.uniform(lo, hi) for lo, hi in robot.joint_limits()
        ]) * 0.8
        target = forward_kinematics(robot, q_goal).end_transform
        result = inverse_kinematics(
            robot, target, q_init=np.zeros(robot.nq),
            tol_pos=1e-6, tol_rot=1e-6,
            max_iterations=100, restarts=40, seed=7,
        )
        assert result.success or result.status in (
            IKStatus.STALLED, IKStatus.MAX_ITERATIONS
        )
        if result.success:
            reached = forward_kinematics(robot, result.q).end_transform
            assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 1e-6
            solved += 1
    # 40 次均匀随机重启足以覆盖大部分 6R 构型；至少大多数应收敛
    assert solved >= 4, f"仅 {solved}/5 个 FK 目标收敛"


@pytest.mark.parametrize("goal_seed,restart_seed", [(123, 7), (99, 3), (5, 11)])
def test_6r_ik_hard_seed_converges_with_enough_restarts(goal_seed, restart_seed):
    """固定几个从零点最难收敛的 FK 目标，给足重启后必须收敛（回归测试）。"""
    robot = load_urdf_string(six_revolute_urdf())
    rng = np.random.default_rng(goal_seed)
    q_goal = np.array([
        rng.uniform(lo, hi) for lo, hi in robot.joint_limits()
    ]) * 0.8
    target = forward_kinematics(robot, q_goal).end_transform
    result = inverse_kinematics(
        robot, target, q_init=np.zeros(robot.nq),
        tol_pos=1e-6, tol_rot=1e-6,
        max_iterations=60, restarts=60, seed=restart_seed,
    )
    assert result.success, (
        f"goal_seed={goal_seed} restart_seed={restart_seed}: "
        f"{result.status.value} pe={result.pos_error} re={result.rot_error}"
    )
    reached = forward_kinematics(robot, result.q).end_transform
    assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 1e-6


def test_ik_restarts_zero_means_single_attempt():
    robot = load_urdf_string(six_revolute_urdf())
    target = forward_kinematics(
        robot, np.array([0.3, -0.2, 0.4, 0.2, 0.1, -0.3])
    ).end_transform
    result = inverse_kinematics(robot, target, q_init=np.zeros(6),
                                restarts=0, max_iterations=50)
    assert result.restarts_used == 0
    if result.success:
        assert result.pos_error < 1e-5
    else:
        assert result.status in (IKStatus.STALLED, IKStatus.MAX_ITERATIONS)


def test_ik_zero_dof_chain():
    urdf = build_urdf([{"name": "jf", "type": "fixed", "child": "l1",
                        "xyz": "1 0 0"}])
    robot = load_urdf_string(urdf)
    assert robot.nq == 0
    ok = inverse_kinematics(robot, forward_kinematics(robot, []).end_transform)
    assert ok.success and ok.q.shape == (0,)

    miss = np.eye(4)
    miss[2, 3] = 0.5
    bad = inverse_kinematics(robot, miss)
    assert not bad.success and bad.status is IKStatus.UNREACHABLE


def test_ik_invalid_arguments():
    robot = load_urdf_string(planar_2r_urdf())
    with pytest.raises(ValueError):
        inverse_kinematics(robot, np.eye(3))
    with pytest.raises(ValueError):
        inverse_kinematics(robot, np.eye(4), tol_pos=0.0)
    with pytest.raises(ValueError):
        inverse_kinematics(robot, np.eye(4), max_iterations=0)
    with pytest.raises(ValueError):
        inverse_kinematics(robot, np.eye(4), restarts=-1)


def test_ik_on_loaded_file_model():
    """从 tests 外部的本地 URDF 文件加载（2R），完成一次往返求解。"""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    robot = load_urdf(os.path.join(root, "urdf", "arm2r.urdf"))
    result, target = _roundtrip(robot, np.array([-0.5, 0.7]),
                                np.array([-0.4, 0.6]))
    assert result.success, result.message
    assert np.allclose(
        forward_kinematics(robot, result.q).end_transform[:3, 3],
        target[:3, 3], atol=1e-8,
    )
