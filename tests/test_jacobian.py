"""Jacobian 测试：解析雅可比与中心有限差分在 2/3/6 关节模型上逐项对比。

位置部分与旋转部分分开断言，分别满足误差阈值。
"""

from __future__ import annotations

import numpy as np
import pytest

from arm_kinematics import (
    check_jacobian,
    finite_difference_jacobian,
    forward_kinematics,
    jacobian,
    load_urdf_string,
)

from build_urdfs import (
    build_urdf,
    planar_2r_urdf,
    six_revolute_urdf,
    three_joint_mixed_urdf,
)


def test_planar_2r_jacobian_closed_form():
    """平面 2R 雅可比有解析公式，直接核对。"""
    robot = load_urdf_string(planar_2r_urdf(1.0, 1.0))
    q = np.array([0.4, -0.9])
    j = jacobian(robot, q)

    assert j.shape == (6, 2)
    q12 = q.sum()
    # Jv = [-s1 - s12, -s12; c1 + c12, c12; 0, 0]
    assert np.allclose(j[0, 0], -np.sin(q[0]) - np.sin(q12))
    assert np.allclose(j[0, 1], -np.sin(q12))
    assert np.allclose(j[1, 0], np.cos(q[0]) + np.cos(q12))
    assert np.allclose(j[1, 1], np.cos(q12))
    assert np.allclose(j[2, :], 0.0)
    # 两关节都绕基座 z 轴
    assert np.allclose(j[3:5, :], 0.0)
    assert np.allclose(j[5, :], 1.0)


@pytest.mark.parametrize(
    "q",
    [
        np.zeros(2),
        np.array([0.3, -0.7]),
        np.array([-1.5, 2.0]),
    ],
)
def test_2r_jacobian_finite_difference(q):
    robot = load_urdf_string(planar_2r_urdf())
    result = check_jacobian(robot, q, eps=1e-6, tol=1e-5)
    assert result.passed, (
        f"位置最大误差 {result.pos_max_err}, 旋转最大误差 {result.rot_max_err}"
    )
    assert result.pos_max_err < 5e-6
    assert result.rot_max_err < 5e-6


def test_mixed_3_joint_jacobian_shapes_and_fd():
    """revolute + fixed + continuous + prismatic：fixed 不出现在列中。"""
    robot = load_urdf_string(three_joint_mixed_urdf())
    q = np.array([0.3, -0.5, 0.1])
    j = jacobian(robot, q)
    assert j.shape == (6, 3)

    result = check_jacobian(robot, q, eps=1e-6, tol=1e-5)
    assert result.pos_max_err < 5e-6, result.abs_err
    assert result.rot_max_err < 5e-6, result.abs_err

    # 第三列是 prismatic：角速度列为 0，线速度列是单位轴
    assert np.allclose(j[3:, 2], 0.0)
    assert np.isclose(np.linalg.norm(j[:3, 2]), 1.0, atol=1e-12)

    # 前两列是转动关节：角速度列是单位向量
    assert np.isclose(np.linalg.norm(j[3:, 0]), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(j[3:, 1]), 1.0, atol=1e-12)


def test_6r_jacobian_finite_difference_multiple_poses():
    robot = load_urdf_string(six_revolute_urdf())
    rng = np.random.default_rng(42)
    for _ in range(5):
        q = rng.uniform(-0.8, 0.8, size=6)
        result = check_jacobian(robot, q, eps=1e-6, tol=2e-5)
        assert result.pos_max_err < 1e-5, result.abs_err
        assert result.rot_max_err < 1e-5, result.abs_err


def test_jacobian_mid_chain_link():
    """中间 link 的雅可比只反映其上游关节，与该 link 作为子链末端的 FD 一致。"""
    robot = load_urdf_string(six_revolute_urdf())
    q = np.array([0.2, -0.3, 0.5, 0.1, -0.4, 0.0])
    j_mid = jacobian(robot, q, link_name="l3")

    sub = robot.extract_chain(end_link="l3")
    fk_sub = forward_kinematics(sub, q[:3])
    j_sub_fd = finite_difference_jacobian(sub, q[:3])
    assert np.allclose(j_mid[:, :3], j_sub_fd, atol=1e-6)
    # 下游关节列应为 0
    assert np.allclose(j_mid[:, 3:], 0.0)
    # 并且确实等于 l3 自身位姿
    assert np.allclose(fk_sub.end_transform,
                       forward_kinematics(robot, q).link_map["l3"])


def test_jacobian_singular_axis_cross():
    """两轴共线且关节点重合时，某列线性速度可为零（边界情形不报错）。"""
    urdf = build_urdf([
        {"name": "j1", "type": "revolute", "child": "l1", "axis": "0 0 1"},
        {"name": "j2", "type": "revolute", "child": "l2", "axis": "0 0 1"},
    ])
    robot = load_urdf_string(urdf)
    q = np.array([0.0, 0.0])
    j = jacobian(robot, q)
    # 零位末端在原点：两个 z 轴转动对位置均无贡献
    assert np.allclose(j[:3, :], 0.0)
    result = check_jacobian(robot, q, tol=1e-6)
    assert result.passed
