"""正运动学 (FK) 测试：解析解对比、原点/rpy 传播、fixed 关节、字典输入、限位检查。"""

from __future__ import annotations

import numpy as np
import pytest

from arm_kinematics import (
    JointType,
    forward_kinematics,
    load_urdf_string,
    make_transform,
    rpy_to_matrix,
)

from build_urdfs import build_urdf, planar_2r_urdf, three_joint_mixed_urdf


def test_planar_2r_fk_analytic():
    """平面 2R：FK 与解析公式 x=l1 c1 + l2 c12, y=..., z=0 一致。"""
    robot = load_urdf_string(planar_2r_urdf(1.0, 1.0))
    q = np.array([0.3, -0.7])
    t = forward_kinematics(robot, q).end_transform

    c1, s1 = np.cos(q[0]), np.sin(q[0])
    c12, s12 = np.cos(q.sum()), np.sin(q.sum())
    expected_xy = np.array([c1 + c12, s1 + s12, 0.0])

    assert np.allclose(t[:3, 3], expected_xy, atol=1e-12)
    # 平面转动：z 轴保持不变，末端姿态为绕 z 转 q1+q2
    assert np.allclose(t[:3, :3], rpy_to_matrix([0, 0, q.sum()]), atol=1e-12)
    # SE(3) 合法性
    assert np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-12)
    assert np.allclose(t[3], [0, 0, 0, 1])


def test_fk_chain_link_order_and_identity_base():
    robot = load_urdf_string(planar_2r_urdf())
    assert robot.link_names() == ["base_link", "l1", "l2", "end_effector"]
    res = forward_kinematics(robot, [0.1, 0.2])
    assert np.allclose(res.link_map["base_link"], np.eye(4))
    # l1 框架就在 j1 轴处：零偏移，只有旋转
    assert np.allclose(res.link_map["l1"][:3, 3], [0, 0, 0], atol=1e-12)
    # l2 框架在 j2 轴处：只由 q1 和第一段杆长决定
    assert np.allclose(res.link_map["l2"][:3, 3],
                       [np.cos(0.1), np.sin(0.1), 0.0], atol=1e-12)
    # 末端含第二段杆长
    assert np.allclose(
        res.end_transform[:3, 3],
        [np.cos(0.1) + np.cos(0.3), np.sin(0.1) + np.sin(0.3), 0.0],
        atol=1e-12,
    )


def test_fk_zero_configuration_origin_offsets():
    """零位时 FK 只由各 origin 平移累加决定。"""
    urdf = build_urdf([
        {"name": "j1", "type": "revolute", "child": "l1",
         "xyz": "1 0 0", "axis": "0 0 1"},
        {"name": "j2", "type": "revolute", "child": "l2",
         "xyz": "0 2 0", "axis": "0 0 1"},
    ])
    robot = load_urdf_string(urdf)
    t = forward_kinematics(robot, [0.0, 0.0]).end_transform
    assert np.allclose(t[:3, 3], [1, 2, 0])
    assert np.allclose(t[:3, :3], np.eye(3))


def test_origin_rpy_transforms_axis():
    """origin rpy=90deg 绕 z 后，关节轴 (1,0,0) 在基座系下应指向 -y。"""
    urdf = build_urdf([
        {"name": "j1", "type": "revolute", "child": "l1",
         "rpy": "0 0 1.5707963267948966", "axis": "1 0 0"},
    ])
    robot = load_urdf_string(urdf)
    theta = 0.4
    t = forward_kinematics(robot, [theta]).end_transform
    # 基座系旋转 = Rz(90°) @ Rx(theta)
    expected = rpy_to_matrix([0, 0, np.pi / 2]) @ rpy_to_matrix([theta, 0, 0])
    assert np.allclose(t[:3, :3], expected, atol=1e-12)


def test_fixed_joint_offset_propagates():
    """fixed 关节不占自由度，但其 xyz/rpy 必须沿链传播。"""
    robot = load_urdf_string(three_joint_mixed_urdf())
    assert robot.nq == 3
    assert [j.joint_type for j in robot.chain] == [
        JointType.REVOLUTE, JointType.FIXED,
        JointType.CONTINUOUS, JointType.PRISMATIC,
    ]
    res = forward_kinematics(robot, [0.0, 0.0, 0.0])
    # 零位：rev=0, fix 平移 (0.3,0,0) + rpy z 90°, cont=0, prism 平移 (0.2,0,0)
    # prism 的 (0.2,0,0) 定义在 fixed 偏转后的框架里 -> 基座系 (0, 0.2, 0)
    expected_xy = np.array([0.3, 0.2, 0.1])
    assert np.allclose(res.end_transform[:3, 3], expected_xy, atol=1e-12)
    assert np.allclose(
        res.end_transform[:3, :3], rpy_to_matrix([0, 0, np.pi / 2]), atol=1e-12
    )


def test_prismatic_translates_along_rotated_axis():
    urdf = build_urdf([
        {"name": "jp", "type": "prismatic", "child": "l1",
         "xyz": "0 0 1", "rpy": "0 1.5707963267948966 0",
         "axis": "1 0 0", "lower": -1.0, "upper": 1.0},
    ])
    robot = load_urdf_string(urdf)
    d = 0.6
    t = forward_kinematics(robot, [d]).end_transform
    # Ry(90°) 把 x 轴转到 -z 方向（URDF 固定角 Rz Ry Rx）
    expected_axis = rpy_to_matrix([0, np.pi / 2, 0]) @ np.array([1, 0, 0])
    assert np.allclose(t[:3, 3], [0, 0, 1] + d * expected_axis, atol=1e-12)
    assert np.allclose(t[:3, :3], rpy_to_matrix([0, np.pi / 2, 0]))


def test_fk_dict_input_and_continuous_unwrapped():
    robot = load_urdf_string(three_joint_mixed_urdf())
    t1 = forward_kinematics(robot, {"j_rev": 0.1, "j_cont": 0.2, "j_prism": 0.0})
    t2 = forward_kinematics(robot, [0.1, 0.2, 0.0])
    assert np.allclose(t1.end_transform, t2.end_transform)
    # continuous 关节允许多圈
    t3 = forward_kinematics(robot, [0.1, 0.2 + 2 * np.pi, 0.0])
    assert np.allclose(t3.end_transform, t2.end_transform, atol=1e-10)


def test_check_limits_raises():
    robot = load_urdf_string(planar_2r_urdf())
    forward_kinematics(robot, [4.0, 0.0])  # 默认不检查，不报错
    with pytest.raises(ValueError, match="超出限位"):
        forward_kinematics(robot, [4.0, 0.0], check_limits=True)
    # 边界值合法
    forward_kinematics(robot, [np.pi, -np.pi], check_limits=True)


def test_fk_wrong_q_size():
    robot = load_urdf_string(planar_2r_urdf())
    with pytest.raises(ValueError, match="数量"):
        forward_kinematics(robot, [0.1])
