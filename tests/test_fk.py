"""正向运动学测试：解析公式对照、fixed joint、prismatic、限位、连续关节。"""

import numpy as np
import pytest

from robot_kinematics import (
    KinematicChain,
    KinematicsError,
    fk_all,
    forward_kinematics,
    parse_urdf,
)

from .urdfs import fixed_only_urdf, planar_arm_urdf, prismatic_urdf


def make_chain(urdf_text: str, **kwargs) -> KinematicChain:
    return KinematicChain(parse_urdf(urdf_text), **kwargs)


def analytic_planar_tip(q, lengths):
    """经典平面 nR 末端位置（关节 i 在连杆 i 起点）。"""
    q = np.asarray(q, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    angle = np.cumsum(q)
    x = float(np.sum(lengths * np.cos(angle)))
    y = float(np.sum(lengths * np.sin(angle)))
    return np.array([x, y, 0.0])


@pytest.mark.parametrize(
    "n_joints,lengths",
    [
        (2, np.array([1.0, 0.7])),
        (2, np.array([0.5, 0.5])),
        (3, np.array([1.0, 0.8, 0.6])),
    ],
)
def test_planar_fk_matches_analytic(n_joints, lengths):
    chain = make_chain(planar_arm_urdf(n_joints, lengths))
    rng = np.random.default_rng(42)
    for _ in range(8):
        q = rng.uniform(-2.5, 2.5, n_joints)
        tip = forward_kinematics(chain, q)
        expected = analytic_planar_tip(q, lengths)
        np.testing.assert_allclose(tip[:3, 3], expected, atol=1e-12)
        # 末端姿态 = 绕 z 旋转 sum(q)
        assert tip[0, 0] == pytest.approx(np.cos(np.sum(q)), abs=1e-12)
        assert tip[1, 0] == pytest.approx(np.sin(np.sum(q)), abs=1e-12)


def test_zero_configuration_pose():
    lengths = np.array([1.0, 0.8, 0.6])
    chain = make_chain(planar_arm_urdf(3, lengths))
    tip = forward_kinematics(chain, np.zeros(3))
    # 零位：所有关节角为 0，末端在 x 轴上，距离为连杆长度之和
    np.testing.assert_allclose(tip[:3, 3], [lengths.sum(), 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(tip[:3, :3], np.eye(3), atol=1e-12)


def test_fk_returns_all_link_poses_in_chain():
    chain = make_chain(planar_arm_urdf(3, [1.0, 0.8, 0.6]))
    q = np.array([0.3, -0.4, 0.2])
    poses = fk_all(chain, q)
    assert set(poses) == {"base", "l0", "l1", "l2", "tip"}
    np.testing.assert_allclose(poses["base"], np.eye(4), atol=1e-12)
    # 逐级与解析结果对照
    np.testing.assert_allclose(poses["l0"][:3, 3], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(
        poses["l1"][:3, 3], [np.cos(q[0]), np.sin(q[0]), 0], atol=1e-12
    )
    # fk_all 的 tip 与 forward_kinematics 一致
    np.testing.assert_allclose(poses["tip"], forward_kinematics(chain, q), atol=1e-12)
    # 指定中间 link
    mid = forward_kinematics(chain, q, link_name="l1")
    np.testing.assert_allclose(mid, poses["l1"], atol=1e-12)


def test_fixed_joints_in_chain_contribute_to_pose():
    # 链中间夹一个带旋转的 fixed joint，它必须参与位姿传播
    urdf = """<robot name="r">
      <link name="base"/><link name="a"/><link name="tip"/>
      <joint name="j1" type="revolute">
        <parent link="base"/><child link="a"/>
        <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
      </joint>
      <joint name="bend" type="fixed">
        <parent link="a"/><child link="tip"/>
        <origin xyz="0.5 0 0" rpy="0 0 1.5707963267948966"/>
      </joint>
    </robot>"""
    chain = make_chain(urdf)
    assert chain.dof == 1
    tip0 = forward_kinematics(chain, [0.0])
    # q=0：fixed 把 tip 平移 (0.5,0,0) 并旋转 90°
    np.testing.assert_allclose(tip0[:3, 3], [0.5, 0.0, 0.0], atol=1e-12)
    assert tip0[0, 0] == pytest.approx(0.0, abs=1e-12)  # cos(90°)
    # q=pi/2：整体再转 90°，tip 位置转到 y 轴
    tip90 = forward_kinematics(chain, [np.pi / 2])
    np.testing.assert_allclose(tip90[:3, 3], [0.0, 0.5, 0.0], atol=1e-12)


def test_fixed_tip_offset_transformed_by_wrist_rotation():
    # 单旋转关节 + fixed 末端偏移：旋转关节时末端沿圆弧运动
    chain = make_chain(planar_arm_urdf(2, [1.0, 0.5]))
    q = np.array([0.0, np.pi / 2])
    tip = forward_kinematics(chain, q)
    # 第二关节转 90° 后，0.5 的 tip 偏移指向 +y（相对 l1 系旋转累计 90°）
    np.testing.assert_allclose(tip[:3, 3], [1.0, 0.5, 0.0], atol=1e-12)


def test_prismatic_fk_translates_along_axis():
    chain = make_chain(prismatic_urdf())
    tip = forward_kinematics(chain, [0.4])
    np.testing.assert_allclose(tip[:3, 3], [0.4, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(tip[:3, :3], np.eye(3), atol=1e-12)
    # 带 rpy 的 origin：平移方向应被 origin 旋转
    urdf = """<robot name="r">
      <link name="base"/><link name="tip"/>
      <joint name="slide" type="prismatic">
        <parent link="base"/><child link="tip"/>
        <origin xyz="0 0 0" rpy="0 0 1.5707963267948966"/>
        <axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/>
      </joint>
    </robot>"""
    chain2 = make_chain(urdf)
    tip2 = forward_kinematics(chain2, [0.3])
    # origin 旋转 90°：x 轴平移映射到世界 y 方向
    np.testing.assert_allclose(tip2[:3, 3], [0.0, 0.3, 0.0], atol=1e-12)


def test_continuous_joint_accepts_any_angle():
    urdf = """<robot name="r">
      <link name="base"/><link name="tip"/>
      <joint name="spin" type="continuous">
        <parent link="base"/><child link="tip"/>
        <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
      </joint>
    </robot>"""
    chain = make_chain(urdf)
    # 远超 2*pi 的角度也必须接受（continuous 无限位）
    tip = forward_kinematics(chain, [12.5 * np.pi])
    expected_angle = 12.5 * np.pi
    assert tip[0, 0] == pytest.approx(np.cos(expected_angle), abs=1e-9)
    assert tip[1, 0] == pytest.approx(np.sin(expected_angle), abs=1e-9)


def test_joint_limit_enforced():
    chain = make_chain(planar_arm_urdf(2, [1.0, 0.7]))
    with pytest.raises(KinematicsError, match="限位"):
        forward_kinematics(chain, [0.0, np.pi + 0.5])
    with pytest.raises(KinematicsError, match="限位"):
        forward_kinematics(chain, [-np.pi - 1.0, 0.0])
    # 恰好边界值允许
    forward_kinematics(chain, [np.pi, -np.pi])


def test_fixed_only_chain_zero_dof():
    chain = make_chain(fixed_only_urdf())
    assert chain.dof == 0
    tip = forward_kinematics(chain, [])
    np.testing.assert_allclose(tip[:3, 3], [0.3, 0.1, 0.0], atol=1e-12)
    assert tip[0, 0] == pytest.approx(np.cos(0.5), abs=1e-12)
