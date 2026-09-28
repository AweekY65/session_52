"""雅可比测试：解析公式、有限差分验证、块结构、prismatic、奇异点。"""

import numpy as np
import pytest

from robot_kinematics import (
    KinematicChain,
    finite_difference_jacobian,
    jacobian,
    parse_urdf,
)

from .urdfs import arm_6r_urdf, planar_arm_urdf, prismatic_urdf


def make_chain(urdf_text: str) -> KinematicChain:
    return KinematicChain(parse_urdf(urdf_text))


def analytic_planar_jacobian(q, lengths):
    """经典平面 nR（z 轴关节）的 6xN 几何雅可比。"""
    q = np.asarray(q, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    n = len(q)
    cum = np.cumsum(q)  # cum[i] = q0+...+qi
    j = np.zeros((6, n))
    for col in range(n):
        # 第 col 列：只累加 col 及之后连杆的贡献
        for i in range(col, n):
            j[0, col] += -lengths[i] * np.sin(cum[i])
            j[1, col] += lengths[i] * np.cos(cum[i])
        j[5, col] = 1.0  # 角速度 z 分量
    return j


def test_planar_jacobian_analytic_columns():
    lengths = np.array([1.0, 0.7])
    chain = make_chain(planar_arm_urdf(2, lengths))
    q = np.array([0.4, -0.9])
    j = jacobian(chain, q)
    expected = analytic_planar_jacobian(q, lengths)
    np.testing.assert_allclose(j, expected, atol=1e-12)


def test_jacobian_shape_and_planar_block_structure():
    chain = make_chain(planar_arm_urdf(3, [1.0, 0.8, 0.6]))
    q = np.array([0.2, 0.5, -0.3])
    j = jacobian(chain, q)
    assert j.shape == (6, 3)
    # 平面 z 轴关节：z 方向线速度、x/y 角速度恒为 0；z 角速度全 1
    np.testing.assert_allclose(j[2, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(j[3, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(j[4, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(j[5, :], 1.0, atol=1e-12)


def test_prismatic_jacobian_columns():
    chain = make_chain(prismatic_urdf())
    j = jacobian(chain, [0.4])
    assert j.shape == (6, 1)
    # prismatic：平移列 = 轴方向，旋转列为 0
    np.testing.assert_allclose(j[:3, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(j[3:, 0], 0.0, atol=1e-12)


def test_fixed_joints_do_not_add_columns():
    # planar_arm_urdf 链尾带一个 fixed joint；雅可比列数只能等于可动关节数
    chain = make_chain(planar_arm_urdf(3, [1.0, 0.8, 0.6]))
    j = jacobian(chain, [0.1, 0.2, 0.3])
    assert j.shape[1] == 3


@pytest.mark.parametrize(
    "n_joints,lengths",
    [
        (2, np.array([1.0, 0.7])),
        (3, np.array([1.0, 0.8, 0.6])),
        (6, None),
    ],
)
def test_analytic_jacobian_matches_finite_difference(n_joints, lengths):
    if n_joints == 6:
        chain = make_chain(arm_6r_urdf())
        q = np.array([0.2, -0.3, 0.5, 0.8, -0.4, 0.1])
    else:
        chain = make_chain(planar_arm_urdf(n_joints, lengths))
        q = np.linspace(0.3, -0.4, n_joints)
    j_analytic = jacobian(chain, q)
    j_numeric = finite_difference_jacobian(chain, q)
    np.testing.assert_allclose(j_analytic, j_numeric, atol=1e-6, rtol=1e-5)


def test_numerical_jacobian_helper_values():
    # 单摆（1R 绕 z，杆长 L）：J = [[-L sin q], [L cos q], [0], [0], [0], [1]]
    chain = make_chain(planar_arm_urdf(1, [0.8]))
    q = np.array([0.6])
    j_num = finite_difference_jacobian(chain, q)
    expected = np.array(
        [[-0.8 * np.sin(0.6)], [0.8 * np.cos(0.6)], [0.0], [0.0], [0.0], [1.0]]
    )
    np.testing.assert_allclose(j_num, expected, atol=1e-6)
    # 数值雅可比与解析雅可比也应一致
    np.testing.assert_allclose(jacobian(chain, q), expected, atol=1e-9)


def test_jacobian_near_singularity_finite():
    # 全伸展（q=0）是平面臂的奇异构型：雅可比必须有限且不含 NaN
    chain = make_chain(planar_arm_urdf(3, [1.0, 0.8, 0.6]))
    j = jacobian(chain, [0.0, 0.0, 0.0])
    assert np.all(np.isfinite(j))
    # 秩亏但数值合法
    assert np.linalg.matrix_rank(j[:3]) < 3
    # 6R 在零位附近同样有限
    chain6 = make_chain(arm_6r_urdf())
    j6 = jacobian(chain6, np.zeros(6))
    assert np.all(np.isfinite(j6))
