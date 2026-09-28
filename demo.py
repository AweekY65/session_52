"""命令行演示：加载本地 URDF，运行 FK / Jacobian 有限差分 / IK。

用法（在工程根目录）::

    python demo.py                      # 默认使用 urdf/arm6r.urdf
    python demo.py urdf/arm2r.urdf
"""

from __future__ import annotations

import os
import sys

import numpy as np

from arm_kinematics import (
    check_jacobian,
    forward_kinematics,
    inverse_kinematics,
    load_urdf,
)


def main() -> int:
    urdf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "urdf", "arm6r.urdf"
    )
    print(f"加载本地 URDF: {urdf_path}")
    robot = load_urdf(urdf_path)
    print(f"  模型名称 : {robot.name}")
    print(f"  运动链   : {' -> '.join(robot.link_names())}")
    print(f"  活动关节 : {[j.name for j in robot.movable_joints]}  (nq={robot.nq})")

    # 构造一个落在各活动关节限位内的初始构型
    q = np.zeros(robot.nq)
    for i, j in enumerate(robot.movable_joints):
        if j.joint_type.value == "continuous":
            q[i] = 0.1 * (i + 1)
        else:
            q[i] = 0.5 * (j.lower + j.upper) + \
                0.1 * (j.upper - j.lower) * ((-1) ** i)
    fk = forward_kinematics(robot, q)
    print("\n[FK] 关节角 q =", np.round(q, 4))
    print("     末端位姿 T_ee =")
    print(np.array2string(fk.end_transform, precision=5, suppress_small=True))

    check = check_jacobian(robot, q)
    print("\n[Jacobian 有限差分验证]")
    print(f"  解析雅可比:\n{np.array2string(check.jac_analytic, precision=5, suppress_small=True)}")
    print(f"  数值雅可比:\n{np.array2string(check.jac_numeric, precision=5, suppress_small=True)}")
    print(f"  位置部分最大误差: {check.pos_max_err:.3e}")
    print(f"  旋转部分最大误差: {check.rot_max_err:.3e}")
    print(f"  结论: {'通过' if check.passed else '未通过'}")

    # IK：用 FK(q_goal) 构造一个必然可达的目标（每个关节只移动一小步，
    # prismatic 关节的步长额外保证不越过自己的限位）
    q_goal = q.copy()
    for i, j in enumerate(robot.movable_joints):
        if j.joint_type.value == "prismatic":
            headroom = max(j.upper - q[i], 0.0)
            q_goal[i] = q[i] + min(0.05, 0.5 * headroom)
        else:
            q_goal[i] = q[i] + 0.05
    target = forward_kinematics(robot, q_goal).end_transform
    print("\n[IK] 目标由 q_goal =", np.round(q_goal, 4), " 的 FK 构造")
    result = inverse_kinematics(robot, target, q_init=q,
                                tol_pos=1e-8, tol_rot=1e-8,
                                max_iterations=300, restarts=10, seed=0)
    print(f"  状态: {result.status.value}  success={result.success}"
          f"  随机重启: {result.restarts_used}")
    print(f"  迭代: {result.iterations}  "
          f"位置误差: {result.pos_error:.3e} m  姿态误差: {result.rot_error:.3e} rad")
    if result.q is not None:
        print("  求解关节角:", np.round(result.q, 5))
    print("  说明:", result.message)

    # 不可达目标演示
    far = np.eye(4)
    far[0, 3] = 1e4
    bad = inverse_kinematics(robot, far)
    print("\n[IK] 10 km 外的不可达目标:")
    print(f"  状态: {bad.status.value}  success={bad.success}  ({bad.message})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
