# arm_kinematics — 本地机械臂运动学工具

一个**完全离线**的机械臂运动学工具：机器人模型、关节状态和全部计算结果只来自本地
URDF 文件（或测试中工程内生成的 URDF 字符串）并保存在本地内存中，**不依赖
ROS、MoveIt、机器人硬件或任何外部服务**。核心运动学计算（FK / Jacobian / IK /
旋转数学）全部在工程内用 NumPy 实现，SciPy 不是依赖项。

## 目录结构

```
.
├── arm_kinematics/
│   ├── urdf_loader.py   # URDF 解析、运动链构建、结构校验
│   ├── transforms.py    # RPY/旋转矩阵/齐次变换/位姿误差等基础数学
│   ├── kinematics.py    # 正运动学 FK 与几何雅可比 Jacobian
│   ├── finite_diff.py   # 雅可比中心有限差分验证
│   └── ik.py            # 阻尼最小二乘逆运动学 IK
├── urdf/                # 随工程提供的本地示例模型
│   ├── arm2r.urdf       # 平面 2R
│   ├── arm3dof.urdf     # revolute + fixed + continuous + prismatic
│   └── arm6r.urdf       # 6R 肘型臂
├── tests/               # 全部自动化测试（含工程内生成 2/3/6 关节模型）
│   └── build_urdfs.py   # 测试用 URDF 生成器（不读外部文件）
├── demo.py              # 终端演示：FK / Jacobian 有限差分 / IK
├── requirements.txt
└── pytest.ini
```

## 安装

只需要 NumPy：

```bash
pip install -r requirements.txt   # numpy>=1.21
```

## 快速开始

```python
import numpy as np
from arm_kinematics import (
    load_urdf, forward_kinematics, jacobian,
    check_jacobian, inverse_kinematics,
)

robot = load_urdf("urdf/arm6r.urdf")
q = np.array([0.1, -0.2, 0.3, 0.0, 0.4, 0.1])

# ---- 正运动学 ----
fk = forward_kinematics(robot, q)
T_ee = fk.end_transform                 # 末端 4x4 齐次变换
T_l3 = fk.link_map["link_3"]            # 任意中间 link 的位姿

# 关节值也可以用字典给出（fixed 关节不占自由度）
fk2 = forward_kinematics(robot, {
    "joint_1": 0.1, "joint_2": -0.2, "joint_3": 0.3,
    "joint_4": 0.0, "joint_5": 0.4, "joint_6": 0.1,
})

# 需要严格检查关节限位时（revolute / prismatic）
forward_kinematics(robot, q, check_limits=True)

# ---- 雅可比及其有限差分验证 ----
J = jacobian(robot, q)                  # 6×nq 基座系几何雅可比
report = check_jacobian(robot, q, eps=1e-6, tol=1e-5)
print(report.pos_max_err, report.rot_max_err, report.passed)

# ---- 逆运动学 ----
target = forward_kinematics(robot, q + 0.05).end_transform
res = inverse_kinematics(
    robot, target, q_init=q,
    tol_pos=1e-8, tol_rot=1e-8,
    max_iterations=300, damping=1e-4, use_limits=True,
    restarts=5, seed=42,
)
if res.success:
    print("收敛于 q =", res.q, "（重启", res.restarts_used, "次）")
else:
    print("失败：", res.status.value, res.message)
```

运行终端演示（不需要 RViz 或任何可视化）：

```bash
python demo.py                  # 默认 6R
python demo.py urdf/arm2r.urdf
python demo.py urdf/arm3dof.urdf
```

## 坐标系与变换顺序

### 坐标系约定

- 全部为右手系；齐次变换统一表示为 4×4 矩阵
  `T = [R p; 0 0 0 1]`，其中 `R ∈ SO(3)`、`p ∈ R^3`。
- 每个 link 的位姿都表示在 **base_link 坐标系（基座系）** 下。
- URDF `<origin xyz="..." rpy="...">` 按 URDF/ROS 约定解释：**固定角 RPY**，
  等价旋转矩阵

  ```
  R_origin = Rz(yaw) · Ry(pitch) · Rx(roll)
  ```

  即相对父 link 框架先绕固定 x 轴 roll，再绕固定 y 轴 pitch，最后绕固定 z 轴 yaw。

- URDF 语义中，**child link 的框架位于上游关节的运动轴线处**；`<origin>`
  属于「父 link → 关节框架」的固定偏移。因此一个关节的完整变换为

  ```
  T_i(q_i) = T_origin_i · T_motion_i(q_i)
  ```

  | 关节类型      | `T_motion`                  | q 的含义       |
  |---------------|-----------------------------|----------------|
  | revolute      | `Rot(axis, q)`              | 弧度，受限位   |
  | continuous    | `Rot(axis, q)`              | 弧度，无限位   |
  | prismatic     | `Trans(axis · q)`           | 米，受限位     |
  | fixed        | `I`（恒等，不占自由度）     | —              |

  旋转用 Rodrigues 公式构造，`<axis>` 在解析时归一化；prismatic 沿关节框架
  （运动前）的轴平移。

- 沿链从基座到末端连乘：

  ```
  T_ee = T_1(q1) · T_2(q2) · ··· · T_n(qn)
  ```

  `fixed` 关节参与连乘（传播其 `xyz/rpy` 偏移），但不出现在广义坐标 `q` 中。

### 雅可比约定

雅可比是 **基座系几何雅可比**（6×nq），行顺序为
`[vx, vy, vz, wx, wy, wz]`，列顺序与 `robot.movable_joints` 一致（跳过
fixed）：

- revolute / continuous：

  ```
  Jv_i = z_i × (p_ee − p_i),  Jw_i = z_i
  ```

  其中 `z_i` 是基座系下的关节轴，`p_i` 是运动前关节轴上一点。
- prismatic：`Jv_i = z_i, Jw_i = 0`。

旋转误差与角速度采用同一约定：

```
e_rot = log( R_target · R_current^T )        # 基座系下的旋转向量（轴×角）
```

### 有限差分验证

`check_jacobian` 用**中心差分**（默认 `eps=1e-6`）分别核对雅可比的平移部分和
旋转部分：

```
Jv[:, i] ≈ (p(q+eps·e_i) − p(q−eps·e_i)) / (2 eps)
Jw[:, i] ≈ ( log(R(q+)R(q)^T) − log(R(q−)R(q)^T) ) / (2 eps)
```

返回的 `pos_max_err` / `rot_max_err` 分别是平移块和旋转块的逐元素最大误差；
测试中二者通常在 `1e-10 ~ 1e-6` 量级。

## IK 方法

阻尼最小二乘（Damped Least Squares，Levenberg–Marquardt 型），每轮迭代：

```
e     = [ p_target − p(q) ; log(R_target R(q)^T) ]      # 6 维基座系误差
J     = geometric_jacobian(robot, q)                    # 6×nq
Δq    = J^T (J J^T + λ² I)^−1 e                        # DLS 步长（对非方阵同样有效）
q_new = q + α Δq
```

工程实现细节：

- **自适应阻尼**：每轮用 `eigvalsh(JJ^T)` 取雅可比最小奇异值 σ_min 度量与
  奇异构型的距离（比完整 SVD 便宜），按
  `λ² = λ0² + max(0, 1−(σ_min/σ0)²)·λ_max²` 增大阻尼，远离奇异时阻尼很小、
  保持快速收敛。
- **回退线搜索 (backtracking)**：从 `α=1` 开始逐次减半（最多 50 次），只接受
  使**收敛尺度化误差** `‖e_pos‖/tol_pos + ‖e_rot‖/tol_rot` 严格相对下降的
  步长。直接把米与弧度相加会因量纲尺度不同而错误拒绝「以极小姿态代价换取
  大幅位置下降」的步长，所以线搜索判据按容差归一化。近奇异构型下阻尼步偏向
  错误零空间方向时，自动退回基础阻尼再试一次。
- **随机重启**：IK 是非凸问题，即使目标可达，从远处初值也可能落入局部驻点。
  线搜索彻底停滞时，求解器在关节限位内做均匀随机重启（默认 5 次，可用
  `restarts=` 调整，`seed=` 保证可复现），`IKResult.restarts_used` 报告
  实际重启次数。
- **关节限位**：`use_limits=True`（默认）时每步把 revolute/prismatic 关节
  钳制在 `<limit lower/upper>` 内；continuous 不钳制。
- `max_step=0.5`：单关节单步变化量上限（弧度/米），避免 Gauss-Newton 大步
  越过最优点。
- `damping=λ0`（默认 `1e-4`）：基础阻尼，奇异点附近自动放大（上限
  `max_damping=1e-3`）。
- 收敛判据：`‖e_pos‖ ≤ tol_pos` **且** `‖e_rot‖ ≤ tol_rot`。
- 对于欠驱动（nq<6）或带限位的臂，**工作空间内部的某些完整位姿（位置+姿态）
  本身就不可达**（例如平面 2R 末端 yaw 与末端位置在非全伸展构型下不独立）。
  这类问题的最小二乘目标存在非零驻点，所有重启都失败后求解器会明确返回
  `stalled`/`max_iterations`，绝不会把驻点伪装成成功。

### 返回状态（绝不伪造成功）

`inverse_kinematics` 返回 `IKResult`，其 `success` 为 True 当且仅当状态为
`CONVERGED`；失败时 `q` 只是「当前最优估计」，调用方不得当作可行解：

| status            | 含义                                                                 |
|-------------------|----------------------------------------------------------------------|
| `converged`       | 位置、姿态误差均低于容差                                             |
| `max_iterations`  | 达到 `max_iterations` 仍未收敛（含全部重启）                         |
| `stalled`         | 线搜索无法继续下降（卡限位、近奇异、目标姿态不可达、局部极小且重启耗尽） |
| `unreachable`     | 零自由度链无法满足目标；或目标点距离基座超过保守可达半径上界         |

可达半径上界由沿链的全部 origin 平移长度加 prismatic 行程求和得到（三角形
不等式的必然上界）：目标超过它**一定**不可达；不超过并不保证可达——这类情形
由 `stalled` / `max_iterations` 明确报告。

## URDF 支持范围与校验

- 解析：`<link>`、`<joint>`（`parent/child`）、`<origin xyz rpy>`、
  `<axis xyz>`（缺省 `(1,0,0)`，自动归一化）、`<limit lower upper effort
  velocity>`。
- 关节类型：`revolute`、`continuous`、`prismatic`、`fixed`；
  `floating` / `planar` / `<mimic>` 等会被明确拒绝。
- 结构校验（抛 `URDFError`）：缺失 parent/child link、重复 link/joint 名、
  自环、汇合/循环结构（一个 link 有多个父关节）、多根 link、非法零轴/非有限轴、
  revolute/prismatic 缺少限位或 `lower > upper`、指定的 base/end link 不存在、
  end link 不在 base 下游、XML 格式错误等。
- 默认链：唯一根 link 为 base；只有一个叶 link 时自动选为 end，多个叶 link
  时需用 `load_urdf(path, base_link=..., end_link=...)` 显式指定，也可对已
  加载模型调用 `robot.extract_chain(end_link=...)` 取分支子链。

## 运行测试

全部测试在终端完成，不依赖 RViz 或其他 GUI：

```bash
python -m pytest                 # 运行全部测试（53 项）
python -m pytest tests/test_fk.py
python -m pytest tests/test_jacobian.py -k finite_difference
python -m pytest tests/test_ik.py
python -m pytest tests/test_urdf_loader.py
```

测试覆盖：

- **FK**：2R 解析解对比、origin xyz/rpy 传播、fixed 关节偏移、prismatic 沿
  旋转后轴平移、continuous 多圈、字典/序列输入、限位检查、SE(3) 合法性。
- **Jacobian**：2R 闭式雅可比、2/3/6 关节模型在多个构型下的中心有限差分
  （平移块与旋转块分别断言）、中间 link 子链雅可比、奇异/共线边界。
- **IK**：2R 往返、不同构型保持、严格姿态容差下不可达姿态明确失败、工作空间
  内部姿态不可达驻点明确失败、超臂展目标 `unreachable`、关节限位收紧后失败
  且解不越限、3 自由度混合关节往返、6R 近邻初值高精度收敛、6R 难初值随机
  重启回归、零自由度 fixed 链、非法参数校验、本地 URDF 文件加载往返。
- **URDF 校验**：缺失 link、循环/汇合、自环、零轴、不支持的关节类型、限位
  缺失/倒置、重名、多根/多叶、非法 XML 等。

## 设计边界

- 这是一个数值运动学库，不做轨迹规划、碰撞检测或动力学。
- IK 是基础数值法（DLS + 线搜索 + 限位），依赖初值、不保证找到全局解；这是
  有意为之——当它无法证明达到目标时，返回明确的失败状态，而不是伪造成功。
- 所有输入输出均为本地 NumPy 数组；代码中没有任何网络、ROS、硬件相关调用。
