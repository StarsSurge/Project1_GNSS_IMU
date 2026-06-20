# IMU 名义状态更新 MVP：基础组件

## 1. 本阶段要回答的问题

在完整 GNSS/IMU ESKF 之前，首先要回答一个更小的问题：

> 已知上一时刻的姿态、速度、位置和 IMU bias，如何把 IMU 的角增量与速度增量变成下一步名义状态？

本阶段补齐基础组件，并在 `python/examples/demo_imu_state_update_mvp.py`
中实现显式的两子样名义状态传播原型。IMU 状态更新最容易出错的地方
不是代码量，而是单位、时间归属、坐标系、四元数乘法顺序和重力符号。
一旦这些基础约定错了，后面的协方差传播和 GNSS 更新都会在错误模型上工作。

## 2. 物理直觉

陀螺仪告诉我们“机体系在这小段时间转了多少”，加速度计告诉我们“机体系在这小段时间积累了多少比力速度增量”。导航解算要做两件事：

1. 用角增量更新 body 到 NED 的姿态。
2. 用姿态把机体系速度增量转到 NED，再加回重力，更新速度和位置。

类比成手里拿着一个小箭头：陀螺仪描述箭头如何旋转，加速度计描述箭头坐标系里感受到的推力。类比的边界是：IMU 的旋转不是普通平面转动，三维旋转不可交换，所以两帧角增量不能只做普通相加。

## 3. 当前代码位置

基础组件位于：

```text
python/gnss_imu/imu_mechanization.py
```

测试位于：

```text
tests/python/test_imu_mechanization.py
```

当前模块提供：

- `NavigationState`
- `IMUIncrement`
- `TwoSampleCorrection`
- `finite_vector`
- `positive_dt`
- `normalize_quat`
- `skew`
- `quat_multiply`
- `rotvec_to_quat`
- `quat_to_dcm`
- `euler_zyx_to_quat`
- `bias_correct_increment`
- `correct_two_sample_increments`

## 4. 定义和约定

本仓库在该 MVP 中采用：

- 导航系：局部 NED。
- 姿态四元数：Hamilton 顺序 `[w, x, y, z]`。
- `q_bn`：表示 body 到 NED 的旋转。
- `C_bn`：把 body 向量转到 NED，满足 `v_n = C_bn @ v_b`。
- IMU 输入：增量而非 rate。
- 角增量单位：rad。
- 速度增量单位：m/s。
- 陀螺 bias 单位：rad/s。
- 加速度计 bias 单位：m/s^2。

如果原始数据是角速度和加速度，需要先乘以采样间隔变成增量。

## 5. 最小标量例子：为什么 bias 要乘以 dt

假设一维陀螺仪静止，但有常值 bias：

```text
omega_m = b_g
```

如果采样间隔是 `dt`，IMU 给出的角增量是：

```text
dtheta_m = omega_m * dt = b_g * dt
```

真实角增量应该为零，因此补偿应为：

```text
dtheta = dtheta_m - b_g * dt
```

单位检查：

```text
[rad] = [rad] - [rad/s] * [s]
```

加速度计同理：

```text
dvel = dvel_m - b_a * dt
```

其中：

```text
[m/s] = [m/s] - [m/s^2] * [s]
```

## 6. 双子样修正的意义

两帧 IMU 增量在三维旋转中不可简单相加。若第一帧绕 x 轴小转角，第二帧绕 y 轴小转角，最终会产生一个二阶的 z 轴旋转项。代码中的圆锥修正为：

```text
dtheta = dtheta1 + dtheta2 + 2/3 * (dtheta1 x dtheta2)
```

`2/3` 系数来自双子样、线性角速度变化假设下的二阶积分近似。它不是任意调参；如果省略这个项，高动态振动下姿态会出现系统性误差。

速度增量也类似。机体系在一边旋转一边积累速度增量时，速度增量方向本身也在转动，因此需要划桨和旋转补偿：

```text
sculling = 2/3 * (dtheta1 x dvel2 + dvel1 x dtheta2)
rotation = 1/2 * ((dtheta1 + dtheta2) x (dvel1 + dvel2))
```

本阶段代码只计算这些修正量，不把它们直接积分到位置速度中。

## 7. 代码映射

`IMUIncrement` 表示一帧增量：

```text
dtheta : (3,) [rad]
dvel   : (3,) [m/s]
dt     : scalar [s]
```

`NavigationState` 表示名义状态：

```text
p_n  : (3,) [m]
v_n  : (3,) [m/s]
q_bn : (4,) unit quaternion
b_a  : (3,) [m/s^2]
b_g  : (3,) [rad/s]
```

`correct_two_sample_increments()` 的输出是后续状态更新函数应使用的中间量：

```text
corrected.dtheta
corrected.dvel
corrected.dt
```

经典 `2/3` 双子样系数要求两个子样本近似等间隔。当前实现允许小量时间戳
抖动，但会拒绝明显不相等的 `imu1.dt` 和 `imu2.dt`，避免把等间隔公式
静默用于不规则采样。

原型中的 `propagate_two_sample_mvp()` 使用这些量完成：

```text
q_new = q_old ⊗ Exp(corrected.dtheta)
dvel_n = C_bn(q_old) @ corrected.dvel
v_new = v_old + dvel_n + g_n * dt
p_new = p_old + 0.5 * (v_old + v_new) * dt
```

## 8. 可失败验证

当前基础组件的测试覆盖：

- 错误维度、NaN、非正时间间隔会被拒绝。
- 四元数会被归一化，零四元数会被拒绝。
- `skew(a) @ b` 等于 `np.cross(a, b)`。
- 90 度 yaw 会把 body x 轴转到 NED east。
- 四元数乘法与 DCM 连乘一致。
- bias 补偿确实使用每帧自己的 `dt`。
- 双子样圆锥、划桨和旋转交叉项非零且可检查。
- 明显不等间隔的两个子样会被拒绝。
- 静止传播保持位置、速度和姿态不变。
- 自由落体在 NED down 方向按重力增长。
- 恒定 yaw 的旋转方向符合 `q_bn` 右乘约定。
- 实测输入只使用初始化时刻之后的 IMU 增量。

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\python\test_imu_mechanization.py `
  tests\python\test_imu_state_update_mvp.py -v
```

## 9. 常见错误

1. 把加速度计测量当成加速度，而不是比力。
2. 静止时把 `dvel` 写成 `[0, 0, +g*dt]`，导致重力加了两次。
3. 混淆 `C_bn` 和 `C_nb`。
4. 把 `q_new = dq ⊗ q_old` 写成左乘，但文档约定是右乘 body 增量。
5. 用 rate 直接调用增量接口。
6. 忘记 bias 单位是每秒量，补偿增量时必须乘 `dt`。

## 10. 下一步

当前原型已通过静止、自由落体、恒定 yaw 和两帧实测数据 sanity check。
下一步是在更长的连续 IMU 序列上回放，并与 `truth.nav` 对齐评估速度和姿态
误差；通过后再把原型提升为正式库函数，并进入误差状态协方差传播。
