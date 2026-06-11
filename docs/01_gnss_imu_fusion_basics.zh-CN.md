# GNSS/IMU 融合基础

## 问题定义

GNSS/IMU 组合导航的目标是估计运动载体的导航状态。它融合两类传感器：

- IMU 测量：高频加速度和角速度
- GNSS 测量：低频绝对位置、速度，或由伪距等观测推导出的定位结果

IMU 的优势是短时间内连续、平滑、频率高；缺点是积分误差会随时间快速累积，尤其受传感器 bias 影响明显。GNSS 的优势是提供全局约束；缺点是低频、噪声较大，并且可能受到遮挡、多路径、延迟和失锁影响。

融合的核心思想是利用二者互补性：IMU 负责短期传播，GNSS 负责长期约束漂移。

在机器人定位中，同样的思想也会出现在移动机器人、无人机、自动驾驶和 SLAM 系统里。IMU 提供局部运动连续性，GNSS、LiDAR、Camera 等外部观测提供长期约束。

## 常见状态变量

一个典型的三维导航状态可以写成：

```text
x = [p, v, q, b_a, b_g]
```

其中：

- `p`：位置，3 x 1
- `v`：速度，3 x 1
- `q`：姿态四元数，4 x 1 nominal 表示
- `b_a`：加速度计 bias，3 x 1
- `b_g`：陀螺仪 bias，3 x 1

如果姿态使用四元数，则 nominal state 维度为：

```text
dim(x_nominal) = 3 + 3 + 4 + 3 + 3 = 16
```

在 Error-State Kalman Filter 中，姿态误差通常使用三维小角度向量表示：

```text
delta_x = [delta_p, delta_v, delta_theta, delta_b_a, delta_b_g]
```

误差状态维度为：

```text
dim(delta_x) = 3 + 3 + 3 + 3 + 3 = 15
```

这是非常常见的面试点：nominal state 可以用四元数维护姿态，但协方差通常维护在局部最小误差状态上。

## IMU 测量模型

加速度计和陀螺仪测量可以简化建模为：

```text
a_m = a_true + b_a + n_a
w_m = w_true + b_g + n_g
```

其中：

- `a_m`：测得的比力
- `w_m`：测得的角速度
- `b_a`、`b_g`：缓慢变化的 bias
- `n_a`、`n_g`：白噪声

一个简单的 bias 随机游走模型为：

```text
b_a(k+1) = b_a(k) + n_ba
b_g(k+1) = b_g(k) + n_bg
```

## GNSS 测量模型

第一版教育实现可以采用松组合模型，假设 GNSS 直接给出位置：

```text
z_gnss = p + n_gnss
```

对于 error-state filter，对应的观测矩阵可以写成：

```text
H = [I_3, 0_3, 0_3, 0_3, 0_3]
```

这是一个适合入门的起点。后续如果做紧组合，可以进一步引入伪距、Doppler、载波相位、接收机钟差和钟漂。

## 融合直觉

预测步骤由 IMU 积分驱动：

```text
x_hat(k+1) = f(x_hat(k), u_imu(k), dt)
P(k+1) = F P(k) F^T + G Q G^T
```

更新步骤由 GNSS 观测驱动：

```text
y = z - h(x_hat)
S = H P H^T + R
K = P H^T S^-1
delta_x = K y
P = (I - K H) P
```

对于 ESKF，`delta_x` 会被注入 nominal state，然后误差状态重新置零。

## 推导路线

1. 定义导航坐标系和传感器坐标系。
2. 写出 IMU 驱动的连续时间 nominal dynamics。
3. 在 nominal state 附近定义小误差变量。
4. 对误差动力学线性化，得到 `F` 和 `G`。
5. 将连续时间误差模型离散化。
6. 定义 GNSS 观测残差。
7. 使用 Kalman 更新修正误差状态。
8. 将修正量注入位置、速度、姿态和 bias。

## 实现时必须记录的细节

- 单位：米、秒、弧度。
- 重力方向：明确 `z` 轴向上还是向下。
- 坐标系：ENU、NED、ECEF 或局部切平面。
- 四元数约定：scalar-first 还是 scalar-last。
- 时间戳：IMU 高频，GNSS 低频。
- 噪声参数：区分测量噪声和 bias random walk。
- 可复现性：合成数据使用固定随机种子。

## 常见面试问题

- 为什么纯 IMU dead reckoning 会快速漂移？
- GNSS 低频且有噪声，为什么仍然有价值？
- EKF 和 ESKF 的区别是什么？
- 为什么姿态误差常用三维向量，而不是四维四元数？
- 协方差在物理上代表什么？
- 加速度计 bias 和陀螺仪 bias 分别如何影响位置误差？
- GNSS outage 期间滤波器会发生什么？
- 如何检查滤波器是否一致？

## 常见错误

- 混用角度和弧度。
- 混用 ENU 和 NED 坐标系。
- 忘记重力补偿。
- 把加速度计输出直接当成世界坐标系加速度。
- 姿态四元数更新后忘记归一化。
- 把测量噪声和过程噪声混淆。
- 忽略 GNSS 和 IMU 的时间同步。
- 只报告最终误差，不画漂移和更新过程。
