# 静止 IMU 初始化：调平、航向可观测性与 gyrocompassing

## 1. 动机

ESKF 开始传播前必须知道初始位置、速度、姿态和 bias。位置可来自 GNSS，静止
速度可设为零，但姿态不能凭空产生。本阶段回答：

> 静止 IMU 能提供哪些姿态信息，如何判断窗口真的静止，以及高精陀螺为什么
> 可以 gyrocompass，而普通 MEMS 通常不行？

实现：`python/gnss_imu/imu_initialization.py`。

## 2. 最小物理直觉

静止时，加速度计看到的比力方向与重力相反。因此把传感器慢慢倾斜，三轴分量
变化能告诉我们 roll 和 pitch。但绕铅垂线原地转动时，重力在 body 中几乎不变，
所以加速度计无法分辨 yaw。

地球自转提供了第二个不与重力平行的参考向量。高精陀螺长时间平均后能看到它，
于是“重力向量 + 地球自转向量”可以确定三维姿态。MEMS bias/noise 常与地球
自转同量级或更大，静态航向会非常不稳定。

## 3. 定义、单位和窗口检查

每个样本先还原 rate：

```text
omega_b = dtheta/dt   [rad/s]
f_b     = dvel/dt     [m/s^2]
```

窗口必须满足：时间严格递增、采样间隔近似均匀、样本数和持续时间足够、陀螺
与加速度计标准差低于门限、平均比力模长接近当地正常重力。

恒加速度运动可能伪装成“低方差”，所以真实系统还应结合轮速、GNSS 速度或
运动学约束，不能只看 IMU 方差。

## 4. 调平公式

FRD body、NED navigation 约定下，静止比力近似：

```text
f_b = C_nb * [0, 0, -g]
```

展开后：

```text
f_x =  g sin(pitch)
f_y = -g cos(pitch) sin(roll)
f_z = -g cos(pitch) cos(roll)
```

因此：

```text
roll  = atan2(-f_y, -f_z)
pitch = asin(f_x/|f|)
```

公式同时解释了符号：水平 FRD IMU 静止时 `f_z` 约为 `-g`。

## 5. yaw 的两条路径

### 5.1 外部航向

MEMS 推荐从双天线 GNSS、磁罗盘、运动航向、地图或人工配置获得 yaw。初始化
接口要求显式传入 `yaw_rad`，不会把不可观测 yaw 偷偷设为零。

### 5.2 高精陀螺 gyrocompassing

NED 中地球自转：

```text
omega_ie_n = [Omega cos(latitude), 0, -Omega sin(latitude)]
```

静止 body 测量：

```text
omega_ib_b ~= C_nb * omega_ie_n + b_g + noise
```

对重力和地球自转构造 TRIAD：分别在 body 和 NED 中建立两个正交基，再计算：

```text
C_bn = T_n * T_b^T
```

若陀螺 bias 或噪声淹没地球自转水平分量，TRIAD 航向会失效；这不是调参问题，
而是传感器信息不足。

## 6. bias 可观测性

已知完整姿态时，静止陀螺 bias 可估计为：

```text
b_g = mean(omega_m_b) - C_nb * omega_ie_n
```

单一静止姿态不能可靠区分水平加速度计 bias 与微小倾角误差，因此当前实现不从
一姿态窗口估计完整 `b_a`，而是保留外部标定先验。六位置标定才能更好分离它们。

## 7. 代码与验证

运行：

```powershell
.\.venv\Scripts\python.exe python\examples\demo_static_imu_alignment.py
```

dataset1 约 50 秒静止窗口的典型结果：roll/pitch 误差小于 `0.05 deg`，
gyrocompass yaw 误差约 `1.1 deg`。该结果只适用于此高精 IMU 和当前窗口。

测试覆盖解析调平、gyro bias、无 yaw 来源拒绝、运动窗口拒绝、DCM/四元数转换，
以及 dataset1 gyrocompass 与 truth 的定量比较。

## 8. 工程边界、常见错误和自检

当前仍缺自动窗口搜索、多窗口一致性、GNSS/轮速联合静止判定、温度稳定判定、
磁航向质量控制和在线精对准。

常见错误：把加速度计当重力而忘记比力符号；用 MEMS 静态陀螺强行算 yaw；
用同一窗口同时估姿态和全部 bias 后给出过小协方差；忽略预热和振动。

自检：

1. 为什么静止重力只能确定两个姿态自由度？
2. 为什么 gyrocompass 在极点附近和低等级陀螺上会退化？
3. 为什么单姿态无法区分水平加速度计 bias 与倾角？
4. 若窗口长度增加四倍，白噪声均值标准差如何变化？
