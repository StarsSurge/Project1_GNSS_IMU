# GNSS/IMU 松组合 ESKF：生产导向 MVP

## 1. 这个算法回答什么问题

IMU 高频提供角增量和速度增量，但误差会随积分累积；GNSS 低频提供有界位置
观测，但可能包含跳变、遮挡和天线杆臂。松组合 ESKF 回答：

> 如何保留非线性 INS 名义状态，同时用线性小误差状态描述不确定性，并在
> GNSS 到达时安全地修正位置、速度、姿态和 IMU bias？

实现位置：

```text
python/gnss_imu/loosely_coupled_eskf.py
python/examples/run_dataset1_eskf.py
```

成熟度是 **verified MVP / production-oriented baseline**。它包含 WGS-84、
地球自转、运输率、科氏项、杆臂、卡方门限、真实时间戳检查和协方差健康
检查，但还不是经过设备标定、现场测试和实时认证的部署产品。

## 2. 物理直觉

名义状态是当前最可信的完整轨迹，误差状态是附着在轨迹附近的小坐标尺。
IMU 到来时名义轨迹按非线性运动前进，协方差扩张；GNSS 到来时，滤波器估计
“名义轨迹现在偏了多少”，再把小修正注入名义状态。

这个直觉只适用于误差足够小。若初始航向差几十度、时间同步错一个采样周期，
或 GNSS 跳变未被拒绝，局部线性误差模型可能失效。

## 3. 最小一维例子

INS 位置 `10 m`，GNSS 测得 `12 m`：

```text
residual = 12 - 10 = 2 m
```

若 INS 方差 `4 m^2`，GNSS 方差 `1 m^2`：

```text
K       = 4/(4+1) = 0.8
delta_p = K * residual = 1.6 m
p_new   = 10 + 1.6 = 11.6 m
```

ESKF 的三维更新本质相同，只是状态相关性还会修正速度、姿态和 bias；杆臂
存在时，姿态误差也会改变预测的 GNSS 天线位置。

## 4. 状态、单位和约定

### 4.1 名义状态

```text
latitude, longitude : WGS-84 纬度/经度 [rad]
height               : 椭球高 [m]
v_n                  : NED 速度 [m/s]
q_bn                 : body -> NED Hamilton 四元数 [w,x,y,z]
b_a                  : 加速度计 bias [m/s^2]
b_g                  : 陀螺 bias [rad/s]
```

### 4.2 15 维右乘误差状态

```text
delta_x = [delta_p_n, delta_v_n, delta_theta_b, delta_b_a, delta_b_g]
维度       3          3          3              3          3
```

- `delta_p_n` 是局部 NED 米制误差，不是经纬度弧度误差。
- `delta_theta_b` 是 body 表达的右乘小姿态误差。
- 误差定义为 `true - nominal`。

### 4.3 IMU 输入

时间戳表示增量区间结束时刻：

```text
dtheta : rad
dvel   : m/s
dt     : s
```

rate 必须先积分成增量。两个子样要求近似等间隔，明显不等间隔会被拒绝。

## 5. 地球模型与名义状态传播

### 5.1 曲率半径

纬度 `L` 处：

```text
Rn = a / sqrt(1 - e^2 sin^2(L))
Rm = a(1-e^2) / (1 - e^2 sin^2(L))^(3/2)
```

它们把 NED 米制速度连接到纬度、经度变化率。

### 5.2 地球自转与运输率

```text
omega_ie_n = [Omega_E cos(L), 0, -Omega_E sin(L)]

omega_en_n = [
    v_E/(Rn+h),
   -v_N/(Rm+h),
   -v_E tan(L)/(Rn+h)
]
```

姿态同时包含导航系左侧旋转和 body 右侧旋转：

```text
q_new = Exp(-omega_in_n dt) ⊗ q_old ⊗ Exp(delta_theta_b)
omega_in_n = omega_ie_n + omega_en_n
```

静止在地球表面的 IMU 会测到地球自转。左右两部分正确时，`q_bn` 不应因
地球自转而漂移，测试会专门检查这一点。

### 5.3 速度和位置

```text
v_dot_n = C_bn f_b + g_n
          - (2 omega_ie_n + omega_en_n) x v_n
```

`g_n=[0,0,g]` 使用随纬度和高度变化的正常重力。正常重力已经包含常规离心
效应，不能再重复加入离心项。

```text
latitude_dot  = v_N/(Rm+h)
longitude_dot = v_E/((Rn+h)cos(latitude))
height_dot    = -v_D
```

极区 `cos(latitude)` 接近零时当前形式会拒绝计算，极区部署应使用 ECEF。

## 6. 双子样、标定与不同等级 IMU

先应用已知线性标定：

```text
dtheta_b = M_g * dtheta_reported
dvel_b   = M_a * dvel_reported
```

`M_g/M_a` 可以组合已标定的比例因子、非正交、交轴耦合和 sensor-to-body
安装旋转。之后减在线估计 bias，再进行圆锥、划桨和旋转补偿。

高精惯导与 MEMS 的适配不是只换名称：

- 算法结构相同；
- 噪声密度、bias 驱动和相关时间不同；
- 标定矩阵、温度模型、量程和饱和阈值不同；
- 高精器件更依赖完整地球模型，MEMS 更易受温漂和振动整流影响。

`mems_default()` 和 `navigation_grade_default()` 只是启动配置，不能替代 Allan
分析、六位置标定、转台标定和温箱标定。

## 7. 误差状态传播

主要连续时间块：

```text
delta_p_dot     = delta_v
delta_v_dot     = -C_bn [f_b x] delta_theta
                  - C_bn delta_b_a
                  - [2omega_ie+omega_en x] delta_v
delta_theta_dot = -[omega_ib_b x] delta_theta - delta_b_g
delta_b_a_dot   = -(1/tau_a)delta_b_a + w_ba
delta_b_g_dot   = -(1/tau_g)delta_b_g + w_bg
```

噪声顺序：

```text
w = [acc white, gyro white, acc bias drive, gyro bias drive]
```

噪声密度使用连续时间 SI 单位，`Qd ~= G Qc G^T dt`。状态转移使用：

```text
Phi ~= I + Fdt + 0.5 F^2 dt^2
```

适用于当前 100 Hz 导航周期。低频或强动态应使用 Van Loan 离散化或更小步长。

## 8. GNSS 杆臂、异常拒绝和更新

GNSS 测量天线位置，状态位置对应 IMU：

```text
lever_n = C_bn * lever_b
r = position_gnss_n - position_imu_n - lever_n
```

右乘姿态误差下：

```text
H_position = I
H_attitude = -C_bn [lever_b x]
```

创新统计量：

```text
NIS = r^T S^-1 r
S   = HPH^T + R
```

默认三维门限约 `16.27`。超过门限的 GNSS 不注入状态，并计入 rejected。

协方差采用 Joseph 形式：

```text
P = (I-KH)P(I-KH)^T + KRK^T
```

它比 `(I-KH)P` 更容易维持浮点对称性和半正定性。

## 9. 注入与 reset

```text
position <- position + delta_p_n
v_n      <- v_n + delta_v_n
q_bn     <- q_bn ⊗ Exp(delta_theta_b)
b_a      <- b_a + delta_b_a
b_g      <- b_g + delta_b_g
```

误差均值归零后，姿态 reset 使用：

```text
J_reset_theta ~= I - 0.5[delta_theta x]
P <- J_reset P J_reset^T
```

跳过 reset Jacobian 会使协方差仍停留在旧切空间。

## 10. 主流程与真实时间戳

```text
静止窗口调平 + gyrocompass/外部 yaw，或显式 truth 评估模式
 -> 用 GNSS 天线位置和杆臂初始化 IMU 位置
 -> 单子样跨越对准窗口边界（若需要）
 -> 恢复双子样主传播
 -> 两子样传播名义状态和 P
 -> 处理已到达且时间差在门限内的 GNSS
 -> 保存状态、标准差和更新统计
 -> 与 truth 插值对齐并输出误差
```

必须按增量区间而非仅按结束时间挑首帧，否则初始化同时间戳的一段会重复积分。

运行：

```powershell
.\.venv\Scripts\python.exe python\examples\run_dataset1_eskf.py `
  --initialization gyrocompass `
  --alignment-duration-s 30 `
  --duration-s 60 `
  --imu-profile navigation-grade `
  --output-dir results\dataset1_eskf
```

输出：`eskf_solution.csv`、`summary.json`、`position_error.png`、
`velocity_error.png`。

## 11. 可以失败的验证

测试覆盖：

- WGS-84 曲率半径和重力量级；
- 地球固定静止状态的姿态、速度保持；
- 连续传播后的协方差对称和半正定；
- GNSS 正常更新、杆臂零残差和 NIS 跳点拒绝；
- 已知 IMU 标定矩阵在机械编排前生效；
- IMU 不连续和 GNSS 时间错位会报错；
- dataset1 短回放生成有限、可审计输出。

```powershell
.\.venv\Scripts\python.exe -m pytest tests\python -q
```

## 12. 当前真实数据证据与边界

dataset1 前 60 秒、30 秒静止 gyrocompass 初始化、当前杆臂和
navigation-grade 启动参数：

```text
导航历元：6000
接受 GNSS：59
拒绝 GNSS：0
三维位置 RMS：约 0.0174 m
姿态角误差幅值 RMS：约 0.793 deg
```

该模式不使用 truth 姿态和速度，但位置及评估起点仍由 dataset1 的 GNSS/truth
时间范围定义，杆臂来自该数据集诊断。数字只证明当前实现与这套数据短时一致，
不能作为跨设备盲测精度宣传。`--initialization truth` 仅保留用于算法回归上限对照。

## 13. 工业部署仍缺什么

1. 在线静止检测、粗对准、精对准和失准恢复；
2. GNSS 速度/Doppler、双天线航向；
3. 延迟状态更新、硬件时间同步和时延标定；
4. 21 状态比例因子在线估计或完整离线标定加载；
5. 温度 bias、振动整流、饱和和 clipping 检测；
6. GNSS 状态质量、城市峡谷多路径和连续异常管理；
7. ECEF 极区机械编排和更完整的位置相关误差项；
8. C++ 实时实现、确定性内存、性能基准和长时间 soak test；
9. 多设备、多路况、多温度和重复上电的独立验证。

## 14. 面试问题、常见错误和自检

常见面试问题：

1. 为什么不用加性 4 维四元数误差？
2. 杆臂为什么让位置观测对姿态误差敏感？
3. `Qd = Qc dt` 的 `Qc` 为什么是噪声密度平方？
4. Joseph 更新和 reset Jacobian 为什么影响长期一致性？
5. 高精 IMU 和 MEMS 为什么不能只换一个噪声标准差？

常见错误：

- 混用 rate 和 increment；
- 初始化后重复使用同时间戳前一段 IMU；
- 混用 `C_bn` 与 `C_nb`；
- 忘记地球自转或重复加入离心项；
- 忽略 GNSS 杆臂、时间错位和跳点；
- 注入姿态后不做 covariance reset；
- 用默认 profile 替代真实设备标定。

自检：

1. 能否推导地球固定静止时左右姿态增量为什么抵消？
2. 为什么 bias 必须乘 `dt` 后才能从增量中扣除？
3. 杆臂为零时，`H_attitude` 为什么为零？
4. GNSS 频率降低十倍时，协方差和 bias 可观测性如何变化？
5. 哪些缺失功能首先限制 MEMS，哪些首先限制高精惯导？
