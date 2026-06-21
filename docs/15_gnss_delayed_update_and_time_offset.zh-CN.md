# GNSS 延迟状态更新与常量时间偏差标定

## 1. 它在回答什么问题

IMU 以高频连续传播，GNSS 低频输出且可能晚到。真正的问题不是“收到 GNSS
时更新一次”这么简单，而是：**这条位置观测描述的是哪一时刻的载体状态？**

本仓库区分三个时间：

- `reported_time`：GNSS 数据包携带的时间戳，单位 `s`；
- `effective_time`：该位置观测在统一系统时标中真正对应的时刻，单位 `s`；
- `arrival_time`：软件收到并处理数据包的时刻，单位 `s`。

当前常量偏差约定为：

```text
effective_time = reported_time + offset
```

`offset > 0` 表示观测对应的物理状态比报告时间更晚。到达延迟只决定何时触发
回放，不应被直接加到观测模型里。

## 2. 先从直觉和一维例子理解

设小车以 `10 m/s` 匀速运动。GNSS 时间错了 `0.04 s`，即使位置本身无噪声，
与错误时刻状态比较也会产生：

```text
position error = speed * time error = 10 m/s * 0.04 s = 0.4 m
```

这解释了为什么厘米级位置源仍可能产生分米级创新。该类比只适用于一维匀速；
真实系统还含三维加速度、姿态、杆臂和地球模型，因此实现采用状态回放而不是只用
`v * dt` 修正位置。

## 3. 状态、坐标系和单位

本模块不增加 ESKF 状态维数，仍使用 15 维右误差状态：

```text
delta_x = [delta_p_n, delta_v_n, delta_theta_b,
           delta_b_a, delta_b_g]
dimension = 3 + 3 + 3 + 3 + 3 = 15
```

- 位置和速度误差在 NED 表达，单位分别为 `m`、`m/s`；
- 姿态误差为右乘小旋转，单位 `rad`；
- 加速度计、陀螺零偏单位为 `m/s^2`、`rad/s`；
- IMU 输入是区间增量 `dtheta [rad]`、`dvel [m/s]`，时间戳表示区间终点；
- GNSS 位置是天线相位中心位置，杆臂为 `antenna - IMU`，在机体系 FRD 表达。

## 4. 为什么必须回到观测时刻

GNSS 位置残差应在有效时刻 `t_g` 构造：

```text
r(t_g) = z_g(t_g) - h(x(t_g))
```

若错误地用当前时刻 `t_k`：

```text
r_wrong = z_g(t_g) - h(x(t_k))
```

一阶展开 `h(x(t_k))`，会多出近似为 `-v * (t_k - t_g)` 的确定性项。滤波器
可能把它误估成位置、速度、姿态或 IMU bias，导致协方差含义也被破坏。

固定滞后更新因此执行：

1. 保存历史导航状态、协方差和原始 IMU 增量；
2. 数据到达后恢复到 `t_g` 之前的状态；
3. 精确传播到 `t_g` 并执行 GNSS 更新；
4. 用原始 IMU 从 `t_g` 重新传播到当前时刻。

必须保存原始 IMU，而不能只保存旧状态轨迹，因为 GNSS 更新会修正 bias 和姿态；
这些修正会改变后续每一段机械编排和协方差传播。

## 5. GNSS 落在一个 IMU 区间内部时

设一条 IMU 增量覆盖 `[t_0, t_1]`，总时长 `Delta t`，GNSS 位于 `t_g`。
当前 MVP 在该小区间内采用角速度和比力分段常值假设：

```text
alpha = (t_g - t_0) / Delta t
dtheta_before = alpha * dtheta
dvel_before   = alpha * dvel
dtheta_after  = (1 - alpha) * dtheta
dvel_after    = (1 - alpha) * dvel
```

量纲检查：`alpha` 无量纲，所以分割后的角增量仍为 `rad`，速度增量仍为 `m/s`。
在分段常值模型下这是精确分割；对区间内剧烈变化只是一阶近似。采样周期减半时，
常规光滑运动下区间内模型误差会快速减小，但饱和、冲击和时间戳跳变不能靠提高回放
精度补救。

## 6. 常量时间偏差怎样标定

对每个候选 `tau_i`，都完整运行一次固定滞后 ESKF：

```text
t_effective = t_reported + tau_i
nu_i,k = z_k - h(x_k^-)
NIS_i,k = nu_i,k^T * S_i,k^(-1) * nu_i,k
```

其中 `nu [m]`，`S [m^2]`，因此 NIS 无量纲。候选评分使用截断均值：

```text
score(tau_i) = mean(min(NIS_i,k, 100))
best_offset = argmin score(tau_i)
```

截断用于降低少量 GNSS 跳点对扫描的支配作用，不意味着异常观测可以忽略。最终还应
检查中位数、接受/拒绝数、不同运动窗口的一致性和评分曲线是否存在清晰极小值。

### 可观性

静止时 `v = 0`，小时间偏差引起的位置变化近似为零，偏差不可观。匀速运动可以提供
信息，但时间偏差容易与初始位置误差相关；加减速、转弯和杆臂角运动会提供更强激励。
当前 API 至少检查峰值速度，不足时直接拒绝标定。工程应用还应检查速度变化、角速度、
轨迹方向多样性和 Fisher 信息或评分曲率。

## 7. 数量级与网格选择

时间误差引起的一阶位置误差量级为：

```text
|delta_p| approximately |v| * |delta_t|
```

例如 `20 m/s` 下 `5 ms` 对应约 `0.1 m`。候选步长应使相邻候选产生的位置差明显
小于 GNSS 噪声，但扫描范围必须覆盖硬件可能的时间误差。可先粗扫，再在极小值附近细扫；
不能根据单一窗口输出很多无依据的小数位。

## 8. 代码映射

- `python/gnss_imu/delayed_eskf.py`
  - `FixedLagGNSSFusion.process_imu()`：保存原始增量和边界快照；
  - `process_gnss()`：计算有效时刻、插入历史观测并触发回放；
  - `_replay_history()`：区间分割、更新和重新传播；
  - `calibrate_constant_gnss_time_offset()`：候选扫描与稳健 NIS 评分。
- `python/examples/run_dataset1_eskf.py`
  - `--gnss-update-mode delayed-replay` 启用精确回放；
  - `--gnss-time-offset-s` 加载已标定常量；
  - `--fixed-lag-s` 设置历史窗口。
- `python/examples/calibrate_dataset1_gnss_time_offset.py`
  - 独立离线标定入口，输出候选评分 CSV 和摘要 JSON。

## 9. 能暴露错误的验证

`tests/python/test_delayed_eskf.py` 包含：

1. GNSS 位于 `0.05--0.06 s` 区间中央，延迟到 `0.10 s` 才到达；自动回放结果必须
   与手工把 IMU 分成两个 `0.005 s` 增量的参考结果一致；
2. 超出固定滞后窗口的观测必须报错，不能悄悄更新当前状态；
3. 合成加速轨迹注入 `+0.04 s` 偏差，扫描必须找回该候选；
4. 静止数据必须以“运动激励不足”拒绝标定。

这些测试能抓到时间偏差符号、增量单位、区间边界、状态深复制和错误静止标定等问题。

## 10. 运行方式与预期输出

精确延迟更新：

```powershell
.\.venv\Scripts\python.exe python\examples\run_dataset1_eskf.py `
  --gnss-update-mode delayed-replay `
  --gnss-time-offset-s 0.0 `
  --fixed-lag-s 2.0 `
  --duration-s 60
```

离线扫描：

```powershell
.\.venv\Scripts\python.exe python\examples\calibrate_dataset1_gnss_time_offset.py `
  --start-offset-s 70 --duration-s 30 `
  --candidate-min-s -0.05 --candidate-max-s 0.05 --candidate-step-s 0.005
```

输出 `time_offset_scores.csv`、`time_offset_scores.png` 和 `summary.json`。先看评分曲线是否有稳定极小值，再把
选出的偏差传给主流程；不要只看 `best_offset_s` 一项。

## 11. 真实工程边界

当前属于离线标定 MVP，而非部署级时间同步系统：

- 只估计常量 offset，不估计时钟频偏、温漂和逐包延迟抖动；
- 区间分割假定增量内 rate 常值；
- 标定示例用 truth 辅助初始状态，以隔离时间误差；
- 未处理 GNSS 周跳状态、时间系统转换错误、IMU/GNSS 硬件触发沿定义；
- Python 列表回放适合验证，不满足硬实时确定性内存和时延要求；
- 结果必须跨窗口、跨运动类型并用独立数据验证。

部署优先级通常是：硬件共钟/时间脉冲 > 驱动层时间戳审计 > 离线常量标定 > 在线估计。
算法不能修复一个定义不清或会跳变的底层时钟。

## 12. 面试问题、常见错误与自检

常见问题：

1. 为什么不能把旧 GNSS 直接更新到当前状态？
2. 为什么回放必须重放原始 IMU，而不是只平移保存的轨迹？
3. 时间偏差何时可观？它与初始位置、杆臂误差如何耦合？
4. `effective = reported + offset` 的正号怎样用一维运动重新推导？
5. 为什么 NIS 可以比较不同方向、不同协方差的创新？

常见错误：把到达时间当有效时间；混用秒和毫秒；切分 rate 而输入其实是 increment；
回放名义状态却不回放协方差；更新后继续使用旧 bias 传播；只在静止数据上标定；用单一
窗口极小值宣称硬件精度。

自检：若忘记最终公式，能否从“位置每秒变化多少”推回时间误差的一阶效应？能否说明
转弯时天线杆臂为何增强时间偏差信息？能否设计一个会让偏差正负号写反立刻失败的测试？
