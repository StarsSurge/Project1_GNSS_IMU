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

## 13. 多窗口交叉标定与时钟漂移判断

### 13.1 为什么不能只用一个窗口

单窗口极小值可能同时受到运动激励、GNSS 跳点、初始状态和杆臂误差影响。跨窗口标定
把全程分成多个互不重叠的运动片段，每个窗口独立恢复初始状态并扫描时间偏差。真正稳定
的硬件常量偏差应在不同时间和运动下重复出现；某个窗口区间很宽，表示该窗口提供的信息
不足，而不是算法“精度变差”这么简单。

### 13.2 剖面 NIS 区间怎样得到

单窗口首先把均值评分恢复为总评分：

```text
J(tau_i) = measurement_count * robust_mean_NIS(tau_i)
```

在网格最优点左右各取一个候选，用局部二次曲线细化网格以下的极小值。然后采用单参数
95% 卡方增量 `3.841`，求解：

```text
J(tau) <= J_min + 3.841
```

与阈值的交点通过相邻网格线性插值得到上下界。若扫描到边界仍未穿过阈值，对应一侧
必须标记为无界，此时不能生成标准不确定度，也不允许该窗口进入加权时钟模型。

因为本仓库使用截断 NIS 且每个候选都会顺序更新 ESKF，这个区间是**近似 profile-NIS
诊断区间**，不是经过覆盖率证明的严格置信区间。双侧有界时，用区间宽度近似标准不确定度：

```text
sigma_tau approximately (tau_upper - tau_lower) / (2 * 1.96)
```

### 13.3 常量模型与线性漂移模型

以各窗口中心时间 `t_i`、细化偏差 `tau_i` 和标准不确定度 `sigma_i` 为输入，权重为：

```text
w_i = 1 / sigma_i^2
```

比较两个模型：

```text
constant: tau(t) = tau_0
linear:   tau(t) = tau_ref + drift * (t - t_ref)
```

`drift` 单位为 `s/s`；乘 `10^6` 后为 `ppm`。例如 `1 ppm` 表示两个时钟每经过
`1000 s` 会累计约 `1 ms` 的相对时间误差。

模型复杂度用 BIC 惩罚：

```text
BIC = n * log(weighted_RSS / n) + k * log(n)
```

其中常量模型 `k=1`，线性模型 `k=2`。本仓库只有同时满足以下条件才选择漂移模型：

1. `BIC_constant - BIC_linear >= 6`；
2. 漂移近似 95% 区间不跨零。

证据冲突时输出 `inconclusive`，而不是强制选择。该规则避免轻微拟合改善被误报成真实
时钟频偏。

### 13.4 代码和运行方式

- `estimate_time_offset_profile_interval()`：局部二次细化、阈值交点和边界检查；
- `compare_clock_offset_models()`：加权最小二乘、漂移区间、BIC 与模型判定；
- `cross_validate_dataset1_gnss_time_offset.py`：多窗口调度、汇总和可视化。

```powershell
.\.venv\Scripts\python.exe python\examples\cross_validate_dataset1_gnss_time_offset.py `
  --window-start-offsets-s 70 650 1230 1810 2390 2970 `
  --window-duration-s 30 `
  --candidate-min-s -0.05 --candidate-max-s 0.05 --candidate-step-s 0.005
```

输出包括每个窗口自己的评分文件，以及汇总的 `window_estimates.csv`、
`clock_model_comparison.png` 和 `summary.json`。至少需要四个双侧有界窗口；否则程序拒绝
模型比较并要求扩大扫描范围、延长窗口或更换运动片段。

### 13.5 仍然没有解决的问题

线性模型只能描述一阶频偏，不能描述时钟跳变、温度相关漂移、GNSS 驱动缓存抖动或
时间系统转换错误。窗口之间也不一定统计独立。部署前仍需硬件 PPS/触发沿审计、跨温度
重复实验和独立日志验证，不能用软件拟合替代底层时间同步设计。

### 13.6 当前 Dataset1 实验结果

在全程分散选取 `6` 个窗口，每个窗口 `30 s`，候选范围 `[-50, 50] ms`、步长
`5 ms`。二次细化后的窗口偏差范围约为 `-2.22 ms` 到 `+5.75 ms`，所有窗口都得到
双侧有界 profile-NIS 区间。

```text
weighted constant offset: 3.53 ms
approximate 95% interval: [0.42, 6.65] ms
linear drift estimate: 1.28 ppm
approximate drift interval: [-1.67, 4.23] ppm
BIC_constant - BIC_linear: 0.20
selected model: constant
```

漂移区间跨零且 BIC 改善远小于 `6`，因此这组数据不支持“存在稳定线性时钟漂移”的
结论。`3.53 ms` 也只是在 truth 辅助初始化、当前杆臂和噪声模型下得到的 Dataset1
离线估计，不能直接作为另一台设备的固定配置。

## 14. 初始化、杆臂和噪声参数敏感性

### 14.1 为什么做敏感性审计

时间偏差不是唯一能产生位置创新相位差的参数。车辆转弯时，错误杆臂会产生随角运动变化
的位置误差；错误初始姿态和速度会改变短时机械编排；过程噪声则改变创新协方差和各窗口
权重。若这些参数稍微变化，标定偏差就大幅移动，那么时间偏差还不能独立解释数据。

敏感性审计比较同一组窗口在不同假设下的加权常量偏差：

```text
shift_scenario = offset_scenario - offset_truth_baseline
```

当前示例把 `|shift| > 5 ms` 标记为显著敏感。该阈值是本实验的工程判据，不是普适
统计阈值。

### 14.2 operational 初始化如何去除窗口 truth

后续运动窗口不能重新做静态对准。实现从数据集开头的静止段开始：

1. GNSS 初始化位置；
2. 静止 IMU 完成水平和 navigation-grade gyrocompass 航向；
3. 从起点连续进行 GNSS/IMU ESKF；
4. 到每个窗口起点保存完整名义状态、bias 和协方差；
5. 用保存状态替代该窗口的 truth 位置、速度和姿态。

该链路不读取窗口时刻 truth 状态，但前序融合暂时假定 GNSS offset 为零，因此与时间
标定仍有一定循环依赖。完整工程方案应迭代“初始 offset -> 连续导航 -> 重新标定”，直至
变化收敛。

### 14.3 杆臂为何会伪装成时间偏差

天线位置为：

```text
p_antenna_n = p_imu_n + C_bn * lever_b
```

对时间求导：

```text
v_antenna_n = v_imu_n + C_bn * (omega_nb_b cross lever_b)
```

时间错位的一阶位置效应约为 `v_antenna_n * delta_t`。因此杆臂错误和时间偏差都会在
转弯时产生与角速度相关的残差，两者天然耦合。先可靠测量杆臂、再标定时间，通常比同时
无约束搜索更稳定。

### 14.4 网格分辨率不能被二次拟合绕过

局部二次曲线可以细化最优点，但不能证明微秒级置信度。实现规定 profile 区间半宽不得
小于局部候选网格半步长，并输出：

```text
grid_half_step_s
resolution_limited
```

例如 `10 ms` 网格最多只能把区间收紧到至少 `±5 ms`。若需要更窄区间，应在极小值
附近重新运行细网格，而不是相信尖锐的三点抛物线。

### 14.5 Dataset1 早期四窗口结果

使用 `+70、+190、+250、+310 s` 四个窗口，每窗 `10 s`，候选步长 `10 ms`：

```text
truth-assisted baseline:       -1.050 ms
continuous operational init:   -1.680 ms   shift -0.630 ms
zero lever arm:               +10.990 ms   shift +12.040 ms
lever +[0.1,-0.1,0.1] m:      -8.435 ms   shift -7.385 ms
IMU process noise x0.5:        -1.050 ms   shift approximately 0 ms
IMU process noise x2.0:        -1.051 ms   shift approximately 0 ms
```

当前结论是：去掉窗口 truth 初始化造成的移动小于 `1 ms`；过程噪声缩放影响很小；错误
杆臂会造成超过 `5 ms` 的明显时间偏差移动。因此下一轮高精度时间标定前，应优先确认
杆臂定义、方向、参考点和独立标定不确定度。

### 14.6 运行方式

```powershell
.\.venv\Scripts\python.exe python\examples\audit_dataset1_time_offset_sensitivity.py
```

输出 `sensitivity_summary.csv`、`sensitivity_shifts.png`、`summary.json`，并在
`scenarios/` 下保留每个场景的完整窗口评分，便于审计失败或异常窗口。

## 15. 杆臂与时间偏差联合标定

### 15.1 最小一阶模型

当 RTK 表示天线位置、truth 表示 IMU 参考点时，在 GNSS 报告时刻构造：

```text
residual_n = p_GNSS_n - p_IMU_truth_n
```

对小常量时间偏差做一阶展开：

```text
residual_n approximately C_bn * lever_b + velocity_n * time_offset
```

每个历元提供三行观测，未知量为：

```text
parameter = [lever_forward, lever_right, lever_down, time_offset]^T
unit      = [m, m, m, s]
```

实现使用 RTK NED 标准差白化残差，并按三维历元 Mahalanobis 范数做 Huber IRLS，避免
单个 RTK 跳点的三个分量被分别处理。输出后验 `4x4` 协方差、相关矩阵、设计矩阵条件数
和降权历元数。

### 15.2 可观性与耦合

若姿态和速度恒定，`velocity_n` 可能落在 `C_bn` 三个杆臂列的张成空间中，此时设计矩阵
秩小于四。算法必须拒绝，而不能返回任意一组杆臂和时间。

Dataset1 六个独立 30 秒窗口的自由联合结果显示：

```text
time offset range:       +0.04 ms to +2.98 ms
forward lever range:      0.103 m to 0.134 m
corr(forward, time):     -0.95 to -0.999
```

这说明单个窗口中“增加前向杆臂、减少时间偏差”几乎能产生相同残差。形式上矩阵可逆，
并不等于参数在工程上可独立辨识。

### 15.3 三种约束模式

1. `joint-unconstrained`：杆臂和时间都由数据估计，用于可观性诊断；
2. `joint-with-lever-prior`：机械杆臂及标准差作为 MAP 伪观测；
3. `fixed-independent-lever`：固定独立测量杆臂，只估时间。

MAP 先验形式为：

```text
[I_3, 0] * [lever, time]^T = measured_lever
covariance = diag(sigma_lever^2)
```

但大量被错误视为独立的毫米级 RTK 观测可能压倒 `1 mm` 先验。若 truth 误差和时间相关性
没有建模，不能通过无限缩小先验标准差来制造可信结果。真正独立且可靠的机械测量更适合
使用固定杆臂模式，并另做机械测量误差敏感性。

### 15.4 Dataset1 结果

全数据结果：

```text
unconstrained lever: [0.1340, -0.3010, -0.1839] m
unconstrained time:  +0.303 ms
corr(forward,time):  -0.863
fixed lever:         [0.14723, -0.29822, -0.18079] m
fixed-lever time:    -0.886 ms
```

固定杆臂后，六窗口时间范围收敛到 `-1.13` 至 `-0.34 ms`，加权常量约 `-0.767 ms`。
线性趋势约 `0.191 ppm`，但 BIC 改善仅 `2.70`，所以常量与漂移仍判为证据不足。

该结果与 ESKF profile 扫描不完全一致，因为两者目标函数不同：本节直接拟合 truth 位置
残差的一阶模型；ESKF profile 会顺序更新位置、速度、姿态和 bias，并让协方差与门限参与
评分。这种差异必须保留并解释，不能挑一个更好看的数字作为最终时间配置。

### 15.5 运行方式

自由联合标定：

```powershell
.\.venv\Scripts\python.exe python\examples\calibrate_dataset1_lever_time.py
```

固定独立机械杆臂：

```powershell
.\.venv\Scripts\python.exe python\examples\calibrate_dataset1_lever_time.py `
  --fixed-lever-b-m 0.14722696 -0.29821683 -0.18079014
```

输出包括 `lever_time_residuals.csv`、`lever_time_residuals.png`、
`parameter_correlation.png`（联合模式）和 `summary.json`。

### 15.6 天线旋转速度项

天线速度不是 IMU 参考点速度。杆臂固定在 body 中，因此：

```text
v_antenna_n = v_imu_n + C_bn * (omega_nb_b cross lever_b)
```

`omega_nb_b` 是 body 相对 NED 的角速度，在 body 中表达，单位 `rad/s`。Dataset1 实现
先在 IMU 增量区间中点计算：

```text
omega_ib_b = dtheta / dt
```

再插值到 GNSS 时刻，并扣除地球率和运输率：

```text
omega_nb_b = omega_ib_b - C_nb * (omega_ie_n + omega_en_n)
```

固定杆臂模式直接用 `v_antenna_n` 作为时间 Jacobian。联合模式的预测为：

```text
r_hat = C_bn * lever_b
      + [v_imu_n + C_bn(omega_nb_b cross lever_b)] * time_offset
```

该式对杆臂和时间是双线性的，所以实现采用 Gauss-Newton，并使用解析 Jacobian：

```text
d(r_hat)/d(lever) = C_bn + time_offset * C_bn * skew(omega_nb_b)
d(r_hat)/d(time)  = v_antenna_n
```

叉乘方向测试规定：`omega=[0,0,+1] rad/s`、`lever=[2,0,0] m` 时，旋转速度必须为
`[0,+2,0] m/s`。这个测试能直接暴露 `lever cross omega` 与 `omega cross lever` 写反。

Dataset1 全数据中，旋转杆臂速度 RMS 约 `0.022 m/s`，峰值约 `0.141 m/s`。加入该项后：

```text
joint time:       0.302790 ms -> 0.302603 ms
fixed-lever time: -0.886343 ms -> -0.886262 ms
```

变化不到 `0.0002 ms`。原因是约 `1 ms` 时间偏差乘 `0.022 m/s`，典型位置贡献只有
`0.022 mm`。因此该项物理上必须保留，但不是当前 ESKF profile 与 truth 残差模型差异的
主要来源。

可使用以下参数做 A/B：

```powershell
python python\examples\calibrate_dataset1_lever_time.py `
  --fixed-lever-b-m 0.14722696 -0.29821683 -0.18079014 `
  --omit-rotational-velocity
```

### 15.7 剩余模型边界

IMU 角速度尚未使用独立标定的陀螺 bias；不过该误差乘短杆臂后通常很小。更重要的是，
truth 本身的空间误差、姿态误差、时间误差和时间相关性没有进入协方差。当前结果适合暴露
耦合和比较模型，不足以替代全站仪杆臂测量、硬件 PPS 审计或跨设备标定。

## 16. 残差相关性与目标函数路径依赖

### 16.1 为什么 3362 个历元不等于 3362 个独立样本

位置 truth、RTK 解算误差、多路径和车辆运动都具有时间相关性。对每个 NED 残差分量计算：

```text
rho(k) = cov(r_i, r_{i+k}) / var(r)
tau_int = 1 + 2 * sum positive rho(k)
N_effective = N / tau_int
```

求和采用初始正序列，在第一个非正相关滞后停止。这不是完整的随机过程辨识，但能阻止
协方差按错误的 `1/sqrt(N)` 速度无限缩小。

固定杆臂全数据结果：

```text
lag-1 correlation N/E/D:       0.713 / 0.693 / 0.511
integrated correlation time:  25.94 / 24.94 / 2.98 epochs
effective samples N/E/D:      129.6 / 134.8 / 1126.6
raw epoch count:              3362
conservative std inflation:   5.09 times
```

固定杆臂时间的形式标准差为 `0.017 ms`，按最小有效样本数保守膨胀后约 `0.089 ms`。
相应近似区间从 `[-0.920,-0.852] ms` 扩为 `[-1.060,-0.712] ms`。即使膨胀后仍未
覆盖全部系统误差，因此它仍是诊断区间，不是设备认证指标。

### 16.2 冻结轨迹与 ESKF 顺序更新有什么不同

冻结轨迹对每个候选只做：

```text
r_k(tau) = GNSS_k - truth_antenna(t_reported + tau)
```

truth 轨迹不会被 GNSS 修改，因而只测量时间对齐。ESKF profile 则会在每条 GNSS 后修正
位置、速度、姿态和 bias；门限决定是否更新，更新又改变下一条创新。因此它的评分不仅是
时间似然，还包含完整滤波策略的路径依赖。

同一 `+70 s`、30 秒窗口、同一杆臂和同一组 29 条 GNSS：

```text
frozen truth trajectory best:       -0.698 ms
sequential ESKF, default gate:      -2.218 ms
sequential ESKF, gate disabled:     -0.215 ms
default ESKF accepted/rejected:      28 / 1
```

关闭门限后最优值重新落入冻结 profile 的网格受限区间 `[-3.198,+1.802] ms`。这说明只拒绝
一条 GNSS 也会通过后续状态与协方差反馈把最优点移动约 `2 ms`。因此不能把“最小顺序
ESKF NIS”直接解释为纯硬件时间偏差。

### 16.3 运行方式

残差相关性由联合/固定杆臂脚本自动输出：

```text
residual_autocorrelation.csv
residual_autocorrelation.png
```

目标函数对照：

```powershell
.\.venv\Scripts\python.exe python\examples\compare_dataset1_time_offset_objectives.py
```

输出冻结、默认 ESKF 和关闭门限 ESKF 三条候选曲线。部署标定应优先采用与状态更新解耦的
冻结/批处理目标，ESKF 顺序 profile 更适合评估某个时间配置对实际滤波策略的影响。

## 17. 移动块 Bootstrap 与独立导航验证

### 17.1 为什么不能逐历元 bootstrap

若逐个随机抽取 GNSS 历元，会破坏连续多路径、truth 平滑误差和车辆运动造成的时间相关
结构，区间仍会过窄。循环移动块 bootstrap 每次抽取连续长度为 `L` 的历元块，超过末尾
时循环回到开头，直到重新组成与原窗口相同长度的样本。

块长取冻结最优残差三轴最大积分相关时间：

```text
L = ceil(max(tau_int_N, tau_int_E, tau_int_D))
```

短窗口中若 `L` 超过样本数一半，则限制为一半，防止每次重采样几乎只是整窗循环平移。
每个 bootstrap 样本都重新计算完整候选目标、选择网格最优点并做局部二次细化。输出还
记录最优点落在候选边界的比例；边界命中率高意味着必须扩大扫描范围。

### 17.2 Dataset1 标定窗口结果

对 `+70 s` 开始的 30 秒窗口，29 个 GNSS 历元：

```text
frozen point estimate:       -0.698 ms
moving-block length:          5 GNSS epochs
bootstrap replicates:         2000
95% bootstrap interval:      [-3.198, +1.802] ms
boundary hit fraction:        0
minimum effective samples:    5.96 / 29
```

区间仍由 `5 ms` 候选网格分辨率限制，并且跨过零。因此 `-0.698 ms` 只能作为待验证
候选，不能直接成为默认配置。

### 17.3 标定与验证必须时间隔离

标定窗口从约 `456370 s` 开始。独立验证使用它之前的 60 秒导航片段，先冻结偏差，再运行
两次完全相同的 ESKF：一次 `0 ms`，一次 `-0.698 ms`。验证指标不再参与参数选择。

```text
3D position RMS:  0.017395136 -> 0.017395637 m
attitude RMS:      0.794350135 -> 0.794315184 deg
GNSS accept/reject: 59/0 -> 59/0
```

位置和三轴速度均没有改善；姿态变化约 `3.5e-5 deg`，不具备实际意义。结合 bootstrap
区间跨零，当前工程建议是保持：

```text
gnss_time_offset_s = 0.0
```

这不是证明真实偏差严格为零，而是说明现有数据没有足够证据支持部署非零补偿。

### 17.4 运行方式

冻结 profile、bootstrap 和目标函数对照：

```powershell
.\.venv\Scripts\python.exe python\examples\compare_dataset1_time_offset_objectives.py
```

固定参数进行持出段验证：

```powershell
.\.venv\Scripts\python.exe python\examples\validate_dataset1_calibrated_time_offset.py
```

验证脚本读取标定摘要，但不会根据持出段重新搜索参数。只有跨零区间缩窄、多个独立窗口
结果一致且持出导航指标稳定改善时，才应修改主流程默认时间配置。
