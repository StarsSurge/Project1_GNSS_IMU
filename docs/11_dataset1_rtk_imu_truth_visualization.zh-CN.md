# Dataset1 RTK / IMU / Truth 可视化学习笔记

## 1. Motivating Question

拿到一组 GNSS/IMU 数据后，第一个问题不是“能不能直接跑滤波器”，而是：

> 这些传感器数据在同一个时间轴、同一个坐标系、同一组单位假设下，看起来是否物理合理？

对 GNSS/INS 来说，可视化不是装饰步骤。RTK 位置、IMU 增量和真值轨迹分别来自不同处理层：

- RTK 是低频绝对位置观测，常用于滤波量测更新。
- IMU 是高频运动增量，常用于惯导机械编排或预积分。
- truth 是参考导航状态，用来评估轨迹、速度、姿态和误差。

如果在这里把经纬高直接当平面坐标、把 IMU 增量当角速度、把 GPS 周内秒错位几十秒，后面的 Kalman filter 即使代码正确，也会表现得像模型错误。

## 2. Physical Intuition

可以把数据集想成三种“记录员”在描述同一辆车：

- RTK 记录员每秒说一次“车大概在地球上的哪个经纬度”。
- IMU 记录员每 0.005 秒说一次“刚才身体坐标系下转了多少、速度增量是多少”。
- truth 记录员高频给出“我认为车辆真实位置、速度、姿态是多少”。

为了比较他们，必须把“地球上的经纬高”投影到车辆附近的一张局部地图上。这个局部地图通常用 NED：

- North：向北为正，单位 m。
- East：向东为正，单位 m。
- Down：向下为正，单位 m。

这个直觉只在局部区域内成立。跨很大距离时，地球曲率和参考点选择会变得重要；本数据集的轨迹范围较小，用局部 NED 足够用于第一轮检查。

## 3. Minimal Scalar Example

先看最小的一维例子。假设一个 RTK 点和 truth 点在同一条北向直线上：

- truth 北向位置是 `100.0 m`
- RTK 北向位置是 `100.3 m`

那么 RTK 相对 truth 的北向误差是：

```text
error_N = rtk_N - truth_N = 0.3 m
```

这个误差有明确物理意义：RTK 点比真值偏北 0.3 m。

如果不先转到局部坐标，而直接用纬度相减：

```text
30.000000 deg - 29.999997 deg
```

这个数的单位是角度，不是米。它不能直接进入以米为单位的 GNSS 位置量测残差，也不能直接和 RTK 标称标准差 `std_N/std_E/std_D` 比较。

## 4. Definitions and Conventions

本仓库对 `data/dataset1` 采用以下约定：

RTK 文件 `GNSS-RTK.txt`：

```text
time_s, latitude_deg, longitude_deg, height_m, std_N_m, std_E_m, std_D_m
```

IMU 文件 `Leador-A15.txt`：

```text
time_s, dtheta_x_rad, dtheta_y_rad, dtheta_z_rad, dvel_x_mps, dvel_y_mps, dvel_z_mps
```

注意这里是增量：

- `dtheta` 是采样间隔内的角增量，单位 rad。
- `dvel` 是采样间隔内的速度增量，单位 m/s。
- 若要画近似角速度或比力，需要除以相邻时间差 `dt`。

真值文件 `truth.nav`：

```text
gps_week, time_s, latitude_deg, longitude_deg, height_m,
velocity_N_mps, velocity_E_mps, velocity_D_mps,
roll_deg, pitch_deg, yaw_deg
```

局部坐标使用以 truth 第一帧为原点的 NED：

```text
reference_llh = truth[0].latitude, truth[0].longitude, truth[0].height
```

## 5. Step-by-Step Derivation

### 5.1 为什么要经过 ECEF？

经纬高是曲面坐标，不适合直接做线性误差。常见做法是先把 WGS-84 geodetic 坐标转成地心地固坐标 ECEF：

```text
N(phi) = a / sqrt(1 - e^2 sin^2(phi))
x = (N(phi) + h) cos(phi) cos(lambda)
y = (N(phi) + h) cos(phi) sin(lambda)
z = (N(phi)(1 - e^2) + h) sin(phi)
```

其中：

- `phi` 是纬度 rad。
- `lambda` 是经度 rad。
- `h` 是椭球高 m。
- `a` 是 WGS-84 长半轴 m。
- `e^2` 是第一偏心率平方。
- `N(phi)` 是卯酉圈曲率半径 m。

单位检查：`N + h` 是 m，三角函数无量纲，所以 `x/y/z` 是 m。

### 5.2 为什么 ECEF 差值还要旋转到 NED？

ECEF 的 `x/y/z` 轴固定在地球上，不是车辆附近直观的北东下方向。对任意点，先计算它相对参考点的 ECEF 差：

```text
delta_ecef = ecef(point) - ecef(reference)
```

再用参考点纬度、经度构造旋转矩阵，把差值投到 NED：

```text
[N, E, D]^T = R_ecef_to_ned(reference_lat, reference_lon) * delta_ecef
```

这样得到的三个分量才可以和 GNSS/INS 中常用的位置误差状态、RTK 的 `std_N/std_E/std_D` 对齐。

### 5.3 为什么 IMU increment 要除以 dt？

如果 IMU 在一个采样间隔内测得角增量 `dtheta`，近似角速度是：

```text
omega = dtheta / dt
```

速度增量同理：

```text
specific_force_like = dvel / dt
```

这里的 `specific_force_like` 是由数据文件增量反推的等效量，用于可视化检查量级。真实惯导机械编排会进一步处理坐标系、地球自转、重力和姿态旋转，不能把这个图直接当作完整导航解算。

## 6. Scaling Behavior

采样间隔影响图的量级：

- 若运动相同，`dtheta` 大约和 `dt` 成正比。
- 若把采样率提高一倍，单个样本的 `dtheta` 通常约减半。
- 除以 `dt` 后的角速度量级应基本保持一致。

这就是为什么可视化同时画增量和 rate：增量图检查文件原始内容，rate 图检查物理量级。

RTK 的轨迹点通常远少于 IMU 点。比较 RTK 和 truth 时，本脚本把 truth 的 NED 轨迹插值到 RTK 时间戳上，再计算：

```text
residual_NED = rtk_NED - interpolated_truth_NED
```

这能暴露时间错位、坐标轴方向错误和高度符号错误。

## 7. Code Mapping

主要代码位置：

- `python/gnss_imu/dataset_visualization.py`
  - `load_dataset1`: 读取 RTK、IMU、truth。
  - `geodetic_to_ecef`: WGS-84 LLH 到 ECEF。
  - `geodetic_to_ned`: LLH 到局部 NED。
  - `increments_to_rates`: IMU 增量转等效 rate。
  - `interpolate_columns`: truth 插值到 RTK 时间。
- `python/examples/visualize_dataset1.py`
  - 生成轨迹、位置分量、RTK 标准差、RTK-truth 残差、IMU、truth 速度姿态图。

输出目录：

```text
results/dataset1_visualization/
```

## 8. Verification That Can Fail

测试文件 `tests/python/test_dataset_visualization.py` 覆盖了几个故障会暴露的问题：

- 向北移动约 10 m 时，NED 的 North 分量应约为 `+10 m`。
- 向东移动约 10 m 时，East 分量应约为 `+10 m`。
- 高度降低 5 m 时，Down 分量应约为 `+5 m`。
- IMU 增量除以不等间隔 `dt` 后，rate 应按后向时间间隔计算。
- 重复时间戳会被拒绝。
- 姿态为 `yaw = 90 deg` 时，body 前向轴应指向 East。
- 若残差由固定 body-frame 杆臂产生，最小二乘应能恢复这个杆臂。

如果 NED 旋转矩阵符号写错，或者 Down 方向按 Up 处理，这些测试会失败。

## 8.1 RTK 与 Truth 差距复核结论

第一版图中 RTK 和 truth 的直接残差看起来偏大：

```text
raw RMS N/E/D = 0.229 / 0.240 / 0.184 m
raw 3D RMS   = 0.379 m
```

这不是普通白噪声形态。North/East 残差会随着道路方向改变而跳变，说明它更像一个固定在车体上的空间偏移经过姿态旋转后投影到 NED，而不是 RTK 随机误差。

用模型

```text
rtk_position_n - truth_imu_position_n ~= C_n_b * lever_arm_b
```

做最小二乘，得到估计杆臂：

```text
lever_arm_b = [forward, right, down]
            = [0.147, -0.298, -0.181] m
```

补偿这个杆臂后：

```text
corrected RMS N/E/D = 0.007 / 0.007 / 0.019 m
corrected 3D RMS   = 0.021 m
```

这与 README 中“导航状态位置是 IMU 位置”的定义一致：truth 更可能表示 IMU 中心，而 RTK 文件表示 GNSS 天线位置。直接相减比较的是两个不同物理点，所以会出现约分米级、随航向变化的系统差。

注意：这里的杆臂是从数据反推的诊断值，不等价于正式标定结果。正式组合导航应从配置或标定文件读取 GNSS 天线相对 IMU 的杆臂，并在量测模型中显式使用。

## 9. Real-World Boundary

当前可视化是数据理解工具，不是完整性能验证：

- 只做了诊断性杆臂拟合，没有替代正式杆臂标定。
- 没有估计 IMU 安装角或时延。
- 没有对 RTK 异常点、周跳、固定/浮点状态做质量控制。
- 没有把 IMU 机械编排成独立 INS 轨迹。
- 没有验证 truth 的来源、坐标框架和时间同步质量。
- 没有处理跨 GPS 周或长时间大范围轨迹。

因此，图像只能说明数据量级、时间覆盖和坐标约定是否初步合理，不能作为系统部署性能结论。

## 10. Common Interview Questions

1. 为什么 GNSS/INS 中经纬高残差通常要转到局部 NED 或 ENU 后再进入滤波器？
2. RTK 文件里的 `std_N/std_E/std_D` 和滤波器量测噪声矩阵 `R` 有什么关系？
3. IMU 角增量 `dtheta` 和角速度 `omega` 有什么区别？采样率改变时它们分别如何变化？
4. 为什么比较 RTK 和 truth 前要先做时间对齐或插值？
5. 如果高度误差图整体符号反了，最可能是哪一个坐标约定出错？

## 11. Common Mistakes

- 把经纬度差直接当米使用。
- 把 `dtheta` 当成 `rad/s`。
- 忽略 truth 和 RTK 的起止时间不同。
- 把 NED 的 Down 当成 Up。
- 只看轨迹图“像不像”，不检查残差、采样间隔和单位量级。

## 12. Self-Check Questions

1. 不看公式，你能说明为什么 `omega = dtheta / dt` 吗？
2. 如果 RTK 和 truth 轨迹形状一致但残差随时间线性增大，可能是什么时间问题？
3. 为什么本脚本选择 truth 第一帧作为 NED 原点？换成 RTK 第一帧会影响哪些图？
4. 如果车辆向东行驶，ENU 和 NED 的第二个分量是否相同？第三个分量呢？
5. 哪些可视化结果只能说明“数据初步合理”，不能说明“滤波器性能好”？
