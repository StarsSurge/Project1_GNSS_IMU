# IMU Error Visual Explanations

本目录用可重复的合成实验解释 IMU 误差模型、统计辨识和增量补偿。
所有实验均为教学用途，不代表某一具体 IMU 的数据手册指标。

## Recommended Order

1. `imu_error_model.py`
   - 确定性误差：bias、比例因子、非正交、温度项
   - 随机误差：白噪声、相关 bias、bias 随机游走
   - 重叠/非重叠 Allan deviation、斜率和有效差分对数量
   - 从原始测量到 ESKF 传播量的完整处理链
2. `coning_error.py`
   - 三维有限转动不可交换
   - 两子样圆锥补偿
   - 相对细积分真值的定量误差
3. `sculling_rotation_error.py`
   - 划桨误差与旋转效应
   - body 系增量与 navigation 系积分的区别
   - 两子样补偿的适用范围

## Run

在仓库根目录执行：

```powershell
$env:MPLCONFIGDIR = "$PWD\.matplotlib-cache"
.\.venv\Scripts\python.exe visual_explanations\imu_error_model.py
.\.venv\Scripts\python.exe visual_explanations\coning_error.py
.\.venv\Scripts\python.exe visual_explanations\sculling_rotation_error.py
.\.venv\Scripts\python.exe -m pytest tests\python\test_imu_visualization_math.py
```

图片生成到 `visual_explanations/outputs/`：

- `imu_error_1_systematic.png`
- `imu_error_2_random.png`
- `imu_error_3_allan.png`
- `imu_error_4_correction_pipeline.png`
- `coning_1_mechanism.png`
- `coning_2_spatial.png`
- `coning_3_impact.png`
- `sculling_1_mechanism.png`
- `sculling_2_rotation.png`
- `sculling_3_impact.png`

共享数学函数位于 `imu_visualization_math.py`，测试覆盖四元数/DCM
一致性、圆锥与划桨交叉项、白噪声 Allan deviation 的 `-1/2` 斜率、
重叠/非重叠估计一致性以及时间戳间断检测。

## Interpretation Boundary

- 离线标定用于估计可重复的确定性参数。
- 真实日志优先使用重叠 Allan deviation，并同时检查时间戳、温度、静止性、
  差分对数量和置信区间；Allan 曲线不能单独证明噪声来源。
- ESKF 在线估计的是随时间变化的误差状态，不会“消除”单次白噪声。
- 圆锥、划桨和旋转补偿属于 strapdown 增量编排，不应与传感器 bias 校准混为一谈。
- 两子样公式建立在等间隔、周期内平滑变化等假设上；高动态强振动需要更高采样率或多子样算法。

## Maturity

- 图解脚本：**educational prototype**，用于解释公式和验证趋势。
- `gnss_imu.imu_allan` 与 `analyze_imu_allan.py`：**verified MVP**，
  已具备真实 CSV、单位、时间戳质量和失败边界接口。
- 当前阶段：**not deployment-ready**。尚未用仓库内可追溯的真实静态 IMU
  数据验证，也未实现自动静止检测、温度分段、饱和检测、异常值审计、EDF
  置信区间和噪声参数自动选段。
