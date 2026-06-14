# IMU 误差建模与改正：图解学习指南

## 阅读方法：先问四个问题

学习每一种 IMU 误差时，先不要急着记公式。依次回答：

1. **它在测量值上表现成什么？** 是整体平移、按比例放大、轴间串扰，还是随时间随机变化？
2. **重复实验能否得到相同结果？** 能重复的通常适合标定；不能重复的需要统计模型。
3. **时间尺度变长后，它如何增长或衰减？** 这决定 Allan 曲线斜率以及滤波器状态模型。
4. **算法在哪一层处理它？** 传感器标定、IMU 前处理、strapdown 机械编排，还是 ESKF 状态估计？

全文采用同一条学习链：

```text
物理现象 → 简单一维例子 → 数学模型 → 图上特征 → 代码实现 → 工程限制
```

建议分三遍阅读：

- **第一遍建立直觉**：1.1、2.1、3.1、3.3～3.7、4.1～4.5、6.1、7.1；
- **第二遍理解实现**：2.2、3.8～3.9、4.6～4.10、5、8；
- **第三遍面向真实数据**：4.7～4.10、工程限制、自检问题和审计文档 09。

如果只记住“白噪声斜率是 -1/2”而不知道为什么，那么换成不同单位、不同
输入类型或不同滤波模型时就很容易用错。本章的目标是让结论可以被重新推导，
而不是背诵结论。

## 1. 问题定义

### 1.0 本文术语与符号约定

IMU 文献中最容易混淆的不是公式本身，而是同一个词在不同参考系中的含义。
本文采用以下约定：

| 概念 | 符号 | 含义 |
|---|---|---|
| 地球引力加速度 | \(\boldsymbol\gamma\) | 地球质量产生的真实引力场，不包含离心效应 |
| 有效重力/导航重力 | \(\mathbf g\) | 在随地球旋转的导航方程中，将引力与离心效应合并后的量 |
| 运动学加速度 | \(\mathbf a\) | 位置对时间的二阶导数；必须说明相对于哪个参考系 |
| 比力 | \(\mathbf f\) | 单位质量所受真实非引力合力，又称 proper acceleration |
| body 系 | \(b\) | 固连于 IMU/载体的坐标系 |
| navigation 系 | \(n\) | 本项目通常采用的局部 NED 导航坐标系 |
| inertial 系 | \(i\) | 理想惯性参考系 |
| Earth-fixed 系 | \(e\) | 随地球旋转的地固坐标系 |

上标表示“该向量用哪个坐标系的分量表达”。例如 \(\mathbf f^b\) 和
\(\mathbf f^n\) 是同一个物理向量，分别用 body 系和 navigation 系坐标表示：

\[
\mathbf f^n=\mathbf C_b^n\mathbf f^b
\]

参考系和表达坐标系是两个概念。例如“物体相对地固系静止”描述运动关系，
而“向量用 NED 分量表示”描述数值表达方式。后文若使用简化公式，会明确说明
忽略了哪些旋转坐标系项。

### 1.1 先建立直觉：IMU 像一把会变化的尺子

把真实角速度或比力想象成待测长度，把 IMU 想象成一把尺子：

- 零刻度不在零点：**bias**；
- 每一格并不等于标准的一格：**比例因子**；
- X 轴运动会漏到 Y 轴：**非正交/交轴耦合**；
- 尺子受温度影响伸缩：**温度误差**；
- 每次读数都轻微抖动：**测量白噪声**；
- 零刻度随时间缓慢漂移：**运行中 bias 随机过程**。

前四项描述“尺子本身怎样变形”，通常具有可重复规律；后两项描述每次测量
和每次运行中的不确定性。这个区别决定了不能用同一种方法处理所有误差。

IMU 给出的不是理想角速度和比力。一个适合算法原型的统一模型是

\[
\mathbf y_m =
\mathbf K(T)\mathbf y_{true}
+ \mathbf b(T)
+ \mathbf b_{run}(t)
+ \mathbf n(t)
\]

其中：

- \(\mathbf y_m\in\mathbb R^3\)：陀螺角速度或加速度计比力测量；
- \(\mathbf K(T)\in\mathbb R^{3\times3}\)：比例因子、非正交、轴间耦合及其温度项；
- \(\mathbf b(T)\in\mathbb R^3\)：可重复的零偏和温度零偏；
- \(\mathbf b_{run}(t)\)：上电后缓慢变化、需要在线估计的 bias；
- \(\mathbf n(t)\)：逐样本不可预测的测量噪声。

关键不是把所有误差都叫作 bias，而是区分它们应由标定、随机建模、
在线估计还是机械编排补偿处理。

### 1.2 加速度计为什么静止时不是零

#### 第一层：传感器物理定义

加速度计测量的是**比力**，也就是单位质量受到的真实非引力合力：

\[
\boxed{
\mathbf f=\frac{\mathbf F_{\text{non-gravitational}}}{m}
}
\]

真实非引力包括支撑力、推力、空气阻力等，但**不包括地球引力**。因此你的
理解是正确的：加速度计并不直接测量引力，也不直接测量有效重力。

#### 第二层：惯性系中的等价公式

在惯性系中，牛顿第二定律写为

\[
m\mathbf a^i
=m\boldsymbol\gamma^i
+\mathbf F_{\text{non-gravitational}}^i
\]

所以

\[
\boxed{
\mathbf f^i=\mathbf a^i-\boldsymbol\gamma^i
}
\]

这里减去的是地球**引力加速度**
\(\boldsymbol\gamma\)，而不是已经包含离心效应的有效重力
\(\mathbf g\)。

#### 第三层：为什么导航方程中又出现了重力 \(\mathbf g\)

地球表面的 NED 导航系不是惯性系，它随地球旋转，并且随载体在地球表面移动。
把惯性系动力学转换到导航系后，常写成

\[
\dot{\mathbf v}^{n}
=\mathbf C_b^n\mathbf f^b
+\mathbf g^n
-\left(
2\boldsymbol\omega_{ie}^n+\boldsymbol\omega_{en}^n
\right)\times\mathbf v^n
\]

其中：

- \(\boldsymbol\omega_{ie}^n\)：地球自转角速度；
- \(\boldsymbol\omega_{en}^n\)：导航系相对地固系的运输角速度；
- \(\mathbf g^n\)：导航方程使用的当地有效重力。

一种常见定义为

\[
\boxed{
\mathbf g
=\boldsymbol\gamma
-\boldsymbol\omega_{ie}\times
(\boldsymbol\omega_{ie}\times\mathbf r)
}
\]

该式中的所有向量必须先用同一个坐标系表达。若采用 NED，向下为正，则当地
\(\mathbf g^n\) 通常近似写成 \([0,0,g]^T\)；静止加速度计比力则近似为
\(\mathbf f^n=[0,0,-g]^T\)。换成 ENU 或改变传感器轴方向后，分量符号会变化，
但物理关系不变。

即有效重力由地球引力和地球自转离心效应合成。不同教材可能把离心项放在
\(\mathbf g\) 内，也可能在导航方程中单独保留；使用公式前必须检查定义，
不能只看符号名称。

因此，惯性系中的
\(\mathbf f=\mathbf a-\boldsymbol\gamma\)
与导航系速度方程中的
\(\mathbf C_b^n\mathbf f^b+\mathbf g^n\)
并不矛盾。前者是传感器物理定义，后者是转换到旋转导航坐标系后的动力学形式。

#### 静止和自由落体分别发生了什么

物体相对地面静止时，主要非引力是真实的桌面支撑力。支撑力使物体跟随旋转
地球运动，其单位质量效果在导航系中近似为

\[
\mathbf f^n\approx-\mathbf g^n
\]

所以加速度计输出模长约为 \(g\)。它测到的是**支撑力对应的比力**，只是该
支撑力在静止平衡条件下与当地有效重力大小近似相等、方向相反，不能说
“加速度计测到了重力”。

自由落体时，忽略空气阻力：

\[
\mathbf F_{\text{non-gravitational}}\approx0
\quad\Rightarrow\quad
\mathbf f\approx0
\]

尽管物体正在受引力加速，理想加速度计仍接近零。

这个区别很重要：若把静止时约 \(g\) 的输出误当作加速度计 bias，后续标定、
姿态和速度传播都会出错。

### 1.3 全文容易混淆的概念对照

| 容易混淆的概念 | 核心区别 |
|---|---|
| 引力 \(\boldsymbol\gamma\) / 有效重力 \(\mathbf g\) | 引力来自地球质量；有效重力是在旋转导航方程中将引力与离心效应组合后的量 |
| 标准重力 \(g_0\) / 当地有效重力 \(\|\mathbf g\|\) | \(g_0=9.80665\,\mathrm{m/s^2}\) 是单位换算约定；当地值随纬度和高度变化 |
| 运动学加速度 \(\mathbf a\) / 比力 \(\mathbf f\) | \(\mathbf a\) 描述运动状态变化；\(\mathbf f\) 是单位质量非引力合力，是加速度计直接响应的量 |
| 力 / 加速度 | 力单位为 N；除以质量后才得到 \(\mathrm{m/s^2}\)。比力名称中有“力”，量纲却是加速度 |
| 参考系 / 表达坐标系 | 前者说明运动相对于谁；后者说明向量分量沿哪些轴表示 |
| body 系 / 传感器敏感轴系 | 理想情况下可重合；真实安装中可能存在安装角，需要标定旋转矩阵 |
| rate / increment | rate 是每单位时间的量，如 `rad/s`；increment 是一个采样区间的积分，如 `rad` |
| IMU `delta-v` / 导航速度变化 | `delta-v` 通常是 body 系比力积分；加入姿态旋转、有效重力和旋转坐标系项后才得到导航速度变化 |
| 陀螺角增量 / 精确有限旋转 | 单个小角度时近似一致；多个子样组合时还需考虑旋转不交换和 coning 修正 |
| 固定 bias / 运行中 bias 状态 | 固定 bias 可通过标定重复估计；运行中残余 bias 随时间变化，需要在线估计 |
| 白噪声 / 随机游走 | 白噪声逐样本近似无记忆；随机游走是白噪声积分后的有记忆状态 |
| 噪声密度 / 离散样本标准差 | 前者含 \(\sqrt{\mathrm{Hz}}\) 或 \(\sqrt{s}\) 量纲；后者取决于采样周期 |
| 相关时间 \(\tau_c\) / Allan 聚类时间 \(\tau\) | \(\tau_c\) 是随机过程自身的记忆尺度；\(\tau\) 是分析者选择的平均窗口长度 |
| Allan variance / Allan deviation | 后者是前者的平方根；工程曲线通常画 deviation |

后文遇到这些术语时，应先确认当前讨论的是传感器物理层、坐标变换层、
离散采样层还是随机过程层。

许多 IMU 数据手册或驱动接口会把加速度计通道简称为 `acceleration`，
把比力增量简称为 `delta_velocity`。这些是工程命名习惯，不能据此改变物理
定义。读取协议时应确认输出究竟是原始比力、已补偿比力、比力增量，还是厂商
已经加入了姿态/重力处理后的量。

对应图片：
[`imu_error_1_systematic.png`](../visual_explanations/outputs/imu_error_1_systematic.png) 和
[`imu_error_4_correction_pipeline.png`](../visual_explanations/outputs/imu_error_4_correction_pipeline.png)。

## 2. 确定性误差与离线标定

### 2.1 为什么确定性误差可以标定

先看单轴模型：

\[
y_m=(1+s)y_{true}+b
\]

如果真实输入为零，反复测得约 \(0.02\)，可估计 \(b\approx0.02\)；如果给定
真实输入 \(1.00\)，去掉 bias 后测得 \(1.01\)，可估计比例因子
\(s\approx0.01\)。之后反解：

\[
\hat y_{true}=\frac{y_m-b}{1+s}
\]

“可标定”的根本原因不是误差恒定不变，而是它与输入、温度等可观测变量之间
存在可重复关系。三轴情况只是把标量 \(1+s\) 扩展成矩阵：对角元素描述各轴
比例因子，非对角元素描述一个轴泄漏到另一个轴。

将比例因子和非正交误差写为

\[
\mathbf K = \mathbf I + \mathbf S + \mathbf M
\]

其中 \(\mathbf S\) 通常取对角矩阵，\(\mathbf M\) 的非对角元素描述
轴间耦合。获得标定参数后，校正为

\[
\hat{\mathbf y}
= \mathbf K(T)^{-1}
[\mathbf y_m-\mathbf b(T)]
\]

实践中还要明确原始码值到物理单位的转换、轴顺序、右手系、符号以及
传感器到机体系的安装矩阵。矩阵次序不能随意交换。

典型标定方法：

- 加速度计六位置静态标定：利用静止时支撑比力模长约等于当地有效重力模长，
  并通过不同轴向约束估计参数；
- 陀螺转台标定：用已知角速度估计 bias、比例因子和轴间耦合；
- 多温点标定：拟合 \(\mathbf b(T)\) 和必要的 \(\mathbf K(T)\)；
- 陀螺 g-sensitivity：需要不同线加速度激励，不能靠纯静态数据完整辨识。

### 2.2 为什么要乘逆矩阵，而不是逐轴相减

若 Y 轴测量中混入了一部分 X 轴真实输入，仅减去 bias 无法消除串扰。例如：

\[
\begin{bmatrix}y_{mx}\\y_{my}\end{bmatrix}
=
\begin{bmatrix}1&0\\0.02&1\end{bmatrix}
\begin{bmatrix}y_x\\y_y\end{bmatrix}
\]

当真实输入是 \([1,0]^T\) 时，测量变成 \([1,0.02]^T\)。Y 轴的 `0.02`
不是 Y 轴 bias，而是 X 轴输入通过标定矩阵泄漏过来的。只有整体求逆才能把
耦合关系解开：

\[
\hat{\mathbf y}_{true}
=\mathbf K^{-1}(\mathbf y_m-\mathbf b)
\]

这也解释了为什么单位转换、轴重排、安装旋转和标定矩阵的顺序不能随意交换：
矩阵乘法一般不满足交换律。

### 2.3 确定性误差的具体分类

确定性误差是指：在给定温度、输入和安装条件下，误差具有可重复的函数关系，
能够通过实验辨识参数并逐样本补偿。

| 误差项 | 典型模型 | 陀螺 | 加速度计 | 主要辨识方法 | 处理方式 |
|---|---|---:|---:|---|---|
| 固定零偏 | \(\mathbf b_0\) | 是 | 是 | 静止均值、六位置、转台 | 离线减去 |
| 比例因子 | \(\mathbf S\mathbf y\) | 是 | 是 | 已知角速率、静止多位置比力 | 标定矩阵逆变换 |
| 非正交/交轴耦合 | \(\mathbf M\mathbf y\) | 是 | 是 | 多轴激励 | 标定矩阵逆变换 |
| 安装角误差 | \(\mathbf C_{bs}\) | 是 | 是 | 外部参考姿态、多姿态标定 | 坐标系旋转 |
| 温度零偏 | \(\mathbf b(T)\) | 是 | 是 | 恒温箱、多温点静止实验 | 查表或多项式补偿 |
| 温度比例因子 | \(\mathbf K(T)\) | 是 | 是 | 多温点已知输入 | 温度相关矩阵补偿 |
| 非线性 | \(k_2y^2+k_3y^3\) | 是 | 是 | 多幅值输入 | 非线性标定 |
| g-sensitivity | \(\mathbf K_g\mathbf f\) | 主要是 | 通常不这样命名 | 转台配合已知比力激励 | 陀螺测量中减去 |
| 加速度计振动整流 | 与振动频率、幅值相关 | 间接 | 是 | 振动台实验 | 专用模型/机械隔振 |
| 时间延迟 | \(y(t-\Delta t)\) | 是 | 是 | 与外部传感器对时 | 时间同步与延迟补偿 |

`g-sensitivity` 这一名称容易误导。它表示陀螺输出对线性比力/加速度激励的
敏感性，指标常按标准重力 \(g_0\) 归一化为 `deg/h/g`。这里的字母 `g`
是工程单位尺度，不表示陀螺或加速度计直接测量了地球重力。

需要特别区分三种“零偏”：

1. **标定零偏**：实验中可重复辨识的常值或温度函数，属于确定性误差。
2. **上电零偏**：每次启动可能不同；启动后可静止估计，或作为滤波初值。
3. **运行中 bias 漂移**：随时间随机变化，属于随机过程，应由 ESKF 在线估计。

圆锥、划桨和旋转效应不是传感器本体的随机噪声，而是有限采样和三维旋转
不可交换造成的**机械编排误差**。它们有确定公式，但应单独放在 strapdown
增量算法中处理。

## 3. 随机误差模型

### 3.1 为什么随机误差仍然可以建模

“随机”不是“毫无规律”。我们不能预测下一次白噪声究竟是正还是负，但可以
描述它的均值、方差、频谱以及相邻时刻的相关性。

区分随机过程时，最重要的是问：

- 当前误差是否记得上一时刻？
- 经过更长时间平均后，它会被抵消，保持不变，还是继续增长？

白噪声没有记忆，平均会变小；随机游走会累积历史扰动，时间越长越不确定；
一阶 Gauss-Markov 既有随机驱动，又有回到零附近的趋势。

### 3.2 随机误差总分类

下表直接回答“哪些是白噪声，哪些是一阶 Gauss-Markov”：

| 现象/名称 | 数学模型 | 是否白噪声 | 是否一阶 Gauss-Markov | 是否作为 ESKF 状态 | Allan 曲线典型特征 |
|---|---|---:|---:|---:|---|
| 陀螺测量噪声 \(n_g\) | \(\omega_m=\omega+b_g+n_g\) | 是 | 否 | 否，作为过程噪声输入 | rate Allan 斜率 \(-1/2\)，积分后称 ARW |
| 加速度计测量噪声 \(n_a\) | \(f_m=f+b_a+n_a\) | 是 | 否 | 否，作为过程噪声输入 | rate Allan 斜率 \(-1/2\)，积分后称 VRW |
| 陀螺 bias 随机游走 | \(\dot b_g=w_{bg}\) | 驱动 \(w_{bg}\) 是白噪声；\(b_g\) 不是 | 否 | 是 | rate Allan 斜率约 \(+1/2\) |
| 加速度计 bias 随机游走 | \(\dot b_a=w_{ba}\) | 驱动 \(w_{ba}\) 是白噪声；\(b_a\) 不是 | 否 | 是 | rate Allan 斜率约 \(+1/2\) |
| 相关 bias | \(\dot b=-b/\tau_c+w_b\) | 驱动 \(w_b\) 是白噪声；\(b\) 不是 | **是** | 是 | 在相关时间附近发生转折或形成宽缓区域 |
| Bias instability | 闪烁噪声，近似 \(1/f\) PSD | 否 | 否 | 不能被单个一阶 GM 精确表示 | 近似水平区，斜率 \(0\) |
| 量化噪声 | ADC/数字输出取整误差 | 通常近似，不严格是 | 否 | 通常否 | 小 \(\tau\) 区域可出现斜率 \(-1\) |
| 速率斜坡/老化 | \(b(t)=b_0+kt\) | 否 | 否 | 可扩展为斜坡状态 | 斜率约 \(+1\) |

最容易混淆的是：

- **ARW/VRW 不是新的噪声源名称**，而是角速度/比力白噪声积分到角度/速度后的结果；
- **bias 随机游走不是白噪声**，它是由白噪声驱动后积分得到的有记忆过程；
- **一阶 Gauss-Markov bias 也不是白噪声**，它是由白噪声驱动的有色、有限相关时间过程；
- **bias instability 通常对应闪烁噪声**，不能严格等同于“一阶 Gauss-Markov”；
- 同一台 IMU 可同时包含测量白噪声、相关 bias、随机游走和确定性温漂。

### 3.3 时间尺度总览：不是按固定秒数分段，而是谁的量级最大

“短时白噪声主导、长时 bias 主导”只是方向性描述。某一时间尺度上真正的
主导项，是该项在此时间尺度产生的误差标准差最大：

\[
\text{dominant component at }\tau
=\arg\max_i \sigma_i(\tau)
\]

因此不存在适用于所有 IMU 的固定分界，例如不能规定“10 秒以前一定是白噪声，
10 秒以后一定是 bias”。分界取决于传感器等级、带宽、温度环境、振动和标定
质量。

#### 3.3.1 在原始 rate 数据和 Allan 图上谁主导

用一个简化的 Allan deviation 量级模型表示几类独立误差：

\[
\sigma_A^2(\tau)
\approx
\underbrace{\frac{A^2}{\tau}}_{\text{白噪声}}
+
\underbrace{B^2}_{\text{近似平台}}
+
\underbrace{K^2\tau}_{\text{bias/rate 随机游走}}
+
\underbrace{R^2\tau^2}_{\text{斜坡或温漂趋势}}
\]

这里 \(A,B,K,R\) 只是对应模型和单位约定下的系数。这个公式的价值不是要求
死记系数，而是让各项随 \(\tau\) 的变化一目了然：

| 时间尺度变化 | 典型主导项 | Allan deviation 变化 | 原因 |
|---|---|---|---|
| 接近单样本 | 量化、采样抖动、测量白噪声、高频振动 | 常见斜率 \(-1\) 或 \(-1/2\) | 窗口太短，平均尚未充分抵消高频误差 |
| 短时到中时 | 测量白噪声 | \(\propto\tau^{-1/2}\) | 平均样本数增加，独立噪声按 \(1/\sqrt m\) 下降 |
| 谷底/平台附近 | flicker、相关 bias、上电残余 bias 的缓慢变化 | 近似 \(\tau^0\) 或宽缓转折 | 白噪声已降低，低频 bias 开始可见 |
| 更长时间 | bias/rate 随机游走 | \(\propto\tau^{+1/2}\) | 随机驱动持续积分，变化量随 \(\sqrt\tau\) 增长 |
| 很长时间 | 温漂、老化、环境趋势、真实运动污染 | 可接近 \(\tau^{+1}\) 或更复杂 | 确定性趋势没有被统计平均消除 |

这些阶段可以重叠。曲线颜色或斜率只是告诉我们“哪一项当前最大”，并不表示
其他误差突然消失。

#### 3.3.2 主导项在哪里交接

交接点由两个误差模型量级相等决定。

白噪声与平台交接：

\[
\frac{A}{\sqrt{\tau_{WB}}}=B
\quad\Rightarrow\quad
\boxed{\tau_{WB}=\left(\frac{A}{B}\right)^2}
\]

平台与随机游走交接：

\[
B=K\sqrt{\tau_{BR}}
\quad\Rightarrow\quad
\boxed{\tau_{BR}=\left(\frac{B}{K}\right)^2}
\]

若忽略平台，白噪声与随机游走直接交接：

\[
\frac{A}{\sqrt{\tau_{WR}}}=K\sqrt{\tau_{WR}}
\quad\Rightarrow\quad
\boxed{\tau_{WR}=\frac{A}{K}}
\]

例如两台 IMU 的白噪声系数 \(A\) 相同，但第二台的随机游走系数 \(K\) 小 10
倍，则其长时上升段大约会推迟 10 倍时间出现。这就是“谁主导”必须结合具体
参数判断的原因。

#### 3.3.3 一阶 Gauss-Markov 在哪个时间尺度可见

一阶 Gauss-Markov 的关键尺度是相关时间 \(\tau_c\)：

- \(\tau\ll\tau_c\)：bias 在观察窗口内变化缓慢，看起来近似保持记忆；
- \(\tau\approx\tau_c\)：均值回归开始显著，Allan 曲线可能出现转折或宽缓区域；
- \(\tau\gg\tau_c\)：跨越多个相关时间后，长窗口平均会逐渐削弱该过程。

所以一阶 Gauss-Markov 不会像纯随机游走那样永远向上发散。若记录时长没有
覆盖数倍 \(\tau_c\)，就很难从数据中可靠识别其回正特性。

#### 3.3.4 机械编排误差属于另一条时间轴

coning、sculling 和 rotation error 主要由**采样周期内**的高动态变化决定，
不是 Allan 图中“长时间随机稳定性”的同一类问题：

- 子样尺度：高频角振动、线振动和不可交换交叉项可能主导单周期误差；
- 导航周期尺度：未补偿的二阶项形成每周期小偏差；
- 长时间：这些小偏差可能累积成姿态、速度和位置漂移。

因此高频机械编排误差与低频 bias 漂移可以同时存在，不能只靠 Allan 分析决定
是否需要 coning/sculling 补偿。

#### 3.3.5 从传感器误差到导航误差：主导关系会再次改变

Allan 图描述的是传感器 rate 数据在不同平均时间下的稳定性；导航系统关心的
却是误差经过一次或多次积分后如何进入姿态、速度和位置。两者不能直接等同。

先考虑无 GNSS、无零速更新的纯惯性传播，并忽略地球曲率等高阶项。

**陀螺测量白噪声**

角速度白噪声积分成姿态随机游走：

\[
\operatorname{Std}(\delta\theta)
\propto N_g\sqrt{t}
\]

若主要通过重力投影误差耦合到水平通道，则伪加速度标准差约按
\(t^{1/2}\) 增长，继续积分后速度和位置误差标准差分别约按
\(t^{3/2}\) 和 \(t^{5/2}\) 增长。

**陀螺常值 bias**

\[
\delta\theta(t)\approx b_g t
\]

它在线性时间内积累姿态误差。若该姿态误差使重力方向投影错误，小角度下产生
的水平伪加速度约为

\[
\delta a_h\approx g\,\delta\theta\approx g b_g t
\]

继续积分：

\[
\delta v_h\approx\frac{1}{2}g b_g t^2,\qquad
\delta p_h\approx\frac{1}{6}g b_g t^3
\]

所以一个看似很小的陀螺 bias，长时可能通过“姿态错误 \(\rightarrow\) 重力
泄漏 \(\rightarrow\) 速度/位置积分”成为最危险的误差源。

**加速度计测量白噪声**

\[
\operatorname{Std}(\delta v)\propto N_a\sqrt{t},\qquad
\operatorname{Std}(\delta p)\propto N_a t^{3/2}
\]

**加速度计常值 bias**

\[
\delta v(t)\approx b_a t,\qquad
\delta p(t)\approx\frac{1}{2}b_a t^2
\]

**Bias 随机游走**

Bias 自身标准差先按 \(t^{1/2}\) 增长，再进入导航积分。因此量级上：

- 加速度计 bias 随机游走导致速度误差约按 \(t^{3/2}\) 增长；
- 位置误差约按 \(t^{5/2}\) 增长；
- 陀螺 bias 随机游走导致姿态误差约按 \(t^{3/2}\) 增长；
- 再经重力耦合后，速度和位置误差分别约按 \(t^{5/2}\) 和
  \(t^{7/2}\) 增长。

这些幂次是用于理解主导关系的局部无辅助惯导量级，不是完整地球模型的精确
闭式解。它们假设局部水平、小姿态误差、近似恒定重力且各噪声独立。地球自转、
运输率、机动、阻尼和误差相关性会改变长期形态；真实 ESKF 中，GNSS、轮速、
视觉、LiDAR 或零速更新也会周期性约束误差，使误差不再无限按上述幂次增长。

可用下表区分“传感器上看起来大”和“导航结果中最终危险”：

| 误差源 | 传感器短时表现 | 无辅助导航中的典型增长 |
|---|---|---|
| 陀螺白噪声 | rate 高频抖动明显 | 姿态 \(t^{1/2}\)，重力耦合位置约 \(t^{5/2}\) |
| 陀螺常值 bias | 原始曲线上可能不显眼 | 姿态 \(t\)，重力耦合位置可达 \(t^3\) |
| 加速度计白噪声 | 比力高频抖动 | 速度 \(t^{1/2}\)，位置 \(t^{3/2}\) |
| 加速度计常值 bias | 小的固定偏移 | 速度 \(t\)，位置 \(t^2\) |
| 加速度计 bias 随机游走 | 长时间才明显 | 速度 \(t^{3/2}\)，位置 \(t^{5/2}\) |
| 陀螺 bias 随机游走 | 长时间才明显 | 姿态 \(t^{3/2}\)，重力耦合位置约 \(t^{7/2}\) |
| 温漂/未建模趋势 | 环境变化时出现 | 可表现为近似 bias 或斜坡并持续积累 |

因此，“谁主导”必须说明观察对象：

1. 原始 IMU 单样本；
2. Allan deviation；
3. 姿态误差；
4. 速度误差；
5. 位置误差；
6. 有外部观测闭环后的 ESKF 误差。

不说明观察对象而直接说“白噪声主导”或“bias 主导”，结论是不完整的。

#### 3.3.6 在 GNSS/IMU 融合系统中谁主导

外部观测会截断纯惯性误差的自由增长，因此还要比较误差时间尺度与观测更新
周期 \(T_{\text{update}}\)：

| 工作阶段 | 常见主导问题 | 原因 |
|---|---|---|
| 单个 IMU 子样内 | 量化、带宽内白噪声、振动、时间戳抖动 | 尚未经过平均或外部约束 |
| 一个导航周期内 | coning/sculling、姿态插值、子样丢失 | 三维增量按时间顺序组合 |
| 两次 GNSS 更新之间 | 测量白噪声、当前 bias 估计误差、机械编排残差 | 主要由 IMU 自由传播 |
| GNSS 连续可用时 | GNSS 噪声、杆臂、时延、观测异常和模型一致性 | 外部观测周期性限制漂移 |
| 短时 GNSS 失锁 | 陀螺/加速度计 residual bias 通常迅速重要 | bias 误差按 \(t,t^2,t^3\) 链式传播 |
| 长时失锁或温度变化 | bias 随机过程、温漂、比例因子、安装误差 | 缺少可观测约束，低频误差持续积累 |

这张表是因果顺序，不是固定秒数表。例如 `1 Hz` GNSS 的
\(T_{\text{update}}\approx1s\)，而 `10 Hz` 视觉里程计只有约 `0.1s`。同一
IMU 在两套系统中的误差主导区间会不同。

还要注意：比例因子、非正交、安装角和时间同步误差是输入或运动相关的系统误差，
不一定能从静止 Allan 曲线中识别。车辆静止时它们可能几乎不显现，高动态时却
可能超过随机噪声。因此真实评估必须同时包含静态噪声实验和动态轨迹实验。

### 3.4 测量白噪声：每次读数独立抖动

设静止传感器的理想输出为零，实际每次读数为 \(n_k\)，并满足

\[
\operatorname E[n_k]=0,\qquad
\operatorname{Var}(n_k)=\sigma^2,\qquad
\operatorname{Cov}(n_i,n_j)=0\;(i\ne j)
\]

最后一条表示不同采样之间没有线性相关性。因为正负误差会部分抵消，平均
\(m\) 个样本后，均值的方差变为

\[
\operatorname{Var}\left(\frac{1}{m}\sum_{i=1}^{m}n_i\right)
=\frac{\sigma^2}{m}
\]

所以白噪声可以通过平均降低，但不能知道并减去某一个样本的随机误差。

白噪声均值可接近零，但积分后形成随机游走。连续时间噪声密度必须按
离散时间间隔正确换算，不能直接把数据手册数值填入离散协方差。

#### 从噪声密度到离散样本

这里必须先声明功率谱密度采用单边还是双边约定，否则不同资料之间可能相差
\(\sqrt{2}\)。

本文优先用连续时间白噪声强度 \(q_g\) 定义：

\[
\operatorname E[n_g(t)n_g(t')]=q_g\delta(t-t')
\]

并记 \(N_g=\sqrt{q_g}\)。\(N_g\) 的量纲可写成
\(\mathrm{rad/s/\sqrt{Hz}}\)，也可写成 \(\mathrm{rad/\sqrt{s}}\)。
在这个连续时间强度约定下，一个采样周期 \(\Delta t\) 内的角度噪声标准差满足

\[
\operatorname{Std}(\delta\theta)
=N_g\sqrt{\Delta t}
\]

若 IMU 输出的角速度 rate 表示该采样周期内的平均值，且用
\(\delta\theta=n_g\Delta t\)
理解该周期的噪声积分，则对应 rate 样本标准差为

\[
\operatorname{Std}(n_g)
=\frac{N_g}{\sqrt{\Delta t}}
\]

这解释了一个看似反直觉的现象：在理想白噪声模型和相应带宽假设下，采样周期
越短，单个 rate 样本看起来越抖；但每个样本作用时间更短，积分后的角度噪声
仍按 \(\sqrt{\Delta t}\) 缩放。

数据手册若给出单边幅度谱密度 \(N_{g,\mathrm{1s}}\)，在常见定义下有
\(q_g=N_{g,\mathrm{1s}}^2/2\)，但厂商也可能把等效噪声带宽、数字滤波器
或 ARW 换算吸收到规格中。真实 IMU 改变输出数据率时，内部滤波带宽也可能
改变，此时样本标准差不会严格按 \(1/\sqrt{\Delta t}\) 缩放。因此工程上
必须同时核对规格定义、输出带宽和采样语义，不能只凭单位直接代入。

在 ESKF 离散化中也必须遵循相同量纲。连续 PSD 先进入
\(\mathbf Q_c\)，再通过状态转移和噪声映射积分得到

\[
\mathbf Q_d
=\int_0^{\Delta t}
\boldsymbol\Phi(\Delta t-\tau)
\mathbf G\mathbf Q_c\mathbf G^T
\boldsymbol\Phi^T(\Delta t-\tau)\,d\tau
\]

简单系统中才可近似为
\(\mathbf Q_d\approx\mathbf G\mathbf Q_c\mathbf G^T\Delta t\)。
因此不能看到数据手册中的噪声密度后，不检查单位就直接把同一个数字填进
离散协方差矩阵。

在 ESKF 中，\(n_a,n_g\) 不作为状态保存，而是通过噪声映射矩阵
\(\mathbf G\) 和连续时间谱密度 \(\mathbf Q_c\) 增大协方差。

### 3.5 Bias 随机游走：每次迈一步，过去不会自动消失

可把 bias 随机游走想象成一个人在直线上连续随机迈步。每一步均值为零，
但走过的步数越多，离起点的典型距离越大：

\[
\dot{\mathbf b} = \mathbf n_b
\]

它没有有限稳态方差，长时间会无界扩散。ESKF 中常把陀螺和加速度计
bias 放入误差状态，并由白色过程噪声驱动。离散形式为

\[
\mathbf b_{k+1}
=\mathbf b_k+\mathbf w_k,\qquad
\mathbf w_k\sim\mathcal N(\mathbf 0,\mathbf Q_b\Delta t)
\]

经过 \(k\) 步后，各次独立扰动的方差相加：

\[
\operatorname{Var}(\mathbf b_k)
=\operatorname{Var}(\mathbf b_0)+k\mathbf Q_b\Delta t
\]

因此随机游走的标准差按 \(\sqrt{t}\) 增长，方差按 \(t\) 增长。这里
\(\mathbf n_b\) 是白色驱动噪声，而 \(\mathbf b\) 由于积累了历史，不是白噪声。

### 3.6 一阶 Gauss-Markov：带回正力的随机漂移

一阶 Gauss-Markov 可以想象成“随机迈步的人被一根弹簧拴在原点”：

- \(\mathbf n_b\) 不断施加随机扰动；
- \(-\mathbf b/\tau_c\) 把偏离零点的状态拉回去；
- \(\tau_c\) 越大，拉回越慢，记忆越长。

\[
\dot{\mathbf b}
=-\frac{1}{\tau_c}\mathbf b+\mathbf n_b
\]

\(\tau_c\) 是相关时间。与随机游走不同，它具有均值回归和有限稳态方差。
若数据表明 bias 有明显相关时间，该模型通常比纯随机游走更贴近物理现象。

若稳态标准差为 \(\sigma_b\)，一种常见连续时间写法是

\[
\dot{\mathbf b}
=-\frac{1}{\tau_c}\mathbf b
+\sqrt{\frac{2\sigma_b^2}{\tau_c}}\mathbf w(t)
\]

其精确离散形式为

\[
\mathbf b_{k+1}
=e^{-\Delta t/\tau_c}\mathbf b_k+\mathbf w_k
\]

\[
\operatorname{Cov}(\mathbf w_k)
=\sigma_b^2\left(1-e^{-2\Delta t/\tau_c}\right)\mathbf I
\]

因此，一阶 Gauss-Markov 的状态转移不是随机游走模型中的 \(1\)，而是
\(e^{-\Delta t/\tau_c}\)。不能只在名称上把二者互换。

其自相关函数具有指数衰减形式：

\[
R_b(\Delta)=\sigma_b^2e^{-|\Delta|/\tau_c}
\]

当时间间隔等于 \(\tau_c\) 时，相关性降到初值的 \(e^{-1}\approx0.37\)。
这才是“相关时间”的具体含义，它不是 Allan 分组使用的聚类时间 \(\tau\)。

### 3.7 Bias instability 与闪烁噪声

Allan 曲线上的近水平区域通常称为 bias instability，其理想化频谱接近
\(1/f\)。它既不是逐样本白噪声，也不是单个一阶 Gauss-Markov 过程。
工程上可以使用以下近似：

- MVP：用 bias 随机游走吸收缓慢漂移，模型简单但不完全物理；
- 更真实的滤波器：使用一个或多个不同相关时间的一阶 Gauss-Markov 状态；
- 离线噪声仿真：直接生成近似 \(1/f\) 的有色噪声。

为什么它在 Allan 图上容易形成平台？直观上，短时间平均不能消除缓慢变化的
bias；但在某一段时间尺度内，延长平均时间也还没有让它像随机游走那样明显
增长，于是 Allan deviation 对 \(\tau\) 不敏感，表现为近似水平。

### 3.8 本项目 MVP 采用什么模型

第一版 GNSS/IMU ESKF 推荐采用：

\[
\begin{aligned}
\boldsymbol\omega_m
&=\boldsymbol\omega+\mathbf b_g+\mathbf n_g,\\
\mathbf f_m
&=\mathbf f+\mathbf b_a+\mathbf n_a,\\
\dot{\mathbf b}_g&=\mathbf n_{bg},\\
\dot{\mathbf b}_a&=\mathbf n_{ba}.
\end{aligned}
\]

其中：

- \(\mathbf n_g,\mathbf n_a\)：**测量白噪声**；
- \(\mathbf n_{bg},\mathbf n_{ba}\)：**驱动 bias 随机游走的白噪声**；
- \(\mathbf b_g,\mathbf b_a\)：**随机游走状态，不是白噪声**；
- MVP 暂不使用一阶 Gauss-Markov；获得足够长的真实静态数据并估计出可靠
  \(\tau_c\) 后，再将 bias 状态转移改为一阶 Gauss-Markov。

### 3.9 为什么测量白噪声不作为状态，bias 却作为状态

状态表示“当前值会影响未来，并且可以通过后续观测逐渐估计”的量。运行中
bias 具有时间连续性：若这一时刻 bias 偏大，下一时刻通常仍然接近该值，因此
值得放入状态向量。

逐样本白噪声没有这种记忆。即使滤波器知道上一时刻噪声为正，也不能据此预测
下一时刻噪声。因此它不适合作为持续状态，而是通过
\(\mathbf G\mathbf Q_c\mathbf G^T\)
描述“传播一步后增加了多少不确定性”。

简化地说：

```text
bias：不知道当前具体值，但它会延续 → 估计为状态
白噪声：下一样本重新随机抽取       → 作为过程噪声输入
```

对应图片：
[`imu_error_2_random.png`](../visual_explanations/outputs/imu_error_2_random.png)。

## 4. Allan Deviation 应如何读

### 4.1 Allan deviation 到底在测什么

普通方差把整段数据混在一起，只回答“总体波动多大”。但 IMU 误差与时间尺度
有关：

- 接近采样周期时，通常首先看到量化、高频振动和测量白噪声；
- 增大平均时间后，白噪声降低，相关 bias 或 flicker 才可能显现；
- 再增大平均时间后，随机游走、温漂和老化可能超过前两者。

这里没有给出固定秒数，因为各阶段边界应由 3.3 节中的系数量级和交接公式
决定。图中的具体秒数只属于当前合成数据，不应推广到另一台 IMU。

Allan deviation 的核心问题是：

> 如果分别把相邻两段、每段长度为 \(\tau\) 的数据取平均，这两个平均值通常
> 相差多少？

如果传感器只有常值 bias \(b_0\)，相邻两段平均值都包含同一个 \(b_0\)，
相减后常值被消掉。因此 Allan deviation 不负责估计固定 bias，而是观察
“平均值在不同时间尺度上变化得有多快”。

可以把计算过程想象成使用不同粗细的尺子观察同一条时间序列：

```text
tau 很小： [平均1][平均2][平均3]...  看到快速抖动
tau 中等： [   平均1   ][   平均2   ]  看到缓慢变化
tau 很大： [       平均1       ][       平均2       ]
```

这里的 \(\tau\) 是**聚类时间/平均时间**，不是一阶 Gauss-Markov 的相关时间
\(\tau_c\)。

设静态 IMU rate 序列为 \(y_0,\ldots,y_{N-1}\)，采样周期为
\(\Delta t=1/f_s\)。对给定聚类时间 \(\tau\)，每组包含的样本数为

\[
m=\operatorname{round}\left(\frac{\tau}{\Delta t}\right)
=\operatorname{round}(\tau f_s)
\]

实际使用的聚类时间是 \(\tau_m=m\Delta t\)，有效的非重叠组数为

\[
K=\left\lfloor\frac{N}{m}\right\rfloor
\]

第 \(k\) 组的平均 rate 为

\[
\bar y_k(\tau_m)
=\frac{1}{m}\sum_{i=0}^{m-1}y_{km+i},
\qquad k=0,\ldots,K-1
\]

理论期望形式的 Allan variance 为

\[
\sigma_A^2(\tau)
=\frac{1}{2}\operatorname E\left[
(\bar y_{k+1}(\tau)-\bar y_k(\tau))^2
\right]
\]

作为教学基线，非重叠有限样本估计为：

\[
\hat\sigma_A^2(\tau_m)
=\frac{1}{2(K-1)}
\sum_{k=0}^{K-2}
\left[
\bar y_{k+1}(\tau_m)-\bar y_k(\tau_m)
\right]^2
\]

图上绘制的是 Allan deviation，即 Allan variance 的平方根：

\[
\hat\sigma_A(\tau_m)
=\sqrt{\hat\sigma_A^2(\tau_m)}
\]

非重叠代码中的

```python
np.sqrt(0.5 * np.mean(np.diff(means) ** 2))
```

等价于上述公式，因为 `np.diff(means)` 有 \(K-1\) 项，而 `np.mean`
已经除以 \(K-1\)。

### 4.2 为什么白噪声对应斜率 \(-1/2\)

设每个 rate 样本都是独立白噪声，方差为 \(\sigma^2\)。长度为 \(\tau\)
的分组包含

\[
m=\frac{\tau}{\Delta t}
\]

个样本。对 \(m\) 个独立样本取平均后：

\[
\operatorname{Var}(\bar y_k)=\frac{\sigma^2}{m}
\]

相邻两个分组相互独立，因此差值的方差是两者方差之和：

\[
\operatorname{Var}(\bar y_{k+1}-\bar y_k)
=\frac{2\sigma^2}{m}
\]

Allan variance 定义前面的 \(1/2\) 正好消去这个 2：

\[
\sigma_A^2(\tau)
=\frac{\sigma^2}{m}
=\sigma^2\frac{\Delta t}{\tau}
\]

开平方：

\[
\sigma_A(\tau)
=\sigma\sqrt{\Delta t}\,\tau^{-1/2}
\]

两边取对数：

\[
\log\sigma_A
=-\frac{1}{2}\log\tau+\text{常数}
\]

因此在 log-log 图上斜率为 \(-1/2\)。这不是经验口诀，而是“独立白噪声
平均 \(m\) 次后，标准差缩小为 \(1/\sqrt m\)”的直接结果。

一个数值直觉：若把平均时间扩大 100 倍，白噪声 Allan deviation 会缩小到
原来的 \(1/\sqrt{100}=1/10\)。

### 4.3 为什么 bias 随机游走对应斜率 \(+1/2\)

随机游走满足

\[
b(t+\tau)-b(t)=\int_t^{t+\tau}w(s)\,ds
\]

这里的 \(b(t)\) 表示完成离线标定后仍存在的**运行中残余 bias 状态**：
陀螺可写为 \(b_g(t)\)，单位为 `rad/s`；加速度计可写为 \(b_a(t)\)，
单位为 `m/s²`。\(w(t)\) 是驱动 bias 变化的白噪声。为了突出时间尺度规律，
公式暂时省略了轴下标。

其中 \(w\) 是白色驱动噪声。独立扰动在长度为 \(\tau\) 的时间内不断累积，
所以增量方差与时间成正比：

\[
\operatorname{Var}[b(t+\tau)-b(t)]\propto\tau
\]

标准差因此满足

\[
\operatorname{Std}[b(t+\tau)-b(t)]\propto\sqrt{\tau}
\]

相邻分组平均值的差也具有相同的时间尺度规律，只差一个与定义有关的常数：

\[
\sigma_A(\tau)\propto\tau^{+1/2}
\]

所以 log-log 斜率为 \(+1/2\)。平均时间越长，看到的不是更多独立噪声被抵消，
而是 bias 在更长时间里累积了更多随机变化。

### 4.4 为什么 bias instability 附近斜率约为 0

若某一时间尺度内，误差的典型变化幅度几乎不随 \(\tau\) 改变，则

\[
\sigma_A(\tau)\approx C
\]

取对数后：

\[
\log\sigma_A\approx\log C
\]

因此斜率约为 0。理想 flicker noise 的 Allan deviation 会形成近似平台。
真实数据通常只是“局部接近水平”，不应期待无限宽、完全平坦的平台。

需要注意：一阶 Gauss-Markov 过程也可能在有限区间形成宽缓转折，所以仅凭
一小段近零斜率不能唯一证明物理噪声一定是 flicker noise。

### 4.5 为什么总曲线常呈 U 型

若不同噪声分量近似独立，它们的 Allan **variance** 近似相加：

\[
\sigma_{A,total}^2(\tau)
\approx
\sigma_{A,white}^2(\tau)
+\sigma_{A,flicker}^2(\tau)
+\sigma_{A,rw}^2(\tau)
\]

注意相加的是平方，不是直接把三条 Allan deviation 相加。

- 小 \(\tau\)：白噪声项最大，曲线按 \(\tau^{-1/2}\) 下降；
- 中等 \(\tau\)：白噪声已被平均压低，低频 bias 项形成谷底或平台；
- 大 \(\tau\)：随机游走项按 \(\tau^{+1/2}\) 上升。

三个主导区间拼起来就形成近似 U 型。若记录不够长、某种噪声过弱、存在温漂
或传感器本身不含明显平台，曲线不一定呈现教科书式 U 型。

#### 实际判读时如何确定各阶段

不要先按横轴秒数划区间，应按以下顺序：

1. 从最小可信 \(\tau\) 开始计算局部 log-log 斜率。
2. 找到连续多个点接近 \(-1/2\) 的区域，作为白噪声候选段。
3. 观察曲线是否出现接近 0 的宽缓区域，并检查温度和趋势是否污染。
4. 找到持续接近 \(+1/2\) 的区域，作为随机游走候选段。
5. 检查每个候选段的差分对数量、EDF 或置信区间。
6. 用局部拟合得到 \(A,B,K\)，再计算
   \(\tau_{WB},\tau_{BR},\tau_{WR}\)，检查它们是否与图中交接位置一致。
7. 换轴、换采集批次和换温度重复分析；只有可重复的阶段才适合作为设备参数。

在交接附近，两项量级接近，没有唯一“主导者”。此时总曲线斜率会处于两个
理论斜率之间。例如白噪声向平台过渡时，局部斜率可能从 \(-1/2\) 平滑变到
0，而不是突然跳变。

### 4.6 真实数据优先使用重叠 Allan deviation

对每一个起点 \(j\) 都计算长度为 \(m\) 的滑动平均：

\[
\bar y_j^{(m)}
=\frac{1}{m}\sum_{i=0}^{m-1}y_{j+i},
\qquad j=0,\ldots,N-m
\]

然后比较相距 \(m\) 个样本的两个相邻时间块：

\[
\hat\sigma_{A,\mathrm{overlap}}^2(\tau_m)
=\frac{1}{2(N-2m+1)}
\sum_{j=0}^{N-2m}
\left(
\bar y_{j+m}^{(m)}-\bar y_j^{(m)}
\right)^2
\]

当前项目使用累积和计算滑动平均，避免对每个窗口重复求和。其复杂度对每个
\(\tau\) 近似为 \(O(N)\)，而不是朴素实现的 \(O(Nm)\)。

两种方法的工程区别：

| 项目 | 非重叠 Allan | 重叠 Allan |
|---|---|---|
| 窗口移动 | 每次移动 \(m\) 个样本 | 每次移动 1 个样本 |
| 差分对数量 | 约 \(N/m-1\) | \(N-2m+1\) |
| 长 \(\tau\) 曲线 | 波动较明显 | 通常更稳定 |
| 数据利用率 | 较低 | 高 |
| 相邻差分独立性 | 相对较高 | 高度相关 |
| 推荐用途 | 教学、快速核对 | 真实 IMU 参数分析 |

重叠方法产生更多差分对，但这些差分对共享大量样本，不能将
\(N-2m+1\) 直接当作独立自由度。严格置信区间需要根据噪声类型估计
等效自由度（EDF），再使用卡方分布构造区间。当前 MVP 图只显示原始差分对
数量，用于发现长 \(\tau\) 端数据不足，不声称它就是置信度。

图中的“至少 20 对”只是可视化告警门槛，不是标准规定。真实报告应同时给出：

- 总采集时长和采样率；
- 每个 \(\tau\) 的差分对数量或 EDF；
- 是否使用重叠估计；
- 数据预处理方法；
- 参数拟合区间和拟合斜率。

### 4.7 真实 IMU 日志预处理

Allan 分析假设输入是等间隔、平稳的 rate 数据。推荐流程：

1. **检查时间戳**：严格递增，统计中位采样周期、RMS 抖动和最大间隔。
2. **识别丢包和重采样**：长间断应切成连续数据段，不应跨间断直接插值。
3. **确认数据类型**：角速度/比力可以直接分析；角增量/速度增量需除以对应
   \(\Delta t\) 转回 rate，尤其要处理可变采样周期。
4. **统一单位**：陀螺推荐 `rad/s`，加速度计推荐 `m/s²`。
5. **静止检测**：剔除碰撞、移动、饱和和明显外界振动区间。
6. **预热**：MEMS IMU 通常需要上电预热；预热段不应混入稳态噪声分析。
7. **温度记录**：先画温度与各轴输出。明显温漂应单独建模，而不是直接归为随机游走。
8. **异常值处理**：保留原始记录和剔除规则；不能为了得到漂亮 U 型随意平滑。
9. **去趋势需谨慎**：减去常值均值不会改变 Allan variance；线性去趋势会压低
   长 \(\tau\) 区域，应同时保存未去趋势结果。
10. **逐轴计算**：X/Y/Z 轴分别分析，不应先取三轴模长，否则噪声分布和重力项会耦合。

项目中的 `sampling_interval_statistics()` 可用于第一步。若
`max_relative_gap` 明显大于 1，或时间抖动相对采样周期不可忽略，应先修复
数据分段/时间同步问题，再运行 Allan 分析。

对于 increment 数据，还必须确认设备时间戳表示区间起点、区间终点还是数据
到达时刻。本项目 CSV 工具约定第 \(i\) 行增量对应
\([t_{i-1},t_i]\)，因此使用

\[
\text{rate}_i
=\frac{\text{increment}_i}{t_i-t_{i-1}}
\]

并丢弃第一行。若设备协议采用其他时间戳语义，必须先适配，不能机械套用。

### 4.8 聚类时间范围如何选择

最小聚类时间通常不小于一个采样周期：

\[
\tau_{\min}\ge\Delta t
\]

最大聚类时间不能只由公式是否可计算决定。即使重叠算法还能输出数值，长
\(\tau\) 端也可能只有很少的独立信息。工程上常把最大 \(\tau\) 限制为总时长
的约 \(1/10\) 到 \(1/3\)，并结合差分对数量、EDF 和置信区间判断。

聚类时间通常使用对数均匀网格：

\[
\tau_i=10^{a+i(b-a)/(L-1)}
\]

再映射到整数 \(m_i=\operatorname{round}(\tau_i f_s)\)。映射后可能出现重复
\(m_i\)，工程实现应去重，否则会重复绘制相同聚类尺度。

### 4.9 从曲线提取噪声参数

参数不能只读一个全局最小值，应先在 log-log 空间寻找目标斜率区间，再做
局部线性拟合：

\[
\log_{10}\sigma_A
=s\log_{10}\tau+c
\]

为什么常把拟合线外推到 \(\tau=1\)？因为
\(\log_{10}1=0\)，此时

\[
\log_{10}\sigma_A(1)=c,\qquad
\sigma_A(1)=10^c
\]

截距可以直接变成所选噪声模型在 1 秒尺度上的系数。但它的物理单位仍取决于
输入是 `rad/s`、`deg/s`、`m/s²`，以及采用哪套标准定义。

例如某段拟合得到

\[
\sigma_A(\tau)=0.02\,\tau^{-1/2}
\]

则 \(\tau=1s\) 时 Allan deviation 为 `0.02`（单位继承输入 rate 的单位），
而 \(\tau=100s\) 时为 `0.002`。这个例子同时验证了 \(-1/2\) 斜率的含义。

仅知道斜率还不够，截距必须按所选随机过程模型转换。以本文连续时间约定为例，
若测量白噪声满足

\[
\operatorname E[n_g(t)n_g(t')]=q_g\delta(t-t')
\]

则

\[
\sigma_A^2(\tau)=\frac{q_g}{\tau}
\]

若拟合式为
\(\sigma_A(\tau)=C_{-1/2}\tau^{-1/2}\)，则

\[
q_g=C_{-1/2}^2
\]

这就是常说的“把 \(-1/2\) 线外推到 \(\tau=1\) 读取 ARW/VRW”的数学前提。
若输入规格采用单边 PSD，则仍需先按其定义转换。

对于偏置随机游走模型

\[
db=\sqrt{q_b}\,dW
\]

其 Allan 方差为

\[
\sigma_A^2(\tau)=\frac{q_b\tau}{3}
\]

若 \(+1/2\) 区域拟合为
\(\sigma_A(\tau)=C_{+1/2}\tau^{1/2}\)，则

\[
q_b=3C_{+1/2}^2,\qquad
\sqrt{q_b}=\sqrt{3}\,C_{+1/2}
\]

所以不能把 \(+1/2\) 直线在 \(\tau=1\) 的数值原样填入 ESKF 的 bias
驱动 PSD。

对于理想 flicker noise，在常见 IEEE 惯性器件约定下：

\[
\sigma_A\approx0.664B,\qquad
B\approx\frac{\sigma_A}{0.664}
\]

这里的 \(\sigma_A\) 应来自可信的平台拟合，不一定等于整条有限数据曲线的
全局最小值。一阶 Gauss-Markov 的宽缓转折、温漂和数据不足都可能制造伪平台。

因此典型判读为：

- \(s\approx-1/2\)：测量白噪声，按 PSD 约定换算 ARW/VRW；
- \(s\approx0\)：候选 bias instability 平台，不能仅凭一个最低点确认；
- \(s\approx+1/2\)：rate random walk，按上式换算 bias 驱动强度；
- \(s\approx+1\)：可能是速率斜坡、温漂或未去除的确定性趋势。

不同资料对噪声系数、单边/双边 PSD 和单位的定义可能不同。提取参数时必须
把“拟合公式、单位转换和采用的标准”写进报告，不能只复制曲线上的数值。

### 4.10 如何用于 ESKF

真实 Allan 分析与 ESKF 的关系是：

- 白噪声段估计 \(\sigma_g,\sigma_a\)，进入连续时间测量噪声 PSD；
- 长期上升段用于设定 bias 随机游走驱动 \(\sigma_{bg},\sigma_{ba}\)；
- 若自相关函数显示明确相关时间 \(\tau_c\)，可将 bias 从随机游走升级为
  一阶 Gauss-Markov；
- bias instability 的平台不能不经模型转换就直接填入 \(\mathbf Q_c\)；
- 最终参数还应通过静态创新统计、动态 GNSS/IMU 融合和一致性指标调试。

Allan 分析告诉我们“传感器在不同时间尺度上的随机稳定性”，但不能单独识别
所有物理来源，也不能替代滤波器在真实运动数据上的验证。

由上述推导，log-log 图上的典型局部斜率可总结为：

- \(-1/2\)：独立白噪声平均 \(m\) 次后按 \(1/\sqrt m\) 衰减；
- \(0\)：误差幅度在该时间尺度内近似不随 \(\tau\) 改变；
- \(+1/2\)：随机扰动随时间积分，标准差按 \(\sqrt{\tau}\) 增长。

注意事项：

- 数据应尽量静止、等采样、无缺失，并先排除确定性温漂和明显趋势；
- 最大可信 \(\tau\) 远小于总数据时长，需要足够多的聚类；
- 一条有限长度曲线可能同时受多种噪声影响，斜率只支持模型辨识，不是唯一物理证明；
- 本项目使用合成的白噪声、低频相关项和 rate random walk，目的是验证斜率，不是伪造真实 IMU 指标。

对应图片：
[`imu_error_3_allan.png`](../visual_explanations/outputs/imu_error_3_allan.png)。
黑色实线为重叠估计，灰色点线为
非重叠对照；下方子图显示两种方法的原始差分对数量。

当前合成图中，为了让主导交接可见，约 `0.1～10 s` 主要展示白噪声下降，
约 `10～70 s` 展示平台/交接，约 `70 s` 以后展示随机游走上升。这些范围由
脚本中人为选择的 \(A,B,K\) 决定，只用于验证机制，不代表真实 IMU 的固定
时间分界。真实设备必须按 4.5 节的方法重新识别。

## 5. ESKF 前的 IMU 处理顺序

### 5.1 为什么处理顺序本身也是算法的一部分

IMU 处理不是若干补偿公式的任意排列。不同步骤所假设的数据含义不同：

- 标定矩阵作用于传感器坐标系中的原始测量；
- bias 改正需要知道每个子样对应的 \(\Delta t\)；
- coning/sculling 假设输入是已完成基本标定的子样增量；
- body 到 navigation 的旋转必须使用与增量时间区间一致的姿态；
- ESKF 的 \(\mathbf F,\mathbf G,\mathbf Q\) 必须与名义状态真正使用的测量一致。

例如，若先计算叉乘补偿再去 bias，叉乘中会出现
\(\mathbf b_g\times\Delta\mathbf v\) 等伪二阶项；若把所有速度增量都用周期末
姿态转到 navigation 系，又会抹掉周期内姿态变化的时间顺序。

本文统一使用
\(\mathbf C_b^n\)
表示把 body 分量转换为 navigation 分量：

\[
\mathbf v^n=\mathbf C_b^n\mathbf v^b
\]

上标/下标方向必须与代码中的 `quat_to_dcm()` 保持一致。若使用
\(\mathbf C_n^b\)，它是前者的转置，而不是同一个矩阵的另一种写法。

对于输出角增量和速度增量的 IMU，推荐按以下概念顺序处理：

1. 解码时间戳、单位、符号和轴定义。
2. 应用温度、bias、比例因子、非正交和安装标定。
3. 用当前 bias 估计修正每个子样：

\[
\Delta\boldsymbol\theta_i^c
=\Delta\boldsymbol\theta_i^m-\hat{\mathbf b}_g\Delta t_i
\]

\[
\Delta\mathbf v_i^c
=\Delta\mathbf v_i^m-\hat{\mathbf b}_a\Delta t_i
\]

4. 对子样做 coning、sculling 和 rotation 补偿。
5. 更新姿态，将 body 系比力增量转到 navigation 系。
6. 在 navigation 系速度方程中加入有效重力，并按所采用的模型加入地球自转、
   运输角速度和科氏项，再积分速度和位置。
7. 用同一线性化点构造 ESKF 的 \(\mathbf F,\mathbf G,\mathbf Q\)。

真实产品的接口可能已经在固件内完成部分补偿，必须查看数据定义，避免重复改正。
本项目当前合成图主要验证局部短时的 coning/sculling 结构，没有实现完整地球
模型；不能把图解中的“加重力”简写直接当作部署级导航方程。

## 6. 圆锥误差

### 6.1 为什么角增量不能像平移一样直接相加

平移基本满足交换律：先向东走 1 米再向北走 1 米，与交换顺序后的终点相同。
三维旋转不满足交换律。可以拿一本书做实验：

1. 先绕书本 X 轴转 \(90^\circ\)，再绕新的 Y 轴转 \(90^\circ\)；
2. 交换两次旋转顺序。

书本的最终朝向不同。这说明角增量除了大小和方向，还隐含发生顺序。

三维有限转动不可交换：

\[
\operatorname{Exp}(\Delta\boldsymbol\theta_1)
\operatorname{Exp}(\Delta\boldsymbol\theta_2)
\ne
\operatorname{Exp}(
\Delta\boldsymbol\theta_1+\Delta\boldsymbol\theta_2)
\]

对于两个小旋转，Baker-Campbell-Hausdorff 展开给出

\[
\operatorname{Exp}(\mathbf a)\operatorname{Exp}(\mathbf b)
\approx
\operatorname{Exp}\left(
\mathbf a+\mathbf b+\frac{1}{2}\mathbf a\times\mathbf b
\right)
\]

叉乘项就是最低阶的“不交换修正”。其方向垂直于两次角运动构成的平面，
大小与两向量围成的面积成正比，所以圆锥误差也可理解为角运动轨迹的面积效应。

下面的两子样系数是 \(2/3\)，而不是 BCH 中的 \(1/2\)。原因是两个子样是
连续变化角速度在两个半周期内的积分，不是两个瞬时、分段恒定旋转；在等间隔、
周期内角速度线性变化的假设下重新积分，系数得到 \(2/3\)。

对两个等时间子样、周期内角速度近似线性变化，一种经典两子样公式为

\[
\Delta\boldsymbol\theta =
\Delta\boldsymbol\theta_1+
\Delta\boldsymbol\theta_2+
\frac{2}{3}
(\Delta\boldsymbol\theta_1\times
\Delta\boldsymbol\theta_2)
\]

叉乘项描述旋转次序带来的二阶面积效应。它不是传感器 bias，也不能通过
简单提高 Kalman 增益消除。

一个反直觉现象是：X、Y 两轴角增量在完整圆锥周期内都可以均值为零，但每个
子周期的
\(\Delta\boldsymbol\theta_1\times\Delta\boldsymbol\theta_2\)
可能始终同号，因此 Z 向姿态误差持续累积。这解释了为什么“每轴都回到零”
并不等于“姿态回到原点”。

该 \(2/3\) 系数依赖等长度子区间和周期内角速度低阶变化近似。子区间明显
不等长、存在强高频振动或需要更高阶精度时，应使用带时间权重的广义多子样
公式，不能继续机械套用该系数。

对应图片：
[`coning_1_mechanism.png`](../visual_explanations/outputs/coning_1_mechanism.png)、
[`coning_2_spatial.png`](../visual_explanations/outputs/coning_2_spatial.png) 和
[`coning_3_impact.png`](../visual_explanations/outputs/coning_3_impact.png)。

## 7. 划桨与旋转效应

### 7.1 从连续速度积分理解交叉项

加速度计输出的比力用当前 body 坐标系表达。若载体在一个导航周期内旋转，
周期前半段和后半段的“body X 方向”在 navigation 系中不是同一方向。

导航系速度增量的连续形式是

\[
\Delta\mathbf v^n
=\int_{t_k}^{t_{k+1}}
\mathbf C_b^n(t)\mathbf f^b(t)\,dt
\]

姿态矩阵 \(\mathbf C_b^n(t)\) 和比力 \(\mathbf f^b(t)\) 都可能随时间变化。
简单相加 body 系速度增量，相当于假设整个周期 body 方向不变，因此会漏掉
角运动与线运动的耦合。

速度增量在不断旋转的 body 系中产生。仅累加
\(\Delta\mathbf v_1+\Delta\mathbf v_2\) 会遗漏角运动与线运动的耦合。
本项目采用

\[
\Delta\mathbf v =
\Delta\mathbf v_1+\Delta\mathbf v_2
+\frac{1}{2}
(\Delta\boldsymbol\theta_1+\Delta\boldsymbol\theta_2)
\times
(\Delta\mathbf v_1+\Delta\mathbf v_2)
\]

\[
\quad+\frac{2}{3}
(\Delta\boldsymbol\theta_1\times\Delta\mathbf v_2+
\Delta\mathbf v_1\times\Delta\boldsymbol\theta_2)
\]

其中 \(1/2\) 项表示导航周期内平均姿态方向造成的 rotation correction，
\(2/3\) 交叉项表示子样间的 sculling coupling。补偿后得到的仍是 body
系表达，随后才用周期起点姿态转入 navigation 系。

两类交叉项可这样理解：

- **rotation correction**：即使比力在 body 系恒定，只要 body 在转，
  它在 navigation 系中的平均方向就不同于周期起点方向；
- **sculling correction**：当角速度和比力都在子样间变化时，变化的先后顺序
  形成额外速度面积项。

这些项来自对
\(\mathbf C_b^n(t)\mathbf f^b(t)\)
进行时间展开和积分，不是为了让曲线好看而经验添加的系数。

上述两子样公式还假设角增量和速度增量已经完成时间同步、单位转换与确定性
标定，两个子样采用一致机体系和增量定义，并且设备固件没有提前做过同类
补偿。否则可能把标定误差卷入交叉项，或发生重复补偿。非均匀子区间和强高频
运动应使用广义多子样或更高阶积分。

### 7.2 一个思想实验

设载体先沿 body X 方向加速，随后快速转向。若直接把两段 body 速度增量相加，
算法会认为它们始终沿同一空间方向；真实情况中，后一段增量已经指向新的方向。
单周期速度误差可能很小，但速度继续积分会形成不断增长的位置误差。

对应图片：
[`sculling_1_mechanism.png`](../visual_explanations/outputs/sculling_1_mechanism.png)、
[`sculling_2_rotation.png`](../visual_explanations/outputs/sculling_2_rotation.png) 和
[`sculling_3_impact.png`](../visual_explanations/outputs/sculling_3_impact.png)。

## 8. 实现细节与验证

Allan 真实数据分析实现位于 `python/gnss_imu/imu_allan.py`，图解脚本通过
`visual_explanations/imu_visualization_math.py` 复用它。姿态与增量补偿
辅助函数仍位于后者：

- Hamilton 四元数顺序为 `[w, x, y, z]`；
- `quat_to_dcm` 返回 \(\mathbf C_{bn}\)，将 body 向量转到 navigation；
- 绘图函数 `attitude_error_rotvec(q_est, q_true)` 计算
  \(\operatorname{Log}(q_{\text{true}}^{-1}\otimes q_{\text{est}})\)，表示
  “估计相对真值”的右乘 body-frame 误差；
- ESKF 文档采用
  \(q_{\text{true}}=q_{\text{nom}}\otimes\delta q\)，其误差状态对应
  \(\delta q=q_{\text{nom}}^{-1}\otimes q_{\text{true}}\)，表示“真值相对
  名义值”的误差；
- 两者乘法方向相反，小角度旋转向量近似互为相反数。绘图误差函数不能不经
  转换就直接用于 ESKF 注入或误差状态真值；
- Allan deviation 同时实现重叠和非重叠估计，真实数据优先使用重叠形式；
- 细时间步四元数积分作为圆锥和划桨实验的数值参考真值。

### 8.1 如何把公式映射到代码

阅读代码时不要只看函数名，应逐项核对：

| 数学对象 | 代码位置 | 应检查什么 |
|---|---|---|
| \(m=\operatorname{round}(\tau f_s)\) | `_cluster_sizes()` | 是否取整、去重并限制最大尺度 |
| 分组平均 | `allan_deviation()` | 是否按 rate 数据平均 |
| 滑动平均 | `overlapping_allan_deviation()` | 累积和的窗口端点是否正确 |
| \(\frac12 E[(\bar y_{k+1}-\bar y_k)^2]\) | 两个 Allan 函数 | 差分间隔和 `0.5` 是否一致 |
| \(\Delta\theta_1\times\Delta\theta_2\) | `coning_correct()` | 叉乘顺序决定符号 |
| rotation/sculling 交叉项 | `sculling_rotation_correct()` | body/nav 表达与系数是否一致 |
| \(\mathbf C_b^n\) | `quat_to_dcm()` | 矩阵方向和四元数乘法约定 |

公式与代码的对应关系是一种验证手段：若无法解释某一行代码来自哪条假设或
哪一步积分，该实现就还不具备可审查性。

测试位于 `tests/python/test_imu_visualization_math.py`。测试重点是约定一致性和
公式结构，不把某一组合成轨迹上的改善倍数当作普适性能指标。

真实 CSV 的边界测试位于 `tests/python/test_imu_allan.py`，覆盖时间戳间断、
时间倒退、NaN、缺失列、单位转换、增量转 rate 和重复聚类尺度。

当前成熟度：

- 图解与合成实验：教学原型；
- Allan CSV 分析工具：验证过的 MVP；
- 完整 IMU 误差标定与 ESKF 前处理链：尚未达到可部署状态。

达到部署级之前仍需要真实设备长时间静态数据、温箱/转台或等效标定数据、
自动静止和饱和检测、EDF 置信区间、参数提取审计，以及动态融合回放验证。

## 9. 学习自检：不要看答案先尝试推导

1. 白噪声平均时间从 \(1s\) 增加到 \(100s\)，Allan deviation 为什么约缩小
   10 倍，而不是 100 倍？
2. 常值 bias 为什么不会直接出现在 Allan deviation 中？
3. bias 随机游走的驱动是白噪声，为什么 bias 本身不是白噪声？
4. 一阶 Gauss-Markov 的 \(\tau_c\) 与 Allan 聚类时间 \(\tau\) 有什么区别？
5. 为什么三个独立噪声分量组合时，应近似相加 Allan variance，而不是直接相加
   Allan deviation？
6. 若 X、Y 圆锥角增量周期均值都为零，为什么 Z 姿态误差仍可能累计？
7. 为什么 sculling 补偿后还要明确增量属于 body 系还是 navigation 系？
8. 为什么不能把 Allan 平台读出的数值直接填进 ESKF 的 \(\mathbf Q_c\)？
9. 两台 IMU 的白噪声系数相同，但第二台随机游走系数小 10 倍，白噪声与随机
   游走的交接时间如何变化？
10. 为什么静态 Allan 曲线很好，仍不能证明高动态 coning、安装角和时间同步
    误差足够小？

能够用方差缩放、时间积分和坐标系变化解释这些问题，才算真正理解本章。

## 10. 常见面试问题

1. 为什么加速度计静止时输出不是零？
   因为加速度计测量单位质量的非引力合力。静止时桌面支撑力不为零，其对应
   比力在导航系中近似为当地有效重力的反方向。
2. 标定 bias 与 ESKF bias 状态有什么区别？
   前者是可重复的确定性常值/温度曲线；后者描述每次上电和运行中的残余变化。
3. 白噪声能否被逐样本改正？
   不能，只能统计建模、滤波和通过更好传感器或带宽设计减小影响。
4. 为什么圆锥误差在各轴角增量周期均值为零时仍会累积？
   因为旋转乘法不可交换，二阶叉乘面积项可以具有非零均值。
5. Allan deviation 能直接给出 ESKF 的 \(\mathbf Q\) 吗？
   它给出候选噪声模型和量级；仍需依据状态模型、单位和离散化推导 \(\mathbf Q_d\)。
6. 为什么先做标定再做 coning/sculling？
   交叉项会把比例因子和 bias 误差耦合进去，错误顺序会制造额外二阶误差。

## 11. 常见错误与当前边界

- 把角度误差直接标成 `deg/h`，而没有除以实验时长；
- 把频率的 `rad/s` 和 `Hz` 混用；
- 对 body 系速度增量求和后直接与 navigation 系真值比较；
- 用几十秒数据声称准确辨识 bias instability；
- 忽略四元数乘法方向、\(\mathbf C_{bn}\) 定义和误差状态左右乘约定；
- 认为两子样公式在任意高动态、非均匀采样下都精确；
- 将合成实验改善倍数当作真实设备性能承诺。

当前原型没有覆盖地球自转、运输率、曲率半径、杆臂、时间同步误差、振动整流误差
和真实 IMU 数据拟合。这些应在 ESKF MVP 的基本 mechanization 稳定后逐项加入。

## 12. 参考资料与约定来源

本文以仓库内统一符号和可执行实现为主，同时参考以下经典资料核对定义：

1. W. J. Riley, [*Handbook of Frequency Stability Analysis*,
   NIST Special Publication 1065](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication1065.pdf)：
   Allan 方差、重叠估计与噪声斜率。
2. A. Makdissi, F. Vernotte, E. De Clercq,
   [“Stability Variances: A Filter Approach”](https://arxiv.org/abs/0904.2660)：
   从滤波器和功率谱角度理解 Allan 方差。
3. Y. Wu, I. Litmanovich,
   [“Strapdown Attitude Computation: Functional Iterative Integration versus
   Taylor Series Expansion”](https://arxiv.org/abs/1909.09935)：
   捷联姿态更新、圆锥误差与多子样思想。
4. Y. Wu, X. Pan,
   [“Velocity/Position Integration Formula (II): Application to Inertial
   Navigation Computation”](https://arxiv.org/abs/1207.1553)：
   速度增量积分与划桨/旋转耦合。
5. IEEE Std 952 的惯性器件噪声术语与 bias instability 常用系数约定。

不同标准和厂商可能采用不同的单边/双边功率谱、角速度/角增量输入及单位约定。
实际配置参数时，应让“公式定义、单位、采样语义和带宽条件”同时匹配，不能
只比较参数名称或曲线截距。
