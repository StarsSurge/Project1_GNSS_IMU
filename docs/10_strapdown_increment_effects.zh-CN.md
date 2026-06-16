# 捷联惯导增量编排：圆锥、划桨与旋转效应

如果阅读过程中遇到术语不清楚，可以先查：

```text
docs/99_glossary.zh-CN.md
```

## 1. 这三个效应回答什么问题

IMU 在一个导航更新周期内输出两类核心增量：

- 陀螺角增量 \(\Delta\boldsymbol\theta\)，单位 rad；
- 加速度计速度增量 \(\Delta\mathbf v\)，单位 m/s，通常先表示在 body 坐标系。

真正的问题不是“能不能把这些小量相加”，而是：

> 当载体在一个更新周期内同时转动、振动和加速时，用有限个子样本近似连续运动，会漏掉哪些二阶项？

圆锥效应、划桨效应和旋转效应都属于捷联机械编排层的问题。它们不是传感器 bias，也不是 Kalman 滤波器在线估计出来的状态；它们是在把 IMU 增量送入姿态、速度、位置更新之前要处理的离散积分误差。

对应的动态图位于：

```text
visual_explanations/strapdown_increment_effects_animation.html
```

直接用浏览器打开即可。

建议按每个画布下方的链式步骤阅读，而不是只盯着运动的线条：

1. 先看 **输入**：这个导航周期里有哪些 IMU 子样本或总增量；
2. 再看 **第一步**：真实物理过程按什么顺序发生；
3. 然后看 **第二步**：简单算法做了什么错误近似；
4. 接着看 **缺失机制**：哪个叉乘项被漏掉，方向和单位是什么；
5. 最后看 **最终输出**：补偿后的姿态增量或速度增量应该送到机械编排的下一层。

如果你只能看到“线在移动”，说明还没有抓住观察目标。圆锥效应要看第三轴小转角从哪里来；划桨效应要看横向速度为什么由角振动和线振动共同产生；旋转效应要看同一个 body 前向速度增量为什么在导航系中有一串不同方向。

## 2. 坐标、变量和约定

本文采用以下约定：

- \(b\)：body 坐标系，固连于 IMU；
- \(n\)：navigation 坐标系，本仓库通常按局部导航系理解；
- \(\mathbf C_b^n\)：把 body 分量表示的向量旋转到 navigation 分量；
- \(\Delta\boldsymbol\theta_i\in\mathbb R^3\)：第 \(i\) 个子样本陀螺角增量，单位 rad；
- \(\Delta\mathbf v_i\in\mathbb R^3\)：第 \(i\) 个子样本加速度计速度增量，单位 m/s；
- 叉乘 \(\mathbf a\times\mathbf b\) 服从右手定则。

### 2.1 什么是导航更新周期和子样本

真实系统里通常有两个时间尺度：

- **IMU 原始采样周期**：传感器硬件输出一次陀螺/加速度计读数的间隔。例如 1000 Hz IMU 的原始采样周期是 1 ms。
- **导航更新周期**：惯导机械编排更新一次姿态、速度、位置的间隔。例如系统每 10 ms 更新一次导航状态。

如果 IMU 是 1000 Hz，而导航解算是 100 Hz，那么一个导航更新周期 \(T=10\,\text{ms}\) 内会收到 10 个 IMU 原始样本。

陀螺原始读数通常是角速度 \(\boldsymbol\omega\)，单位 rad/s。把某一小段时间内的角速度积分起来，就得到角增量：

\[
\Delta\boldsymbol\theta
\approx
\boldsymbol\omega\,\Delta t
\]

单位检查：

\[
(\text{rad/s})\cdot \text{s}=\text{rad}
\]

如果为了使用两子样本补偿，把这 10 个原始样本分成前半段和后半段，那么：

\[
\Delta\boldsymbol\theta_1
=
\sum_{k=1}^{5}\boldsymbol\omega_k\Delta t,
\qquad
\Delta\boldsymbol\theta_2
=
\sum_{k=6}^{10}\boldsymbol\omega_k\Delta t
\]

这里的 \(\Delta\boldsymbol\theta_1\)、\(\Delta\boldsymbol\theta_2\) 就叫 **陀螺子样本角增量**。它们不一定是 IMU 硬件直接输出的一条原始记录，也可以是算法把多个原始读数合并出来的小段增量。

加速度计同理。原始读数是比力 \(\mathbf f\)，单位 m/s²；小段积分后得到速度增量：

\[
\Delta\mathbf v
\approx
\mathbf f\,\Delta t
\]

单位检查：

\[
(\text{m/s}^2)\cdot \text{s}=\text{m/s}
\]

所以，“一个导航周期被分成两个子样本”的意思不是传感器只能采两次，而是：为了推导和补偿，我们把一个导航更新周期内的多次 IMU 原始采样按时间顺序合成若干段小增量，再研究这些小增量的先后顺序和叉乘项。

量纲检查很重要：

- \(\Delta\boldsymbol\theta_1\times\Delta\boldsymbol\theta_2\) 的单位仍是 rad，常把 rad 视为无量纲角度；
- \(\Delta\boldsymbol\theta\times\Delta\mathbf v\) 的单位是 m/s，正好可以作为速度增量修正；
- 修正项都是二阶小量，所以采样周期变短、角振动和线振动变弱时会快速变小。

## 3. 圆锥效应：为什么角增量相加会错

### 3.1 直觉图像

想象你手里拿着 IMU，让它的转轴尖端绕一个小圆锥面运动。每一瞬间的角速度可能都在 \(x-y\) 平面内，也就是 \(\omega_z=0\)。但姿态更新不是普通向量加法，而是三维旋转的连续乘积。

一个小实验可以暴露问题：先绕 \(x\) 轴转一点，再绕 \(y\) 轴转一点；反过来先绕 \(y\) 再绕 \(x\)，最终姿态并不完全一样。这个差异的主方向就是 \(x\times y=z\)。

类比到动态图：红色和蓝色的两个角增量虽然看起来都很小，但它们的叉乘沿第三个轴，长期累积后表现得像一个慢慢长出来的姿态漂移。

### 3.2 最小二维到三维例子

令

\[
\Delta\boldsymbol\theta_1=[a,0,0]^T,\quad
\Delta\boldsymbol\theta_2=[0,b,0]^T .
\]

如果只做向量相加，得到

\[
\Delta\boldsymbol\theta_{\text{sum}}=[a,b,0]^T .
\]

但是有限转动的乘积包含二阶交换误差，主导项方向为

\[
\Delta\boldsymbol\theta_1\times\Delta\boldsymbol\theta_2=[0,0,ab]^T .
\]

这说明即使两个输入都没有 \(z\) 分量，离散姿态更新也可能漏掉 \(z\) 向修正。

### 3.3 两子样本补偿

在等时间间隔、周期内角速度近似线性变化的假设下，一个常用两子样本圆锥补偿为：

\[
\boxed{
\Delta\boldsymbol\theta_{\text{corr}}
=\Delta\boldsymbol\theta_1+\Delta\boldsymbol\theta_2
+\frac{2}{3}
\left(
\Delta\boldsymbol\theta_1\times\Delta\boldsymbol\theta_2
\right)
}
\]

\(2/3\) 不是任意经验系数，它来自对子样本内线性角速度变化的积分近似。换一种采样结构或更高阶运动模型，系数和组合方式都可能改变。

### 3.4 工程解释

圆锥补偿的处理层是姿态机械编排。它应该在陀螺 bias、比例因子、安装误差等标定或预处理之后进行，然后再把补偿后的 \(\Delta\boldsymbol\theta_{\text{corr}}\) 用于四元数或方向余弦矩阵更新。

如果把圆锥误差误认为 bias，滤波器可能在某些机动下“估计出”一个与运动模式相关的假 bias。这在真实车辆、机器人或手持设备的高频振动场景中很危险。

## 4. 划桨效应：为什么角振动会造成速度误差

### 4.1 直觉图像

划桨效应可以理解为：桨叶一边改变方向，一边推动水。如果只统计推力大小而忽略方向变化，就会错估净位移。

IMU 中的对应关系是：

- 陀螺告诉你 body 系在周期内如何转；
- 加速度计给出 body 系中的 \(\Delta\mathbf v\)；
- 但 body 系方向在周期内变了，所以不同子样本的 \(\Delta\mathbf v_i\) 不能被当作同一方向的量直接相加。

类比的边界是：真实划桨涉及流体动力学，而这里讨论的是坐标旋转与离散积分，不是流体模型。

### 4.2 最小例子

令第一个半周期主要有 \(x\) 轴角增量，第二个半周期主要有 \(z\) 向速度增量：

\[
\Delta\boldsymbol\theta_1=[a,0,0]^T,\quad
\Delta\mathbf v_2=[0,0,c]^T .
\]

那么

\[
\Delta\boldsymbol\theta_1\times\Delta\mathbf v_2
=[0,-ac,0]^T .
\]

它的单位是 m/s，方向在 \(y\) 轴。这意味着一个看似只有 \(x\) 轴转动和 \(z\) 向线振动的组合，会产生横向速度修正。

### 4.3 两子样本划桨项

常见两子样本形式中的划桨相关项为：

\[
\boxed{
\Delta\mathbf v_{\text{scul}}
=\frac{2}{3}
\left(
\Delta\boldsymbol\theta_1\times\Delta\mathbf v_2
+
\Delta\mathbf v_1\times\Delta\boldsymbol\theta_2
\right)
}
\]

这个式子强调了两个事实：

1. 角增量和速度增量的相位关系会决定符号；
2. 只看 \(\Delta\boldsymbol\theta\) 或只看 \(\Delta\mathbf v\) 都不足以判断误差。

### 4.4 代码映射

本仓库中的共享函数：

```text
visual_explanations/imu_visualization_math.py
```

包含：

```python
sculling_rotation_correct(dtheta1, dvel1, dtheta2, dvel2)
```

其中 `sculling` 返回的就是上面的两子样本划桨项。动态图中的紫色箭头用于显示 \(\Delta\boldsymbol\theta\times\Delta\mathbf v\) 这类横向修正的方向。

## 5. 旋转效应：为什么总速度增量也要修正方向

### 5.1 直觉图像

设机器人在一个导航更新周期内持续左转，同时车体系 \(x\) 轴方向有恒定比力。周期开始时的“前方”和周期结束时的“前方”不是同一个导航系方向。

如果直接用旧姿态 \(\mathbf C_{b,\text{old}}^n\) 旋转整个 \(\Delta\mathbf v\)，就等于假设整个周期内 body 系没有转。这会低估横向速度分量。

### 5.2 一阶平均方向修正

在一个周期内总角增量为 \(\Delta\boldsymbol\theta\)、总速度增量为 \(\Delta\mathbf v\) 时，一阶旋转效应修正常写成：

\[
\boxed{
\Delta\mathbf v^n
\approx
\mathbf C_{b,\text{old}}^n
\left[
\Delta\mathbf v
+\frac{1}{2}
\left(
\Delta\boldsymbol\theta\times\Delta\mathbf v
\right)
\right]
}
\]

\(1/2\) 来自“周期内方向大约从旧方向线性转到新方向，所以平均方向在中间”的一阶积分直觉。角速度越大、周期越长、比力越大，这个项越明显。

### 5.3 与划桨效应的区别

旋转效应和划桨效应都包含角增量与速度增量的叉乘，但物理来源不同：

- 旋转效应：总角运动导致总 \(\Delta\mathbf v\) 的平均方向变化；
- 划桨效应：子样本内角振动和线振动的相关性额外产生二阶项。

实际机械编排中二者都要考虑。不能因为公式都长得像 \(\Delta\boldsymbol\theta\times\Delta\mathbf v\)，就把它们当作同一个误差。

## 6. 验证应该能抓住什么错误

一个有效验证不能只证明代码能运行，而要能暴露符号、单位和处理顺序错误。建议检查：

- 圆锥实验：\(\omega_x\) 和 \(\omega_y\) 相差 90 度时，误差应主要出现在第三轴；
- 划桨实验：改变线振动相位后，横向速度修正应改变大小甚至符号；
- 旋转实验：把导航周期缩短一半，忽略旋转造成的单步速度误差应明显下降；
- 单位检查：角增量必须是 rad，不是 deg；
- 顺序检查：补偿后的 body 系 \(\Delta\mathbf v\) 仍要用旧姿态旋转到 navigation 系。

## 7. 真实工程边界

当前仓库中的动画和静态脚本属于 educational prototype，用来解释机制和验证趋势。它们还不是部署级惯导机械编排库。

主要边界包括：

- 两子样本公式依赖等间隔和周期内平滑变化假设；
- 高频强振动可能需要多子样本或更高阶补偿；
- 真实系统还需要处理时间戳抖动、饱和、量化、温度、安装误差、比例因子和 bias；
- 动图使用合成运动，不代表任何具体 IMU 的真实性能；
- 真机验证应使用可追溯日志、独立参考轨迹和一致的坐标/单位约定。

## 8. 面试自检问题

1. 为什么 \(\omega_z=0\) 的圆锥运动仍可能造成 z 向姿态误差？
2. 圆锥补偿为什么属于机械编排层，而不是 ESKF 的 bias 估计层？
3. 划桨效应中，为什么改变角振动和线振动的相位会改变横向速度误差？
4. 旋转效应里的 \(1/2\) 可以怎样从“周期内平均方向”直觉解释？
5. \(\Delta\boldsymbol\theta\times\Delta\mathbf v\) 的单位是什么？为什么它可以加到 \(\Delta\mathbf v\) 上？
6. 如果代码把 deg 当 rad 使用，动态图或数值测试会出现什么量级异常？
7. 为什么补偿后的 \(\Delta\mathbf v\) 不能直接当作 navigation 系速度增量？
8. 两子样本补偿在哪些真实机器人场景中可能不够？
