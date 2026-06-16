# 1D Kalman Filter：位置与速度观测融合

## 问题定义

本节实现一个最小但完整的 1D Kalman Filter，用来估计沿直线运动目标的状态：

```text
x = [position, velocity]^T
```

传感器同时给出带噪的位置和速度观测：

```text
z = [z_position, z_velocity]^T
```

这个例子对应 GNSS/机器人定位中的一个简化问题：GNSS 或其他外部传感器提供位置，轮速计、多普勒速度或其他模块提供速度，滤波器根据运动模型和观测噪声进行加权融合。

## 状态变量和维度

状态向量：

```text
x = [p, v]^T, dim(x) = 2 x 1
```

协方差和模型矩阵：

```text
P: 2 x 2, 状态估计误差协方差
F: 2 x 2, 状态转移矩阵
H: 2 x 2, 观测矩阵
Q: 2 x 2, 过程噪声协方差
R: 2 x 2, 观测噪声协方差
K: 2 x 2, Kalman gain
```

## 数学公式

采用匀速运动模型：

```text
p_k = p_{k-1} + v_{k-1} dt
v_k = v_{k-1}
```

矩阵形式：

```text
x_k = F x_{k-1}

F = [1 dt
     0  1]
```

预测：

```text
x_pred = F x
P_pred = F P F^T + Q
```

更新：

```text
residual = z - H x_pred
S = H P_pred H^T + R
K = P_pred H^T S^{-1}
x_upd = x_pred + K residual
P_upd = (I - K H) P_pred       （简化协方差更新）
```

代码实现使用数值上更稳健的 Joseph 形式
``A P_pred Aᵀ + K R Kᵀ``，其中 ``A = I - K H``。使用精确最优增益时，
两种形式在代数上等价。

代码中为了数值稳定性，`K` 使用 `np.linalg.solve` 计算，而不是显式构造 `S^{-1}`。

## 推导轮廓

1. 用匀速模型预测下一时刻的位置和速度。
2. 用 `Q` 表示真实运动和匀速模型之间的不匹配。
3. 用 `H` 把状态映射到观测空间，本例中位置和速度都直接可观测，所以 `H = I`。
4. 用 residual 表示观测和预测观测之间的差。
5. 用 `S` 表示 residual 的不确定性。
6. 用 `K` 自动决定本次更新更相信模型还是观测。
7. 更新 `x` 和 `P`。

## 实现细节

核心代码在：

```text
python/gnss_imu/kalman_filter_1d.py
```

示例脚本在：

```text
python/examples/demo_1d_kalman_filter.py
```

测试在：

```text
tests/python/test_kalman_filter_1d.py
```

本例的默认观测为：

```text
z = [1.2, 0.9]^T
```

一次预测更新后的状态约为：

```text
x_upd = [1.02387807, 0.96858594]^T
```

这说明位置观测 `1.2 m` 把预测位置轻微拉高，速度观测 `0.9 m/s` 把预测速度拉低。由于默认位置观测噪声 `R[0,0] = 4.0` 大于速度观测噪声 `R[1,1] = 1.0`，位置更新幅度更保守。

## 常见面试问题

- `P`、`Q`、`R` 的物理含义分别是什么？
- 为什么 `R` 越大，滤波器越不相信观测？
- 为什么工程实现中常用 `solve` 代替 `inv`？
- residual 和 innovation covariance `S` 分别表示什么？
- 如果真实目标有加速度，但模型是假设匀速，应该调大 `Q` 还是 `R`？
- `P` 是否一定会越来越小？什么情况下会变大？

## 常见错误

- 把状态向量写成 `(2,)`，导致矩阵乘法维度不清晰。
- 把 `Q` 理解成观测权重；实际上它表示过程噪声。
- 把 `R` 理解成观测值本身；实际上它表示观测噪声协方差。
- 忘记 `P_pred = F P F^T + Q` 中的转置。
- 直接复制公式时写出错误的 `solve(A, b)` 参数。
- 在测试中只检查最终数值，不检查矩阵维度。
