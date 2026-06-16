# ESKF 完整算法原理 —— 从理论到实现

> 本文档是 `prototype/eskf_mvp.py` 的**完整理论输入**。
> 覆盖了从 IMU 误差建模、双子样机械编排、误差状态动力学到 GNSS 更新的全部数学推导。
> 每一节都对应 MVP 代码中的一个或几个函数，可直接对照实现。

---

## 目录

1. [问题场景](#1-问题场景)
2. [双状态架构](#2-双状态架构)
3. [IMU 误差模型](#3-imu-误差模型)
4. [双子样 INS 机械编排](#4-双子样-ins-机械编排)
5. [误差状态动力学](#5-误差状态动力学)
6. [GNSS 松耦合更新](#6-gnss-松耦合更新)
7. [误差注入与重置](#7-误差注入与重置)
8. [完整 ESKF 循环](#8-完整-eskf-循环)
9. [噪声参数选取](#9-噪声参数选取)
10. [数值注意事项](#10-数值注意事项)

---

## 1. 问题场景

```
传感器配置:
  IMU  Leador-A15  200 Hz  角度增量 Δθ [rad] + 速度增量 Δv [m/s]
  GNSS RTK           1 Hz  大地坐标 (纬度/经度/椭球高) + 标准差

坐标系:
  IMU 系 (body): 前-右-下 (X-front, Y-right, Z-down)
  导航系 (nav):  北-东-地 (N-E-D)
  重力:          g_n = [0, 0, +9.78]^T  (向下为正)

目标:
  输出 200 Hz (或 100 Hz) 连续平滑的位置、速度、姿态估计
  GNSS 到达时进行修正, 同时在线估计 IMU 偏置
```

### ESKF 相比 EKF 的核心优势

```
EKF:  在完整状态 [p, v, q, b_a, b_g] 上线性化
      → 状态量很大 (p 几百米, v 十几 m/s)
      → 线性近似在大值上不准确

ESKF: 在误差 [δp, δv, δθ, δb_a, δb_g] 上线性化
      → 误差始终很小 (δp 十几 cm, δv 几 cm/s, δθ 零点几度)
      → 线性近似在小值上非常准确
      → 线性化点始终是 δx=0, 无需担心偏移
```

---

## 2. 双状态架构

### 2.0 本 MVP 冻结的约定

后续公式统一采用：

```text
C_bn: body → NED
q_bn: 与 C_bn 对应的 Hamilton 四元数 [w,x,y,z]
姿态增量: q_new = q_old ⊗ dq_body
误差定义: δx = x_true - x_nom
姿态误差: q_true = q_nom ⊗ δq
δtheta: body frame 中的右乘小角度误差
```

同时：

```text
omega_m = omega_true + b_g + n_g
f_m     = f_true     + b_a + n_a
```

`F`、`G` 均为连续时间矩阵。MVP 使用
`Phi = I + F*T` 和 `Qd = Qc*T`。忽略地球自转、运输角速度、科氏项、
曲率变化、杆臂、比例因子和安装误差。

### 2.1 标称状态 (16 DOF) — 用完整非线性方程传播

```
x_nom = [p, v, q, b_a, b_g]^T

p   : (3×1)  NED 位置 [m]                  — 初始化为 0 (局部原点)
v   : (3×1)  NED 速度 [m/s]                — 从真值初始化
q   : (4×1)  姿态四元数 body→nav, Hamilton  — 从真值欧拉角初始化
b_a : (3×1)  加速度计偏置 [m/s²]           — 初始化为 0
b_g : (3×1)  陀螺仪偏置 [rad/s]            — 初始化为 0
```

### 2.2 误差状态 (15 DOF) — 协方差在此维护

```
δx = [δp, δv, δθ, δb_a, δb_g]^T

δp   : (3×1)  位置误差 [m]
δv   : (3×1)  速度误差 [m/s]
δθ   : (3×1)  姿态误差 (小角度向量) [rad]   ← 关键: 3维而非4维!
δb_a : (3×1)  加速度计偏置误差 [m/s²]
δb_g : (3×1)  陀螺仪偏置误差 [rad/s]
```

**为什么姿态误差是 3 维？**

标称姿态用 4 维四元数表示（无奇点，但带约束 ‖q‖=1）。真实姿态与标称姿态的
差异是一个微小的旋转——任何小旋转都可以用 3 维轴角向量表示。
协方差 P 在 15 维误差空间上维护，避免了四元数的冗余度和约束问题。

```text
真实姿态 = 标称姿态 ⊗ body-frame 误差旋转

q_true = q_nom ⊗ δq

δq ≈ [1, δθ_x/2, δθ_y/2, δθ_z/2]^T    (小角度近似)
```

### 2.3 两套状态的协同

```
标称状态 ← 用完整的非线性 INS 机械编排推进 (可以累加大量)
误差状态 ← 始终在线性化框架内传播协方差 (始终是小量)
                  ↓
         GNSS 观测到达 → KF 更新计算出 δx
                  ↓
         将 δx 注入标称状态 → δx 重置为 0 → P 保留
```

---

## 3. IMU 误差模型

### 3.1 原始输出

Leador-A15 输出的是**增量** (积分量), 不是瞬时值:

```
Δθ̃ = ∫ ω̃ dt     角度增量测量值 [rad]
Δṽ  = ∫ ã dt     速度增量测量值 [m/s]
```

### 3.2 误差组成

```
Δθ̃ = Δθ_true                        ← 真实角增量
    + b_g · dt                       ← 陀螺仪零偏 (缓慢漂移)
    + S_g · Δθ_true                  ← 陀螺仪比例因子 (灵敏度偏差)
    + M_g · Δθ_true                  ← 交轴耦合 (轴不垂直)
    + n_g · √dt                      ← 角度随机游走 (高频白噪声的积分)

Δṽ  = Δv_true                        ← 真实速度增量
    + b_a · dt                       ← 加速度计零偏
    + S_a · Δv_true                  ← 加速度计比例因子
    + M_a · Δv_true                  ← 交轴耦合
    + n_a · √dt                      ← 速度随机游走
```

### 3.3 MVP 的简化

MVP 阶段不考虑比例因子和交轴耦合（S=M=0），仅保留零偏和白噪声:

```
Δθ̃ = Δθ_true + b_g·dt + n_g·√dt
Δṽ  = Δv_true + b_a·dt + n_a·√dt
```

因此机械编排不能直接使用原始增量。对于每个长度为 `T/2` 的子样：

```text
Δtheta_i^c = Δtheta_i^m - b_g * (T/2)
Δv_i^c     = Δv_i^m     - b_a * (T/2)
```

圆锥、划桨和旋转补偿都使用去偏后的增量。

### 3.4 零偏的时间演化

零偏本身随时间缓慢变化。MVP 将其近似为**随机游走**：

```text
ḃ_a(t) = n_ba(t)      加速度计偏置的漂移由白噪声 n_ba 驱动
ḃ_g(t) = n_bg(t)      陀螺仪偏置的漂移由白噪声 n_bg 驱动

离散形式:
b_a(k+1) = b_a(k) + n_ba · √dt
b_g(k+1) = b_g(k) + n_bg · √dt
```

这里 \(n_{ba},n_{bg}\) 是白色驱动噪声，但 \(b_a,b_g\) 本身不是白噪声。
一阶 Gauss-Markov 模型还包含 \(-b/\tau_c\) 均值回归项，具有有限相关时间；
随机游走没有该项，长期方差无界。二者不能只用“\(\tau\to\infty\)”无条件
等同，具体极限还取决于驱动噪声强度如何参数化。

### 3.5 完整的 IMU 噪声向量 (12 维)

```
n_imu = [n_a(3), n_g(3), n_ba(3), n_bg(3)]^T

n_a   : 加速度计白噪声          [m/s²/√Hz]
n_g   : 陀螺仪白噪声             [rad/s/√Hz]
n_ba  : 加速度计偏置随机游走驱动  [m/s²/√Hz]
n_bg  : 陀螺仪偏置随机游走驱动    [rad/s/√Hz]
```

| 噪声类型 | 符号 | Allan 方差斜率 | 在 Q_d 中的方差 | 物理来源 |
|---------|------|:--:|------|------|
| 速度/角度随机游走 | n_a, n_g | −1/2 | 连续 PSD 在 `Qc` 中为 σ² | 白噪声积分 |
| 偏置随机游走 | n_ba, n_bg | +1/2 | σ²·I·dt | 偏置的缓慢漂移 |

---

## 4. 双子样 INS 机械编排

### 4.1 为什么需要双子样

在一个导航周期 T = 10ms 内, 载体可能在做显著的角运动。
如果简单地把两个子样的增量相加, 忽略角运动导致的方向变化,
会产生三种不可交换误差:

| 误差类型 | 物理图像 | 后果 |
|---------|------|------|
| **圆锥 (Coning)** | 绕 X 振 + 绕 Y 振 → 绕 Z 的净旋转 | 航向虚假漂移 |
| **划桨 (Sculling)** | 角振动 × 线振动 → 虚假的常值速度 | 位置积分后漂移 |
| **旋转 (Rotation)** | dt 内 body 系方向在变, 速度增量方向也在变 | 速度向量方向偏差 |

**数学本质**:

```
真实的连续旋转合成 ≠ 离散角增量向量的简单相加

真实: 先绕 a 转 |a|, 再绕 b 转 |b|  →  合成旋转 = a + b + ½(a×b) + 高阶
简单: 直接加                          →  合成旋转 = a + b

差值: ½(a×b) ← 这就是需要补偿的叉积项
```

### 4.2 数据结构

```
一个导航历元 T = 10ms, dt = T/2 = 5ms (每个子样)

子样 1:  [t, t+dt]  内的 Δθ₁, Δv₁
子样 2:  [t+dt, t+T] 内的 Δθ₂, Δv₂

总导航周期的时间步长: T = 2·dt = 0.01s
```

### 4.3 完整机械编排方程

输入: `p, v, q` (上一历元), `Δθ₁, Δv₁, Δθ₂, Δv₂, T`

```
步骤 ① — 每个子样做 bias 补偿:
    Δθ_i ← Δθ_i - b_g·T/2
    Δv_i ← Δv_i - b_a·T/2

步骤 ② — 总增量:
    Δθ = Δθ₁ + Δθ₂
    Δv = Δv₁ + Δv₂

步骤 ③ — 圆锥补偿 (加到总角度增量):
    Δθ += (2/3)·(Δθ₁ × Δθ₂)
          ↑
          两子样的叉积, 系数 2/3 源于线性角速度假设

步骤 ④ — 划桨补偿 (加到总速度增量):
    Δv_scul = (2/3)·(Δθ₁ × Δv₂ + Δv₁ × Δθ₂)
              ↑                    ↑
        角×线交叉              线×角交叉

步骤 ⑤ — 旋转效应补偿 (加到总速度增量):
    Δv_rot = (1/2)·(Δθ × Δv)
             ↑
        总公转 × 总速度增量, 旋转中的方向变化

步骤 ⑥ — 总速度增量:
    Δv = Δv + Δv_scul + Δv_rot

步骤 ⑦ — 姿态更新 (用补偿后的 Δθ, 右乘):
    angle = ‖Δθ‖
    if angle > 1e-15:
        axis = Δθ / angle
        δq = [cos(angle/2), axis·sin(angle/2)]^T
    else:
        δq = [1, 0, 0, 0]^T

    q_new = q_old ⊗ δq         ← Hamilton 乘法
    q_new /= ‖q_new‖           ← 归一化防止数值漂移

步骤 ⑧ — 速度更新 (用旧姿态 + 补偿后的 Δv):
    C_b^n = quat_to_dcm(q_old)
    Δv_n = C_b^n @ Δv           ← body → nav 旋转
    v_new = v_old + Δv_n + g_n · T
            ↑                  ↑
      比力积分贡献         重力贡献

步骤 ⑨ — 位置更新 (梯形积分):
    p_new = p_old + (v_old + v_new) · T / 2
```

### 4.4 验证机械编排正确性的方法

用真值做交叉检查:

```
输入: 真值姿态初始化 q0 = euler_to_quat(truth.roll, truth.pitch, truth.yaw)
      真值速度初始化 v0 = truth.v_ned
      参考点 = truth 初始位置

执行 1 步机械编排 (T=0.01s, 2 个子样):
      p1, v1, q1 = ins_mechanize_2sample(p0, v0, q0, IMU[0], IMU[1])

对比真值:
      真值在 T=0.01s 后的位置 ≈ p0 + v0·T (近似)
      真值姿态 ≈ truth 第一帧的姿态 (10ms 内变化极小)

验证:
      ‖p1‖ 应很小 (静止时 < 1mm)
      ‖v1 - v0‖ 也应在噪声水平
      q1 ≈ q0
```

---

## 5. 误差状态动力学

### 5.1 连续时间微分方程

误差状态 δx 的每个分量如何随时间变化:

```
δṗ  = δv                                                     (5.1)
      位置误差的变化率 = 速度误差

δv_dot = -C_b^n · [f_b×] · δtheta                              (5.2a)
      姿态误差导致的速度误差: "如果姿态错了 δθ,
      比力 f_b 转到导航系时方向就偏了 → 错误的加速度"
      ┌               ┐
      │  0   −f_z  f_y │
[f×] =│ f_z   0   −f_x │    比力 f_b 的反对称矩阵
      │ −f_y  f_x   0  │
      └               ┘

    - C_b^n · δb_a                                            (5.2b)
      bias 估计偏小时，标称状态会多积分比力，因此 true-nom 误差为负

    - C_b^n · n_a                                             (5.2c)
      加速度计白噪声

δtheta_dot = -[omega_b×] δtheta - δb_g                         (5.3a)
      右乘 body-frame 误差随 body 角速度旋转，并受陀螺 bias 驱动

    - n_g                                                     (5.3b)
      陀螺仪白噪声

δḃ_a = n_ba                                                  (5.4)
      偏置自身由随机游走驱动

δḃ_g = n_bg                                                  (5.5)
```

### 5.2 写成矩阵形式: δẋ = F·δx + G·n_imu

**F 矩阵 (15×15) 的分块构造:**

```
         δp  δv    δθ    δb_a  δb_g
      ┌                            ┐
 δṗ   │ 0    I₃    0     0     0 │  行 0-2
      │                            │
 δv̇   │ 0    0   −C[f×]   −C     0 │  行 3-5
      │                            │
F =  δθ̇   │ 0    0   −[ω×]    0    −I │  行 6-8
      │                            │
 δḃ_a │ 0    0     0     0     0 │  行 9-11
      │                            │
 δḃ_g │ 0    0     0     0     0 │  行 12-14
      └                            ┘
      列0-2 列3-5 列6-8 列9-11 列12-14
```

**逐块构造代码:**

```python
C = quat_to_dcm(q)

f_cross = skew(specific_force_b)
omega_cross = skew(angular_rate_b)

F = np.zeros((15, 15))

F[0:3, 3:6]   = np.eye(3)         # δṗ = δv
F[3:6, 6:9]   = -C @ f_cross       # 姿态误差 → 速度误差
F[3:6, 9:12]  = -C                 # 加速度计 bias 误差
F[6:9, 6:9]   = -omega_cross       # body-frame 姿态误差旋转
F[6:9, 12:15] = -np.eye(3)         # 陀螺仪 bias 误差
# 其余全零
```

**G 矩阵 (15×12) 的分块构造:**

```
         n_a  n_g  n_ba n_bg
      ┌                      ┐
 δṗ   │ 0    0    0    0  │  行 0-2
      │                      │
 δv̇   │−C    0    0    0  │  行 3-5
      │                      │
G =  δθ̇   │ 0   −I    0    0  │  行 6-8
      │                      │
 δḃ_a │ 0    0    I₃   0  │  行 9-11
      │                      │
 δḃ_g │ 0    0    0    I₃ │  行 12-14
      └                      ┘
      列0-2 3-5  6-8  9-11
```

```python
G = np.zeros((15, 12))

G[3:6, 0:3]   = -C               # n_a → 速度误差
G[6:9, 3:6]   = -np.eye(3)       # n_g → body-frame 姿态误差
G[9:12, 6:9]  = np.eye(3)       # n_ba → 加速度计偏置
G[12:15, 9:12] = np.eye(3)       # n_bg → 陀螺仪偏置
```

### 5.3 离散化: F → Φ

```
一阶近似 (MVP):
    Φ = I_15 + F · T            ← T = 0.01s, 一阶项占主导

精确版 (后续):
    Φ = exp(F·T) = I + F·T + (F·T)²/2! + (F·T)³/3! + ...
    对本项目 0.01s 的短步长，高阶项通常较小
```

### 5.4 离散过程噪声 Q_d (12×12)

```
连续时间功率谱密度 Q_c (12×12):
    Q_c = diag(
        σ_a² · I₃,     ← 加速度计白噪声 PSD [m²/s⁴/Hz]
        σ_g² · I₃,     ← 陀螺仪白噪声 PSD     [rad²/s²/Hz]
        σ_ba² · I₃,    ← 加速度计偏置随机游走 PSD
        σ_bg² · I₃     ← 陀螺仪偏置随机游走 PSD
    )

一阶近似离散化:
    Q_d = Q_c · T      ← T = 0.01s
```

```python
Qc_diag = np.concatenate([
    [SIGMA_ACC**2] * 3,     # n_a: σ_a² * I₃
    [SIGMA_GYRO**2] * 3,    # n_g: σ_g² * I₃
    [SIGMA_BA**2] * 3,      # n_ba: σ_ba² * I₃
    [SIGMA_BG**2] * 3,      # n_bg: σ_bg² * I₃
])
Qc = np.diag(Qc_diag)       # (12, 12)
Qd = Qc * T                  # 一阶离散化
```

### 5.5 协方差传播

```
P_pred = Φ @ P @ Φ^T + G @ Q_d @ G^T
          ↑              ↑
     确定性传播      噪声注入 (增大不确定性)
```

---

## 6. GNSS 松耦合更新

### 6.1 观测方程

GNSS RTK 提供 WGS84 坐标。先转换到 NED 局部坐标系:

```
z_gnss_ned = lla_to_ned(rtk_lat, rtk_lon, rtk_alt, ref_lla)

观测方程 (在误差状态空间):
    z_gnss_ned = p_nom + δp + n_gnss        ← 观测 = 标称位置 + 位置误差 + 噪声

    残差 = z_gnss_ned - p_nom = δp + n_gnss  ← GNSS 给出"位置误差"的直接观测
```

### 6.2 观测矩阵 H (3×15)

```
H = [I₃, 0₃, 0₃, 0₃, 0₃]
     ↑
   只观测位置误差分量, 速度/姿态/偏置的误差通过 P 矩阵的交叉项间接约束
```

```python
H = np.zeros((3, 15))
H[0:3, 0:3] = np.eye(3)
```

### 6.3 观测噪声 R_gnss (3×3)

从 RTK 标准差构造:

```
R_gnss = diag(σ_N², σ_E², σ_D²)  = diag(0.010², 0.009², 0.019²)
                                      ≈ diag(1e-4, 8.1e-5, 3.6e-4) [m²]
```

### 6.4 卡尔曼更新

```
残差:     y = z_gnss - p_nom                     (3×1)

新息协方差: S = H @ P @ H^T + R_gnss              (3×3)

卡尔曼增益: K = P @ H^T @ S^{-1}                  (15×3)
                ↑
              用 solve 而非 inv, 但 S 只有 3×3, inv 也安全

误差修正:   δx = K @ y                             (15×1)
                ↑
              这 15 个数就是 "每种误差该修正多少"

协方差更新: P = A @ P @ A.T + K @ R_GNSS @ K.T      (15×15) Joseph 形式
```

```python
residual = z_gnss - state["p"]                     # (3,1)

S = H @ P @ H.T + R_GNSS                           # (3,3)
B = P @ H.T
K = np.linalg.solve(S.T, B.T).T                    # (15,3)

dx = K @ residual                                   # (15,1)

I15 = np.eye(15)
A = I15 - K @ H
P = A @ P @ A.T + K @ R_GNSS @ K.T                  # Joseph 形式
```

---

## 7. 误差注入与重置

### 7.1 注入 (Inject)

将卡尔曼更新算出的误差修正加到标称状态上:

```
从 δx (15×1) 提取各分量:
    δp   = δx[0:3]      位置修正
    δv   = δx[3:6]      速度修正
    δθ   = δx[6:9]      姿态修正 (小角度向量)
    δb_a = δx[9:12]     加速度计偏置修正
    δb_g = δx[12:15]    陀螺仪偏置修正

位置/速度/偏置 — 直接加:
    p_nom += δp
    v_nom += δv
    b_a   += δb_a
    b_g   += δb_g

姿态 — 用四元数乘法 (不是加法!):
    angle = ‖δθ‖
    if angle > 1e-15:
        axis = δθ / angle
        δq = axis_angle_to_quat(axis, angle)
    else:
        δq = [1,0,0,0]^T
    q_nom = q_nom ⊗ δq           ← 与 q_true = q_nom ⊗ δq 一致
    q_nom /= ‖q_nom‖              ← 归一化
```

### 7.2 重置 (Reset)

注入后, 误差状态的定义变为 "当前标称状态与真实状态之差",
而卡尔曼更新已将此差估计为零 → 误差状态归零:

```
δx = 0_15

P 的微小修正 (理论上是 J·P·J^T, 但小角度下 J ≈ I):
    P ← P    (教学版保持不变)
```

**为什么必须重置？** 不重置的话, 下一次 GNSS 到达时的残差累积了
上次的 δx, 导致 KF 重复修正同一个误差。

---

## 8. 完整 ESKF 循环

```python
# ═══ 初始化 ═══════════════════════════════════════════════════

# 从真值初始化
q0 = euler_to_quat(truth.roll, truth.pitch, truth.yaw)
v0 = np.array([[truth.vn], [truth.ve], [truth.vd]])
p0 = np.zeros((3, 1))            # 当前位置 = NED 原点
ref_lla = (truth.lat, truth.lon, truth.alt)

# 偏置初始化为 0 (无先验)
b_a = np.zeros((3, 1))
b_g = np.zeros((3, 1))

# 协方差初始对角
P0_diag = [
    1.0, 1.0, 1.0,              # δp: 1m 不确定度
    0.1, 0.1, 0.1,              # δv: 0.3m/s
    1e-2, 1e-2, 1e-2,           # δθ: 0.1rad ≈ 5.7°
    1e-4, 1e-4, 1e-4,           # δb_a
    1e-6, 1e-6, 1e-6,           # δb_g
]
P = np.diag(P0_diag)

# ═══ 主循环 (100 Hz 导航) ════════════════════════════════════

for each 导航历元:
    # 取出 2 个 IMU 子样 (200 Hz → 100 Hz)
    dθ₁, dv₁ = IMU[2*k]
    dθ₂, dv₂ = IMU[2*k+1]

    # ── 预测步 ──
    q_old = q.copy()

    # ① 机械编排：函数内部先对每个子样做 bias 补偿
    p, v, q = ins_mechanize_2sample(
        p, v, q, b_a, b_g, dθ₁, dv₁, dθ₂, dv₂, T=0.01
    )

    # ② 构建 F, G, Q_d
    C = quat_to_dcm(q_old)
    omega_b = (dθ₁ + dθ₂ - b_g.ravel()*T) / T
    f_b = (dv₁ + dv₂ - b_a.ravel()*T) / T
    F = build_F(C, f_b, omega_b)
    G = build_G(C)
    Qd = build_Qd(T=0.01)

    # ③ 离散化 + 传播
    Phi = np.eye(15) + F * T
    P = Phi @ P @ Phi.T + G @ Qd @ G.T

    # ── GNSS 更新 (仅在有观测时) ──
    if gnss_available:
        z_gnss_ned = lla_to_ned_simple(rtk_lat, rtk_lon, rtk_alt, ref_lla)

        residual = z_gnss_ned - p

        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)

        S = H @ P @ H.T + R_GNSS
        B = P @ H.T
        K = np.linalg.solve(S.T, B.T).T
        dx = K @ residual
        A = np.eye(15) - K @ H
        P = A @ P @ A.T + K @ R_GNSS @ K.T
        P = 0.5 * (P + P.T)

        # 注入
        p += dx[0:3]
        v += dx[3:6]
        b_a += dx[9:12]
        b_g += dx[12:15]

        # 姿态注入
        dtheta = dx[6:9]
        angle = np.linalg.norm(dtheta)
        if angle > 1e-15:
            dq = axis_angle_to_quat(dtheta.ravel() / angle, angle)
            q = quat_multiply(q, dq)
            q /= np.linalg.norm(q)

        # 重置: δx = 0 (P 不变)

    # 记录状态用于评估
```

---

## 9. 噪声参数选取

### 9.1 Leador-A15 典型参数

| 参数 | 符号 | 典型值 | 在连续时间 Q_c 中 |
|------|------|------|------|
| 加速度计 VRW | σ_a | 0.01 m/s²/√Hz | `σ_a² I₃` |
| 陀螺仪 ARW | σ_g | 0.001 rad/s/√Hz | `σ_g² I₃` |
| 加速度计偏置 RW | σ_ba | 1e-4 m/s²/√Hz | `σ_ba² I₃` |
| 陀螺仪偏置 RW | σ_bg | 1e-5 rad/s/√Hz | `σ_bg² I₃` |

一阶离散化统一使用 `Q_d = Q_c * T`。

### 9.2 调参指南

```
现象: 滤波过于平滑, 转弯时跟不上
  → P 太小或 Q_d 太小 → 增大 Q_d (滤波器更信任观测)

现象: 滤波太噪, GNSS 噪声直接映射到输出
  → R 太小 → 增大 R_GNSS (更信任模型预测)

现象: 偏置估计不收敛
  → σ_ba/σ_bg 太小 → 增大偏置随机游走噪声
  → 或者 P0 中对偏置的初始不确定度太小

现象: P 快速趋零 (过度自信)
  → Q_d 太小, 过程噪声注入不足 → 增大 Q_d
```

---

## 10. 数值注意事项

### 10.1 四元数归一化

每次四元数乘法后必须归一化:
```python
q = q / np.linalg.norm(q)
```
不归一化 → ‖q‖ 缓慢偏离 1 → DCM 不再正交 → 姿态漂移。

### 10.2 反对称矩阵的负号

```
[f_b×] 的构造: 右上三角为 -f_z, +f_y
F[3:6,6:9] = -C @ [f_b×]
F[3:6,9:12] = -C
F[6:9,6:9] = -[omega_b×]
F[6:9,12:15] = -I
```

### 10.3 姿态更新用旧姿态

```
速度更新时: 用旧姿态的 C (此时增量发生时的坐标系)
位置更新时: 用梯形法则 (新旧速度的平均)
```

### 10.4 协方差对称性

Joseph 形式 `A P A^T + K R K^T` 理论上保持半正定性, 但长期运行可能有
浮点漂移。如果需要, 可以在每步后强制:
```python
P = (P + P.T) / 2
```

### 10.5 坐标系一致性

```
GNSS 经过 lla_to_ned 后与 INS 的 p 在同一个 NED 系
→ 可以直接相减
→ 绝不能用 (lat,lon,alt) 和 (N,E,D) 混在一起做算术!
```

---

## 附录: 关键函数签名速查

```
euler_to_quat(roll, pitch, yaw) → q (4,1)
quat_to_dcm(q) → C (3,3)
quat_multiply(q1, q2) → q (4,1)
axis_angle_to_quat(axis(3,), angle) → q (4,1)

ins_mechanize_2sample(p,v,q, dθ1,dv1, dθ2,dv2, T=0.01) → p,v,q

build_F(C, specific_force_b, angular_rate_b) → F (15,15)
build_G(C) → G (15,12)
build_Qd(T, σ_a, σ_g, σ_ba, σ_bg) → Qd (12,12)

lla_to_ned_simple(lat,lon,alt, ref_lat,ref_lon,ref_alt) → ned (3,1)
```
