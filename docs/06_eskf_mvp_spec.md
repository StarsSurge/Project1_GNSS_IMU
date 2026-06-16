# ESKF MVP 接口约定（双子样精密机械编排）

> **完整算法原理见**: [07_eskf_algorithm_complete.zh-CN.md](07_eskf_algorithm_complete.zh-CN.md)
>
> 单文件原型 `prototype/eskf_mvp.py`，约 400 行，仅依赖 numpy。
> 导航周期 100 Hz（dt=0.01s），每个导航历元使用 **2 个 IMU 子样** (200 Hz 原始数据)。
> 包含完整的圆锥/划桨/旋转效应补偿。

## 零、冻结的数学约定

实现前不再混用下列约定：

1. `C_bn` 将 body 向量旋转到 NED：`v_n = C_bn @ v_b`。
2. 四元数采用 Hamilton 顺序 `[w, x, y, z]`，`q_bn` 对应 `C_bn`。
3. 姿态增量右乘：`q_new = q_old ⊗ dq_body`。
4. 误差定义为真实值减标称值：
   `δp = p_true - p_nom`，其余加性状态相同。
5. 姿态采用右乘、body-frame 误差：
   `q_true = q_nom ⊗ δq`，`δq ≈ [1, δθ/2]`。
6. IMU bias 定义为测量值中的加性误差：
   `ω_m = ω_true + b_g + n_g`，`f_m = f_true + b_a + n_a`。
7. `F`、`G` 是连续时间矩阵；`Phi = I + F * dt`，
   `Qd = Qc * dt`。
8. MVP 忽略地球自转、运输角速度、曲率变化、科氏项、杆臂和比例因子。

由这些约定直接得到：

```text
δp_dot     = δv
δv_dot     = -C_bn [f_b×] δθ - C_bn δb_a - C_bn n_a
δtheta_dot = -[omega_b×] δθ - δb_g - n_g
δb_a_dot   = n_ba
δb_g_dot   = n_bg
```

## 一、文件结构

```
prototype/
└── eskf_mvp.py          # 单文件，约 400 行，仅依赖 numpy
```

## 二、数据流

```
IMU 200 Hz 原始数据 (8 帧)
   │
   ├── 导航历元 0 ── sub1=IMU[0], sub2=IMU[1] ── dt=0.01s
   ├── 导航历元 1 ── sub1=IMU[2], sub2=IMU[3] ── dt=0.01s
   ├── 导航历元 2 ── sub1=IMU[4], sub2=IMU[5] ── dt=0.01s
   └── 导航历元 3 ── sub1=IMU[6], sub2=IMU[7] ── dt=0.01s
                                      │
GNSS t=456300.000 ──→ 初始化附近的近似更新（先于首帧 IMU 约 4.4 ms）
                                      │
                                      ▼
initialize()  ──→  GNSS update()  ──→  eskf_state
                                      │
  for each 导航历元 (2 subsamples):
     predict(sub1_dtheta, sub1_dvel,
             sub2_dtheta, sub2_dvel, dt)
         ├── ins_mechanize_2sample()  → 圆锥+划桨+旋转补偿 → 更新 p,v,q
         ├── build_F()                → 组装 15×15 F 矩阵
         ├── build_G()                → 组装 15×12 G 矩阵
         └── propagate_P()            → P_pred = Φ·P·Φᵀ + G·Qd·Gᵀ

     if GNSS epoch:
         update(z_gnss)
             ├── kalman_update() → K, δx, P_upd
             └── inject_reset()  → δx → 标称状态, δx = 0

     print_state()  ← 每个导航历元打印关键量
```

## 三、数据结构

### 标称状态 (packed in a dict or dataclass)

```python
state = {
    "p":  np.array([[N], [E], [D]]),       # (3,1)  NED 位置 [m]
    "v":  np.array([[VN], [VE], [VD]]),    # (3,1)  NED 速度 [m/s]
    "q":  np.array([[w], [x], [y], [z]]),  # (4,1)  姿态四元数 body→nav
    "b_a": np.zeros((3, 1)),               # (3,1)  加速度计偏置 [m/s²]
    "b_g": np.zeros((3, 1)),               # (3,1)  陀螺仪偏置 [rad/s]
    "P":  np.eye(15) * 1.0,                # (15,15) 误差协方差
}
```

### 常量

```python
GRAVITY = np.array([[0.0], [0.0], [9.78]])   # NED 重力 [m/s²]
R_GNSS  = np.diag([0.010**2, 0.009**2, 0.019**2])

# 初始协方差对角线 (顺序: δp, δv, δθ, δb_a, δb_g)
P0_diag = np.concatenate([
    [1.0]*3,      # 位置不确定度 1m²
    [0.1]*3,      # 速度不确定度 0.1 (m/s)²
    [0.01]*3,     # 姿态方差 0.01 rad²，即标准差约 0.1 rad
    [1e-4]*3,     # 加速度计偏置不确定度
    [1e-6]*3,     # 陀螺仪偏置不确定度
])

# IMU 噪声 (用于构造 Q_d)
SIGMA_ACC  = 0.01     # 加速度计白噪声 [m/s²/√Hz]
SIGMA_GYRO = 0.001    # 陀螺仪白噪声 [rad/s/√Hz]
SIGMA_BA   = 1e-4     # 加速度计偏置随机游走
SIGMA_BG   = 1e-5     # 陀螺仪偏置随机游走
```

## 四、函数签名与约定

### 4.1 初始化

```python
def initialize(lat0, lon0, alt0, v_ned0, roll0, pitch0, yaw0) -> dict:
    """
    用真值初始化标称状态。

    参数
    ----
    lat0, lon0, alt0 : float
        初始大地坐标 [deg, deg, m]
    v_ned0 : (3,) array
        初始 NED 速度 [VN, VE, VD] [m/s]
    roll0, pitch0, yaw0 : float
        初始欧拉角 [deg], ZYX 顺序

    返回
    ----
    state : dict  含 p, v, q, b_a, b_g, P, ref_lla
    """
```

内部步骤：
1. 保存 `ref_lla = (lat0, lon0, alt0)` 作为 NED 局部原点（后续 GNSS 转为 NED 时使用同一个参考点）
2. `p = np.zeros((3,1))`（初始点为 NED 原点）
3. `v = v_ned0.reshape(3,1)`
4. `q = euler_to_quat(roll0, pitch0, yaw0)`（需要你实现 `euler_to_quat`）
5. `b_a = b_g = zeros`
6. `P = diag(P0_diag)`

### 4.2 姿态转换

```python
def euler_to_quat(roll, pitch, yaw):
    """
    ZYX 欧拉角 [deg] → 四元数 [w, x, y, z] (Hamilton)

    旋转顺序: 先绕 Z 转 yaw → 再绕 Y 转 pitch → 再绕 X 转 roll
    数学: q = q_z(yaw) ⊗ q_y(pitch) ⊗ q_x(roll)
    """

def quat_to_dcm(q):
    """
    四元数 [w, x, y, z] → 方向余弦矩阵 C_b^n (3×3)
    body 系向量 = C_b^n @ nav 系向量? 不——
    v_nav = C_b^n @ v_body
    即 C_b^n 的第 i 行第 j 列 = nav_i 方向在 body_j 轴上的投影
    """

def quat_multiply(q1, q2):
    """四元数乘法 q1 ⊗ q2 (Hamilton)"""

def axis_angle_to_quat(axis, angle):
    """轴角 [3,] + 角度 [rad] → 四元数 (4,1)"""
```

### 4.3 INS 双子样机械编排

```python
def ins_mechanize_2sample(
    p, v, q, b_a, b_g,
    dtheta1, dvel1, dtheta2, dvel2, dt,
):
    """
    双子样机械编排 — 含圆锥/划桨/旋转效应补偿。

    每个导航历元的 dt 被分为两个子样窗口, 假设角速度在 dt 内线性变化,
    用叉积交叉项补偿不可交换误差。

    参数
    ----
    p : (3,1)  当前 NED 位置 [m]
    v : (3,1)  当前 NED 速度 [m/s]
    q : (4,1)  当前姿态四元数 (body→nav)
    dtheta1, dtheta2 : (3,)  子样 1, 2 的角度增量 [rad]
    dvel1,   dvel2   : (3,)  子样 1, 2 的速度增量 [m/s]
    b_a : (3,1)  当前加速度计 bias 估计 [m/s²]
    b_g : (3,1)  当前陀螺仪 bias 估计 [rad/s]
    dt : float  导航历元步长 (2×T_imu, = 0.01 s)

    返回
    ----
    p_new, v_new, q_new : 更新后的位置/速度/姿态
    """
```

内部步骤：
```python
dtheta1 = np.asarray(dtheta1).reshape(3,)
dtheta2 = np.asarray(dtheta2).reshape(3,)
dvel1   = np.asarray(dvel1).reshape(3,)
dvel2   = np.asarray(dvel2).reshape(3,)

# ── ① 每个子样先做 bias 补偿 ──
dt_sub = dt / 2.0
dtheta1 = dtheta1 - b_g.ravel() * dt_sub
dtheta2 = dtheta2 - b_g.ravel() * dt_sub
dvel1 = dvel1 - b_a.ravel() * dt_sub
dvel2 = dvel2 - b_a.ravel() * dt_sub

# ── ② 总增量 ──
dtheta = dtheta1 + dtheta2
dvel   = dvel1 + dvel2

# ── ③ 圆锥补偿 ──
#     "角振动 × 角振动 → 在第三轴产生净旋转"
coning = np.cross(dtheta1, dtheta2)         # (3,) 叉积
dtheta = dtheta + (2.0 / 3.0) * coning

# ── ④ 划桨补偿 ──
#     "角振动 × 线振动 + 线振动 × 角振动 → 虚假的常值速度"
sculling = (np.cross(dtheta1, dvel2) +
            np.cross(dvel1, dtheta2))
dvel_scul = (2.0 / 3.0) * sculling

# ── ⑤ 旋转效应补偿 ──
#     "dt 内 body 系旋转 → 速度增量的方向在变"
dvel_rot = 0.5 * np.cross(dtheta, dvel)

# ── ⑥ 总速度增量 (划桨 + 旋转均已补偿) ──
dvel = dvel + dvel_scul + dvel_rot

# ── ⑦ 姿态更新 (用补偿后的 dtheta) ──
angle = np.linalg.norm(dtheta)
if angle > 1e-15:
    axis = dtheta / angle
    dq = axis_angle_to_quat(axis, angle)
else:
    dq = np.array([[1.0],[0.0],[0.0],[0.0]])
q_new = quat_multiply(q, dq)
q_new = q_new / np.linalg.norm(q_new)

# ── ⑧ 速度更新 (用旧姿态 + 补偿后的 dvel) ──
C = quat_to_dcm(q)                           # 旧姿态
dvel_ned = C @ dvel.reshape(3, 1)            # body→nav
v_new = v + dvel_ned + GRAVITY * dt          # 加重力

# ── ⑨ 位置更新 (梯形积分) ──
p_new = p + (v + v_new) * dt / 2.0

return p_new, v_new, q_new
```

### 4.4 误差状态传播 (build F, G, P propagation)

```python
def build_F(C_bn, specific_force_b, angular_rate_b):
    """
    组装 15×15 系统矩阵 F。

    参数
    ----
    C_bn : (3,3)  当前姿态的方向余弦矩阵 C_b^n
    specific_force_b : (3,)  去偏后的平均比力 [m/s²]
    angular_rate_b : (3,)  去偏后的平均角速度 [rad/s]

    返回
    ----
    F : (15,15)  连续时间系统矩阵
    """

def build_G(C_bn):
    """
    组装 15×12 噪声驱动矩阵 G。

    返回
    ----
    G : (15,12)
    """
```

右乘 body-frame 误差约定下：

```python
G = np.zeros((15, 12))
G[3:6, 0:3] = -C_bn
G[6:9, 3:6] = -np.eye(3)
G[9:12, 6:9] = np.eye(3)
G[12:15, 9:12] = np.eye(3)
```

噪声符号不会改变单独的对角协方差贡献，但必须与误差微分方程保持一致，
否则以后加入相关噪声或交叉项时会出错。

```python
def build_Qd(dt):
    """
    组装 12×12 离散时间过程噪声。

    Qc = diag(σ_a² I₃, σ_g² I₃, σ_ba² I₃, σ_bg² I₃)
    Qd = Qc * dt

    这里的 σ 是连续时间噪声密度，G 是连续时间噪声驱动矩阵。
    """

def predict(state, dtheta1, dvel1, dtheta2, dvel2, dt):
    """
    完整的 IMU 预测步 (双子样):
    1. ins_mechanize_2sample() 更新标称状态 (含圆锥/划桨/旋转补偿)
    2. 用去偏后的总增量除以 dt，得到平均比力和角速度
    3. build_F() — 用平均比力、角速度构建连续时间 F
    4. build_G() build_Qd()
    5. Φ = I + F·dt
    6. P = Φ @ P @ Φᵀ + G @ Qd @ Gᵀ
    """
    # 标称状态传播
    p, v, q = state["p"], state["v"], state["q"]
    b_a, b_g = state["b_a"], state["b_g"]
    p_new, v_new, q_new = ins_mechanize_2sample(
        p, v, q, b_a, b_g,
        dtheta1, dvel1, dtheta2, dvel2, dt,
    )
    state["p"], state["v"], state["q"] = p_new, v_new, q_new

    # F 使用连续时间物理量，不能直接传速度/角度增量。
    dtheta_total = dtheta1 + dtheta2 - b_g.ravel() * dt
    dvel_total = dvel1 + dvel2 - b_a.ravel() * dt
    angular_rate_b = dtheta_total / dt
    specific_force_b = dvel_total / dt

    C = quat_to_dcm(q)                   # 传播前旧姿态
    F = build_F(C, specific_force_b, angular_rate_b)
    G = build_G(C)
    Qd = build_Qd(dt)

    # Φ = I + F·dt (一阶近似)
    I15 = np.eye(15)
    Phi = I15 + F * dt

    # 协方差传播
    P = state["P"]
    state["P"] = Phi @ P @ Phi.T + G @ Qd @ G.T
```

**F 矩阵的具体构造：**

```python
F = np.zeros((15, 15))
C = C_bn
fx, fy, fz = specific_force_b
wx, wy, wz = angular_rate_b

# 反对称矩阵 [f_b ×] 和 [omega_b ×]
f_cross = np.array([
    [0,    -fz,   fy],
    [fz,    0,   -fx],
    [-fy,   fx,    0],
])
omega_cross = np.array([
    [0,    -wz,   wy],
    [wz,    0,   -wx],
    [-wy,   wx,    0],
])

# 块 (0,1): δṗ = δv
F[0:3, 3:6] = np.eye(3)

# 块 (1,2): δv_dot = -C [f_b×] δtheta
F[3:6, 6:9] = -C @ f_cross

# 块 (1,3): δv_dot += -C · δb_a
F[3:6, 9:12] = -C

# 块 (2,2): 右乘 body-frame 姿态误差自身随角速度旋转
F[6:9, 6:9] = -omega_cross

# 块 (2,4): δtheta_dot += -δb_g
F[6:9, 12:15] = -np.eye(3)

# 其余块为零 (e.g. δb_a_dot = 0 由纯噪声驱动)
```

### 4.5 GNSS 更新 + 注入

```python
def update(state, z_gnss):
    """
    GNSS 松耦合更新步。

    参数
    ----
    state : dict  当前状态 (含 P)
    z_gnss : (3,1)  GNSS NED 位置观测 [m]

    返回
    ----
    state : dict  更新注入后的状态
    """
```

内部步骤：
```python
# 1. 观测矩阵 H = [I₃, 0₃, 0₃, 0₃, 0₃]  (3 × 15)
H = np.zeros((3, 15))
H[0:3, 0:3] = np.eye(3)

# 2. 残差
residual = z_gnss - state["p"]

# 3. 卡尔曼增益
P = state["P"]
S = H @ P @ H.T + R_GNSS
B = P @ H.T
K = np.linalg.solve(S.T, B.T).T    # (15×3)

# 4. 误差状态修正
dx = K @ residual                    # (15×1)

# 5. 协方差更新 (Joseph form)
I15 = np.eye(15)
A = I15 - K @ H
P = A @ P @ A.T + K @ R_GNSS @ K.T
P = 0.5 * (P + P.T)

# 6. 注入 (inject)
dp   = dx[0:3]        # 位置修正
dv   = dx[3:6]        # 速度修正
dth  = dx[6:9]        # 姿态修正 (小角度向量)
dba  = dx[9:12]       # 偏置修正
dbg  = dx[12:15]

state["p"] += dp
state["v"] += dv
state["b_a"] += dba
state["b_g"] += dbg

# 姿态注入与误差定义一致:
# q_true = q_nom ⊗ δq，因此 q_nom <- q_nom ⊗ δq
angle = np.linalg.norm(dth)
if angle > 1e-15:
    axis = (dth / angle).ravel()
    dq = axis_angle_to_quat(axis, angle)
    state["q"] = quat_multiply(state["q"], dq)
    state["q"] /= np.linalg.norm(state["q"])

# 7. 重置: 误差状态归零。
# MVP 暂用一阶近似 J_reset ≈ I，因此 P 不额外投影。
state["P"] = P
```

### 4.6 LLA → NED 的简化

MVP 用简化近似（不实现完整 WGS84，差几十米范围内误差可忽略）：

```python
def lla_to_ned_simple(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    简化版 LLA → NED。用参考点处的曲率半径近似。
    仅适用于局部范围 (< 10 km)。

    返回: NED (3,1) [m]
    """
    # WGS84
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2*f - f**2

    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    ref_lat_r = np.deg2rad(ref_lat)
    ref_lon_r = np.deg2rad(ref_lon)

    sin_ref = np.sin(ref_lat_r)
    Rn = a / np.sqrt(1 - e2 * sin_ref**2)
    Rm = a * (1 - e2) / (1 - e2 * sin_ref**2)**1.5

    dlat = lat_r - ref_lat_r
    dlon = lon_r - ref_lon_r
    dalt = alt - ref_alt

    dn = (Rm + dalt) * dlat
    de = (Rn + dalt) * np.cos(ref_lat_r) * dlon
    dd = -dalt

    return np.array([[dn], [de], [dd]])
```

## 五、硬编码测试数据

从真实数据集提取 IMU 在 t≈456300.0 附近的连续 8 帧 (200 Hz 原始数据)，
配对为 4 个导航历元 (100 Hz, dt=0.01s)。

```python
# ── IMU 数据: 8 帧, 200 Hz ──
# 格式: [dtheta_x, dtheta_y, dtheta_z, dvel_x, dvel_y, dvel_z]
# 来自 Leador-A15.txt, t ≈ 456300.004 ~ 456300.039

IMU_RAW = [
    # frame 0: t=456300.004412
    ( 0.0000043453, -0.0000018374, -0.0000012908,
     -0.0023051314, -0.0010445054, -0.0488077663),

    # frame 1: t=456300.009412
    ( 0.0000002888, -0.0000027108, -0.0000018521,
     -0.0021418147, -0.0012695923, -0.0491048507),

    # frame 2: t=456300.014412
    ( 0.0000007072,  0.0000007140, -0.0000007045,
     -0.0013359998, -0.0010260993, -0.0490441956),

    # frame 3: t=456300.019412
    ( 0.0000002223,  0.0000043659, -0.0000010201,
     -0.0018919073, -0.0017334900, -0.0490068980),

    # frame 4: t=456300.024412
    ( 0.0000004925, -0.0000013996,  0.0000005337,
     -0.0016432911, -0.0012742251, -0.0487665012),

    # frame 5: t=456300.029413
    (-0.0000009892,  0.0000002849,  0.0000019280,
     -0.0017654452, -0.0002076474, -0.0482605062),

    # frame 6: t=456300.034412
    (-0.0000004215,  0.0000020124,  0.0000004384,
     -0.0022247143, -0.0005494262, -0.0489067286),

    # frame 7: t=456300.039412
    (-0.0000036294,  0.0000028565,  0.0000008919,
     -0.0011102571, -0.0007864570, -0.0488879941),
]

# 配成 4 个双子样导航历元:
# epoch 0: sub1=frame[0], sub2=frame[1], dt=0.01
# epoch 1: sub1=frame[2], sub2=frame[3], dt=0.01
# epoch 2: sub1=frame[4], sub2=frame[5], dt=0.01
# epoch 3: sub1=frame[6], sub2=frame[7], dt=0.01

DT_NAV = 0.01   # 100 Hz 导航周期
```

```python
# ── GNSS 数据: t=456300.000 (1 Hz) ──
# 来自 GNSS-RTK.txt index 50
GNSS_DATA = {
    "lat": 30.4447858174,
    "lon": 114.4718660799,
    "alt": 21.1020,
    "std": (0.010, 0.009, 0.019),   # N, E, D [m]
}
```

```python
# ── 真值初始化: t=456300.004412 (truth.nav 第一行) ──
TRUTH_INIT = {
    "lat": 30.4447873710,
    "lon": 114.4718631927,
    "alt": 20.9040,
    "vn": 0.0003,  "ve": -0.0009,  "vd": -0.0009,
    "roll": 0.85266,  "pitch": -2.03401,  "yaw": 185.67273,
}
```

## 六、主循环伪代码

```python
def main():
    # ── 初始化 ──
    t = TRUTH_INIT
    state = initialize(t["lat"], t["lon"], t["alt"],
                       [t["vn"], t["ve"], t["vd"]],
                       t["roll"], t["pitch"], t["yaw"])

    print("=" * 64)
    print(" 初始状态 ")
    print_state(state)

    # 注意: GNSS 时间 456300.000 比首帧 IMU 早约 4.4 ms。
    # MVP smoke test 将它视为“初始化时刻附近”的近似观测，先更新再传播；
    # 这不是严格时间同步策略。
    g = GNSS_DATA
    z_gnss = lla_to_ned_simple(
        g["lat"], g["lon"], g["alt"], *state["ref_lla"],
    )
    update(state, z_gnss)
    print("\n--- 初始化附近 GNSS 更新后 ---")
    print_state(state)

    # ── 4 个双子样 IMU 导航历元 ──
    for epoch in range(4):
        i = epoch * 2  # IMU 原始帧索引

        dtheta1 = np.array(IMU_RAW[i][0:3])
        dvel1   = np.array(IMU_RAW[i][3:6])
        dtheta2 = np.array(IMU_RAW[i+1][0:3])
        dvel2   = np.array(IMU_RAW[i+1][3:6])

        predict(state, dtheta1, dvel1, dtheta2, dvel2, DT_NAV)
        print(f"\n--- Epoch {epoch}: 纯 IMU 预测后 ---")
        print_state(state)

    print("\n" + "=" * 64)
    print(" 验证检查 ")
    print("  [ ] GNSS 更新后 trace(P) 小于更新前")
    print("  [ ] 后续纯 IMU 传播时 trace(P) 逐步增大")
    print("  [ ] GNSS NED 约为 [-0.172, 0.277, -0.198] m")
    print("  [ ] ‖q‖ ≈ 1.0")
    print("  [ ] K 矩阵前 3 行有显著值 (GNSS 主要修正位置)")
    print("  [ ] 圆锥补偿 (2/3)·(dθ1×dθ2) 非零")
    print("=" * 64)
```

## 七、打印格式

```python
def print_state(state):
    p = state["p"].ravel()
    v = state["v"].ravel()
    q = state["q"].ravel()
    ba = state["b_a"].ravel()
    bg = state["b_g"].ravel()

    print(f"  p  (NED)    : [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m")
    print(f"  v  (NED)    : [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}] m/s")
    print(f"  q  (w,x,y,z): [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
    print(f"  b_a         : [{ba[0]:.2e}, {ba[1]:.2e}, {ba[2]:.2e}] m/s²")
    print(f"  b_g         : [{bg[0]:.2e}, {bg[1]:.2e}, {bg[2]:.2e}] rad/s")
    print(f"  trace(P)    : {np.trace(state['P']):.4f}")
    print(f"  P[0:3,0:3] diag: {np.diag(state['P'][0:3,0:3])}")
```

## 八、验证标准

| 检查点 | 预期行为 |
|--------|---------|
| 初始化附近 GNSS 更新后 `trace(P)` | 应**减小**（观测带来信息） |
| 后续 4 帧纯 IMU 的 `trace(P)` | 应总体**增大**（不确定性积累） |
| GNSS 转换后的 NED | 约 `[-0.172, 0.277, -0.198] m`，不是零 |
| K (卡尔曼增益) | 前 3 行应有显著非零值（GNSS 修正位置的主要方向） |
| `-C[f_b×]` 块 | 检查 F[3:6,6:9] 非零且单位为 `1/s` |
| q 始终是单位四元数 | `norm(q) ≈ 1.0` |
