# Project1_GNSS_IMU

这是一个面向机器人定位、多传感器融合、SLAM 和 embodied AI 岗位准备的 GNSS/IMU 组合导航学习项目。

本项目不仅追求代码能运行，更强调：

- 数学原理讲清楚
- 算法从简单版本逐步实现
- 每一步都有可复现验证
- 用图像和实验结果解释滤波行为
- 文档和学习笔记适合放入 GitHub 作品集

## 项目组织方式

当前仓库会按照学习型工程项目组织：

- `docs/`：数学推导、学习笔记、面试问题
- `python/`：Python 原型实现
- `tests/`：最小可复现测试和验证
- `data/`：小型合成数据或公开数据集说明
- `results/`：生成的图像、日志和实验结果

第一阶段以 Python 原型为主，先把状态估计和传感器模型讲清楚；后续再扩展到 C++17、Eigen、ROS 和 RViz。

## 学习路线

1. 坐标系与导航状态
2. 捷联惯导基础
3. Kalman Filter 与 Extended Kalman Filter
4. GNSS/IMU Error-State Kalman Filter
5. 合成轨迹、传感器仿真与可视化
6. 基于 Eigen 的 C++17 实现
7. ROS/RViz 风格的机器人定位演示

## Windows 11 环境准备

检查当前工具：

```powershell
git --version
python --version
cmake --version
```

如果没有 Python，可以安装后重新打开 PowerShell：

```powershell
winget install Python.Python.3.12
```

创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:MPLCONFIGDIR = "$PWD\.matplotlib-cache"
python -m pip install --upgrade pip
pip install -r requirements.txt
```

后续运行 Python 测试：

```powershell
pytest
```

## 当前状态

当前仓库只包含第一阶段项目结构和学习笔记，还没有加入完整算法实现。

下一步建议先做一个合成数据 demo：

- 生成简单二维或三维真值轨迹
- 生成低频 GNSS 位置观测
- 生成高频 IMU 加速度和角速度观测
- 用 Matplotlib 画出轨迹和传感器数据

这样可以先把单位、坐标系、噪声、采样率这些基础问题站稳，再进入 Kalman Filter 或 ESKF。

## 生成结果说明

`results/` 用于保存由脚本生成的图像、日志和实验输出。

建议原则：

- 重要图像要能通过脚本重新生成
- 每个实验说明使用了哪个命令
- 不提交大型二进制结果文件
