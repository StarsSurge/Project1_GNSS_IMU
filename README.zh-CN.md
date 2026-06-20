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

当前 Python 阶段已经包含：

- 教学版一维恒速 Kalman Filter
- 通用 N 维线性 Kalman Filter
- 距离-方位角 Extended Kalman Filter
- 可复现单元测试和雅可比数值校验
- 可运行的可视化示例与 GNSS/IMU、ESKF 学习笔记
- 面向真实命名列 IMU CSV 的 Allan 分析 MVP，包含时间戳校验、
  rate/增量转换以及可审计的 CSV、JSON 和 PNG 输出
- 生产导向的 15 状态 GNSS/IMU 松组合 ESKF 基线，包含 WGS-84 地球模型、
  双子样机械编排、杆臂、GNSS NIS 门限、Joseph 更新和 dataset1 回放
- 静止窗口检测、重力调平、外部航向/高精陀螺 gyrocompass 初始化

```powershell
$env:PYTHONPATH = "$PWD\python"
python python\examples\demo_1d_kalman_filter.py
python python\examples\demo_kalman_filter.py
python python\examples\demo_extended_kalman_filter.py
python python\examples\run_dataset1_eskf.py --duration-s 60
python -m pytest tests
```

Allan CSV 工具当前属于“验证过的 MVP”，尚未达到部署级：仓库仍缺少可追溯的
真实静态 IMU 数据、自动静止/温度/饱和检查以及基于等效自由度的置信区间。

GNSS/IMU ESKF 当前属于 verified MVP / production-oriented baseline，尚未完成
在线对准、精确延迟状态更新、温度模型、21 状态比例因子估计和实时部署验证。

## 生成结果说明

`results/` 用于保存由脚本生成的图像、日志和实验输出。

建议原则：

- 重要图像要能通过脚本重新生成
- 每个实验说明使用了哪个命令
- 不提交大型二进制结果文件
