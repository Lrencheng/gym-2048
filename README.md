# 2048 智能体 — 人工智能概论课程大作业

本项目基于 [Quentin18/gymnasium-2048](https://github.com/Quentin18/gymnasium-2048) 进行二次开发，在此致谢原作者的优秀工作。

## 项目背景

北京理工大学 2026 年春学期《人工智能概论》课程大作业。

项目以 2048 游戏为实验平台，综合运用多种人工智能算法实现游戏决策，涵盖启发式搜索、遗传算法、Expectimax 搜索、监督学习（知识蒸馏）以及强化学习等多种方法。

## 任务目标

1. **启发式搜索 + 遗传算法调参** — 设计多维度的启发式评价函数，对棋盘的空格数、平滑度、单调性、合并潜力等特征进行加权评分，并使用遗传算法自动搜索最优权重组合。

2. **监督学习** — 使用当前最强的Expertimax算法生成教师数据，构建一个小型的CNN学生网络对教师数据进行蒸馏；全程数据集和训练过程支持高分数据加权，使用mask进行违规动作抑制。

3. **强化学习** — 实现基于 N-Tuple Network 的强化学习算法（TD-Learning / Q-Learning），通过与环境的持续交互优化决策策略。

## 项目结构

```
gymnasium-2048/
├── scripts/                          # 可执行脚本
│   ├── play.py                       # 人工键盘游玩
│   ├── random_policy.py              # 随机策略演示
│   ├── train.py                      # N-Tuple 网络强化学习训练
│   ├── enjoy.py                      # 加载模型观看游玩
│   ├── evaluate.py                   # 评估策略并生成统计图
│   ├── plot.py                       # 训练日志绘图
│   └── generate_expectimax_data.py   # Expectimax 教师数据生成
│
├── src/gymnasium_2048/
│   ├── __init__.py                   # 环境注册
│   ├── envs/
│   │   └── twenty_forty_eight.py     # 2048 游戏环境（基于 Gymnasium）
│   ├── agents/
│   │   ├── heuristic/                # 启发式搜索策略
│   │   │   ├── features.py           # 评价函数库（空格数、平滑度、单调性等）
│   │   │   └── policy.py             # 启发式策略类
│   │   ├── evolution/                # 遗传算法调参
│   │   │   ├── config.py             # 参数边界与进化配置
│   │   │   ├── parameters.py         # 参数向量与权重的转换
│   │   │   ├── evaluation.py         # 适应度评估
│   │   │   ├── genetic.py            # 遗传算法核心（选择、交叉、变异）
│   │   │   ├── plotting.py           # 进化过程可视化
│   │   │   └── run_evolution.py      # 进化算法入口
│   │   ├── expectimax/               # Expectimax 搜索（教师模型）
│   │   │   ├── policy.py             # Expectimax 搜索策略
│   │   │   ├── heuristic.py          # 搜索用启发式评价函数
│   │   │   ├── board.py              # 棋盘工具函数
│   │   │   ├── data.py               # 教师数据生成与加载
│   │   │   └── fast_move.py          # 加速版棋盘移动
│   │   ├── supervised_cnn/           # 监督学习（CNN 知识蒸馏）
│   │   │   ├── data.py               # 数据集管理与采样
│   │   │   ├── encoding.py           # 棋盘编码与动作掩码
│   │   │   ├── model.py              # CNN 学生网络
│   │   │   ├── loss.py               # 加权交叉熵损失
│   │   │   ├── train.py              # 训练脚本
│   │   │   └── policy.py             # 推理策略类
│   │   └── ntuple/                   # N-Tuple Network 强化学习
│   │       ├── network.py            # N-Tuple 网络（查表）
│   │       ├── factory.py            # 特征提取（棋盘 → n-tuples）
│   │       └── policy.py             # TD-Learning / Q-Learning 策略
│   └── wrappers/                     # Gymnasium 环境包装器
│       ├── illegal_reward.py
│       ├── terminate_goal.py
│       └── terminate_illegal.py
│
├── models/                           # 训练产出的模型与结果
│   ├── evolution/                    # 遗传算法调参结果（小规模）
│   └── evolution_large/              # 遗传算法调参结果（大规模）
│
├── figures/                          # 统计图表与可视化结果
├── tests/                            # 单元测试
│   ├── test_heuristic.py
│   ├── test_evolution.py
│   ├── test_expectimax_*.py
│   ├── test_supervised_*.py
│   ├── test_envs.py
│   └── test_agents.py
│
├── HEURISTIC_GUIDE.md                # 启发式算法开发指南
├── pyproject.toml                    # 项目元数据与依赖
└── README.md                         # 本文件
```

## 快速开始

```bash
# 安装依赖
pip install -e .

# 执行脚本
# 注意一下脚本默认是在linux环境下，如果是在windows环境下需要将“/”统一修改为“\”

# 运行启发式策略评估
python scripts/evaluate.py --algo heuristic -n 1000 -o figures/heuristic_stats.png

# 观看启发式策略游玩
python scripts/enjoy.py --algo heuristic -n 5

#  遗传算法调参
python -m gymnasium_2048.agents.evolution.run_evolution --generations 20 --population-size 20 --episodes 50 --seed 42 --out-dir models/evolution_large

# Expertimax教师数据生成
## 稳健指令
python scripts/generate_expectimax_data.py --episodes 1000 --depth 2 --chance-samples 6 --workers 4 --out data/expectimax_d2_sampled_1000eps.npz --seed 42

#CPU核心充足
python scripts/generate_expectimax_data.py --episodes 1000 --depth 2 --chance-samples 6 --workers 8 --out data/expectimax_d2_sampled_1000eps_w8.npz --seed 42
```
## 依赖

- Python >= 3.10
- gymnasium 1.2.1
- pygame 2.6.1
- numpy
