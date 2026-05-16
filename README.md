# 2048 智能体 — 人工智能概论课程大作业

本项目基于 [Quentin18/gymnasium-2048](https://github.com/Quentin18/gymnasium-2048) 进行二次开发，在此致谢原作者的优秀工作。

## 项目背景

北京理工大学 2026 年春学期《人工智能概论》课程大作业。

项目以 2048 游戏为实验平台，综合运用多种人工智能算法实现游戏决策，包括启发式搜索、遗传算法、自监督学习以及强化学习方法。

## 任务目标

1. **启发式搜索 + 遗传算法调参** — 设计多维度的启发式评价函数，对棋盘的空格数、平滑度、单调性、合并潜力等特征进行加权评分，并使用遗传算法自动搜索最优权重组合。

2. **自监督学习** — 不依赖外部标注，让模型通过游戏自身的交互过程进行学习，自主发现有效的决策策略。

3. **强化学习** — 实现基于 N-Tuple Network 的强化学习算法（TD-Learning / Q-Learning），通过与环境的持续交互优化决策策略。

## 项目结构

```
src/gymnasium_2048/
├── envs/
│   └── twenty_forty_eight.py       # 2048 游戏环境（基于 Gymnasium）
├── agents/
│   ├── heuristic/                   # 启发式搜索策略
│   │   ├── features.py              # 评价函数库
│   │   └── policy.py                # 启发式策略类
│   └── ntuple/                      # N-Tuple Network 强化学习
│       ├── network.py               # N-Tuple 网络
│       ├── factory.py               # 特征提取
│       └── policy.py                # TD / Q-Learning 策略
├── wrappers/                        # 环境包装器
└── __init__.py                      # 环境注册
```

## 快速开始

```bash
# 安装依赖
pip install -e .

# 运行启发式策略评估
python scripts/evaluate.py --algo heuristic -n 1000 -o figures/heuristic_stats.png

# 观看启发式策略游玩
python scripts/enjoy.py --algo heuristic -n 5

#  遗传算法调参
python -m gymnasium_2048.agents.evolution.run_evolution --generations 20 --population-size 20 --episodes 50 --seed 42 --out-dir models\evolution_large
```
## 依赖

- Python >= 3.10
- gymnasium 1.2.1
- pygame 2.6.1
- numpy
