# 2048 智能体 — 人工智能概论课程大作业

本项目基于 [Quentin18/gymnasium-2048](https://github.com/Quentin18/gymnasium-2048) 进行二次开发，在此致谢原作者的优秀工作。

## 项目背景

北京理工大学 2026 年春学期《人工智能概论》课程大作业。

项目以 2048 游戏为实验平台，综合运用多种人工智能算法实现游戏决策，包括启发式搜索、遗传算法、自监督学习以及强化学习方法。

## 任务目标

1. **启发式搜索 + 遗传算法调参** — 设计多维度的启发式评价函数，对棋盘的空格数、平滑度、单调性、合并潜力等特征进行加权评分，并使用遗传算法自动搜索最优权重组合。

2. **监督学习** — 使用当前最强的Expertimax算法生成教师数据，构建一个小型的CNN学生网络对教师数据进行蒸馏；全程数据集和训练过程支持高分数据加权，使用mask进行违规动作抑制。

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
