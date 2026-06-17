# 2048 智能体项目

本项目基于 [Quentin18/gymnasium-2048](https://github.com/Quentin18/gymnasium-2048) 二次开发，用 2048 游戏环境对比和训练多种智能体策略，包括启发式搜索、Expectimax、遗传算法调参、SL1 动作概率模仿、SL2 afterstate value 监督学习、RLSL 以及 N-Tuple 强化学习。

## Python 环境

运行脚本、测试和检查时请统一使用当前环境中的 `python`：

```powershell
python -m <module-or-tool>
```

安装开发依赖：

```powershell
python -m pip install -e .[training,testing]
```

## 项目结构

```text
gymnasium-2048/
├── scripts/                         # 命令行脚本入口
│   ├── play.py                      # 人工键盘游玩
│   ├── random_policy.py             # 随机策略演示
│   ├── enjoy.py                     # 加载策略并可视化运行
│   ├── evaluate.py                  # 按 YAML 评估智能体
│   ├── train.py                     # 按 YAML 训练各类可训练 agent
│   └── generate_expectimax_data.py  # 生成 Expectimax 教师数据
├── src/gymnasium_2048/
│   ├── envs/                        # Gymnasium 2048 环境
│   ├── wrappers/                    # 环境包装器
│   └── agents/
│       ├── heuristic/               # 一步启发式策略与共享特征库
│       ├── expectimax/              # Expectimax 搜索策略与教师数据生成
│       ├── evolution/               # 遗传算法调参，支持 heuristic / expectimax
│       ├── SL1/                     # 早期动作概率 CNN 模仿学习
│       ├── supervised_cnn/          # CNN afterstate value 回归
│       ├── RLSL/                    # Search-improved afterstate value learning
│       └── ntuple/                  # N-Tuple TD/Q-Learning
├── tests/                           # 单元测试与集成冒烟测试
├── models/                          # 训练产物
├── data/                            # 数据集
└── figures/                         # 评估图表
```

## 常用命令

以下命令默认在仓库根目录 `gymnasium-2048/` 下运行。

### 测试

```powershell
python -m pytest -q
```

### 评估智能体

配置文件位于 `src/gymnasium_2048/agents/<agent>/configs/evaluate.yaml`。

```powershell
# 评估 heuristic
python -m scripts.evaluate --agent heuristic

# 评估 expectimax
python -m scripts.evaluate --agent expectimax

# 评估 SL1 动作概率网络，直接贪心选择动作
python -m scripts.evaluate --agent SL1

# 评估 supervised_cnn
python -m scripts.evaluate --agent supervised_cnn

# 评估指定 YAML
python -m scripts.evaluate --config src/gymnasium_2048/agents/expectimax/configs/evaluate.yaml

# 只打印解析后的配置
python -m scripts.evaluate --agent heuristic --print-config
```

### 可视化运行策略

```powershell
# heuristic
python -m scripts.enjoy --agent heuristic -n 5

# expectimax，使用采样 chance node
python -m scripts.enjoy --agent expectimax --depth 2 --chance-samples 6 -n 1

# supervised_cnn，depth=0 表示直接用 CNN 估值 afterstate
python -m scripts.enjoy --agent supervised_cnn --checkpoint models/supervised_cnn/train1/checkpoints/best.pt --depth 0 -n 5

# SL1，直接用动作概率网络输出选择合法动作
python -m scripts.enjoy --agent SL1 --checkpoint models/supervise/train4/checkpoints/best.pt -n 5

# 录制视频
python -m scripts.enjoy --agent heuristic --record-video --video-folder videos/heuristic -n 3
```

### 人工游玩与随机策略

```powershell
# 人工键盘游玩
python -m scripts.play

# 随机策略演示
python -m scripts.random_policy --n-timesteps 100
```

## Evolution 遗传算法调参

Evolution 现在通过 `--agent heuristic|expectimax` 选择调参目标，并自动加载对应训练配置：

- `src/gymnasium_2048/agents/evolution/configs/train_heuristic.yaml`
- `src/gymnasium_2048/agents/evolution/configs/train_expectimax.yaml`

训练配置中的 `policy_config` 会引用对应 agent 的 `evaluate.yaml`，训练时可覆盖 `reward_transform`、搜索深度和权重等策略参数。`workers` 控制并行评估 candidate 的进程数；`out_dir` 用于指定训练结果保存目录。命令行 `--workers` 和 `--out-dir` 可临时覆盖 YAML 中的设置；输出的 `best_<agent>_weights.json` 会保存最佳权重、训练配置和合并后的策略配置。

```powershell
# 快速冒烟：heuristic
python -m gymnasium_2048.agents.evolution.run_evolution --agent heuristic --population-size 2 --generations 1 --episodes 1 --workers 1 --elite-size 1 --tournament-size 1 --no-progress --out-dir models/evolution_smoke/heuristic

# 快速冒烟：expectimax
python -m gymnasium_2048.agents.evolution.run_evolution --agent expectimax --population-size 2 --generations 1 --episodes 1 --workers 2 --elite-size 1 --tournament-size 1 --no-progress --out-dir models/evolution_smoke/expectimax

# 正式训练 heuristic 权重
python -m gymnasium_2048.agents.evolution.run_evolution --agent heuristic --generations 13 --population-size 20 --episodes 50 --workers 8 --seed 42 --out-dir models/evolution/heuristic

# 正式训练 expectimax 权重
python -m gymnasium_2048.agents.evolution.run_evolution --agent expectimax --generations 15 --population-size 20 --episodes 15 --workers 8 --seed 42 --out-dir models/evolution/expectimax

# 使用指定训练 YAML
python -m gymnasium_2048.agents.evolution.run_evolution --agent expectimax --config src/gymnasium_2048/agents/evolution/configs/train_expectimax.yaml --out-dir models/evolution/expectimax_custom

# Expectimax 分阶段训练：stage1 / stage2 / final
python -m gymnasium_2048.agents.evolution.run_expectimax_staged --seed 42 --workers 8 --out-dir models/evolution/expectimax_staged

# Expectimax 分阶段冒烟测试
python -m gymnasium_2048.agents.evolution.run_expectimax_staged --seed 42 --workers 1 --no-progress --out-dir models/evolution_smoke/expectimax_staged
```

分阶段脚本的阶段参数写在 `src/gymnasium_2048/agents/evolution/run_expectimax_staged.py` 顶部，便于快速调整。训练阶段会分别写出 `stage1/parameter_evolution.png`、`stage2/parameter_evolution.png`、对应的 `evolution_history.csv` 和 `best_expectimax_weights.json`；最终复评阶段写出 `final/final_ranking.csv`、`final/best_expectimax_weights.json` 和顶层 `staged_expectimax_summary.json`。

## Expectimax 教师数据

每个根状态会枚举所有合法动作，并保存：

```text
after_boards -> target_us
```

其中动作价值始终按 `immediate_reward + target_u` 计算；CNN 只拟合 `target_u`。

```powershell
# 单个压缩 NPZ
python -m scripts.generate_expectimax_data --episodes 1000 --depth 2 --workers 4 --out data/expectimax_afterstates.npz --seed 42

# 分片保存，并预生成全部 8 种对称样本
python -m scripts.generate_expectimax_data --episodes 1000 --depth 2 --workers 8 --out data/expectimax_afterstates --shard-size 100000 --symmetry-augmentation --seed 42

python -m scripts.generate_expectimax_data --episodes 10 --chance-samples 6 --depth 2 --workers 8 --out data/expectimax_afterstates --shard-size 100000 --symmetry-augmentation --seed 42

# 小规模数据生成冒烟
python -m scripts.generate_expectimax_data --episodes 2 --depth 0 --max-steps 20 --out data/expectimax_smoke.npz --seed 42 --no-progress
```

`depth` 表示随机新块生成后还向前搜索多少次玩家决策；`depth=0`
直接用叶 evaluator 估值当前 afterstate。

## Afterstate Value CNN

训练配置位于 `src/gymnasium_2048/agents/supervised_cnn/configs/train.yaml`，评估配置位于 `src/gymnasium_2048/agents/supervised_cnn/configs/evaluate.yaml`。

```powershell
# 训练标量 U(x) 回归网络；训练集每次读取会随机应用一种 D4 对称
python -m scripts.train --agent supervised_cnn

# 使用指定训练 YAML
python -m scripts.train --config src/gymnasium_2048/agents/supervised_cnn/configs/train.yaml

# 用 CNN 作为 expectimax 叶 evaluator 评估
python -m scripts.evaluate --agent supervised_cnn
```

## SL1 动作概率 CNN

SL1 模块位于 `src/gymnasium_2048/agents/SL1/`，来自早期 `expectimax` 分支。它输入当前棋盘的 16 通道 one-hot 编码，输出四个动作 logits；推理时对非法动作做 mask，再直接取 `argmax(masked_logits)`。因此 SL1 默认评估是动作概率网络的裸贪心策略，不叠加 Expectimax 搜索。

SL1 使用旧格式教师数据，数据集中需要包含 `boards`、`legal_masks`、`action_probs`、`actions`、`final_scores`、`final_max_tiles`、`empty_counts` 和 `steps` 字段。默认配置位于 `src/gymnasium_2048/agents/SL1/configs/`。

```powershell
# 训练 SL1 动作概率模仿网络
python -m scripts.train --agent SL1

# 使用已有 SL1 checkpoint 评估贪心策略
python -m scripts.evaluate --agent SL1

# 查看解析后的 SL1 配置
python -m scripts.train --agent SL1 --print-config
```

## RLSL Search-Improved Value Learning

RLSL 模块位于 `src/gymnasium_2048/agents/RLSL/`，实现单局 self-play 后再做监督学习更新的 search-improved afterstate value learning。当前 `search_depth` 固定支持任务定义中的 `depth=1`：动作选择使用 `immediate_reward + U_search_depth1(afterstate)`，训练标签只保存被实际选中的根 afterstate 的 `U_search_depth1(afterstate)`，不包含当前动作奖励。

默认配置位于 `src/gymnasium_2048/agents/RLSL/configs/train.yaml`。训练产物默认写入 `models/RLSL/`：

```powershell
# 使用默认 RLSL 配置训练
python -m scripts.train --agent RLSL

# 查看解析后的配置
python -m scripts.train --agent RLSL --print-config

# 使用自定义 YAML
python -m scripts.train --config src/gymnasium_2048/agents/RLSL/configs/train.yaml
```

训练会持续更新 `checkpoints/last.pt` 和 `checkpoints/best.pt`，并写出 `history.json`、`config.json`。当 `eval_interval > 0` 时，会在 `evaluate_<round>/` 下保存 depth=0 评估的 `.png` 图和 `summary.json`。

## N-Tuple 强化学习

可训练 agent：`ql`、`tdl`、`tdl-small`。默认配置位于 `src/gymnasium_2048/agents/ntuple/configs/`。

```powershell
# 训练 TD-Learning
python -m scripts.train --agent tdl

# 训练 Q-Learning
python -m scripts.train --agent ql

# 评估已有 N-Tuple 模型
python -m scripts.evaluate --agent tdl
```

## 配置文件说明

- `heuristic/configs/evaluate.yaml`：heuristic 评估配置，默认 `reward_transform: raw`，保留旧行为。
- `expectimax/configs/evaluate.yaml`：expectimax 评估配置；`depth` 使用 afterstate recurrence 语义。
- `SL1/configs/*.yaml`：旧动作概率 CNN 模仿学习训练与直接贪心评估。
- `supervised_cnn/configs/*.yaml`：CNN value 回归训练与 expectimax 集成评估。
- `evolution/configs/train_heuristic.yaml`：heuristic 权重搜索空间和 evolution 超参数。
- `evolution/configs/train_expectimax.yaml`：expectimax 权重搜索空间和 evolution 超参数。

`reward_transform` 支持：

- `raw`：直接使用环境 reward。
- `log2p1`：使用 `log2(reward + 1)`。
- `none`：忽略即时 reward，主要用于消融实验。

## 依赖

- Python >= 3.10
- gymnasium 1.2.1
- pygame 2.6.1
- numpy
- PyYAML
- tqdm
- matplotlib
- torch，用于 supervised_cnn

