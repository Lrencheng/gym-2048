import gymnasium as gym

from scripts.evaluate import make_policy as make_evaluate_policy
from scripts.evaluate import run_episodes
from scripts.enjoy import make_policy as make_enjoy_policy
from scripts.train import parse_args


def test_evaluate_expectimax_integration_smoke():
    env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    policy = make_evaluate_policy("expectimax", depth=1, seed=13)

    lengths, _rewards, _max_tiles, scores, illegal_counts, _runtime = run_episodes(
        env=env,
        policy=policy,
        n_episodes=1,
        seed=13,
    )
    env.close()

    assert len(lengths) == 1
    assert len(scores) == 1
    assert illegal_counts[0] == 0


def test_enjoy_expectimax_policy_factory():
    policy = make_enjoy_policy("expectimax", depth=1, seed=17)

    assert policy.predict is not None


def test_train_parser_accepts_supervised_agent(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--agent",
            "supervised_cnn",
            "--data",
            "data.npz",
            "--epochs",
            "1",
        ],
    )

    args = parse_args()

    assert args.algo == "supervised_cnn"
    assert args.data == "data.npz"
