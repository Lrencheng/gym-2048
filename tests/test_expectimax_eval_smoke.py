import gymnasium as gym

from gymnasium_2048.agents.expectimax import ExpectimaxPolicy


def test_expectimax_tiny_episode_smoke():
    env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0")
    policy = ExpectimaxPolicy(depth=1, seed=1)
    _observation, info = env.reset(seed=1)

    for _ in range(3):
        action = policy.predict(info["board"])
        _observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
    assert info["illegal_count"] == 0
    assert info["total_score"] >= 0
