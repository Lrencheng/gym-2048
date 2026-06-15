import torch

from gymnasium_2048.agents.supervised_cnn import CNNConfig, SupervisedCNN


def test_supervised_cnn_forward_returns_one_value_per_board():
    model = SupervisedCNN(CNNConfig())
    boards = torch.zeros((2, 16, 4, 4), dtype=torch.float32)

    values = model(boards)

    assert values.shape == (2,)
    assert torch.isfinite(values).all()
