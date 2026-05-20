import torch

from gymnasium_2048.agents.supervised_cnn import CNNConfig, SupervisedCNN


def test_supervised_cnn_forward_shape():
    model = SupervisedCNN(CNNConfig())
    boards = torch.zeros((2, 16, 4, 4), dtype=torch.float32)

    logits = model(boards)

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
