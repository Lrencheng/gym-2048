import pytest
import torch

from gymnasium_2048.agents.supervised_cnn import regression_loss


@pytest.mark.parametrize("kind", ["huber", "mse"])
def test_regression_loss_is_finite_and_differentiable(kind):
    predictions = torch.tensor([1.0, 2.0], requires_grad=True)
    targets = torch.tensor([1.5, 3.0])

    loss = regression_loss(predictions, targets, kind=kind)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(predictions.grad).all()


def test_regression_loss_rejects_unknown_kind():
    with pytest.raises(ValueError):
        regression_loss(torch.zeros(1), torch.zeros(1), kind="cross_entropy")
