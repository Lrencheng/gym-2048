import torch

from gymnasium_2048.agents.SL1 import masked_soft_cross_entropy


def test_masked_soft_cross_entropy_is_finite_and_masks_illegal_actions():
    logits = torch.tensor([[1.0, 2.0, 10.0, -1.0]], requires_grad=True)
    targets = torch.tensor([[0.25, 0.75, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    loss = masked_soft_cross_entropy(
        logits=logits,
        target_probs=targets,
        legal_mask=mask,
        temperature=2.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] == 0.0


def test_masked_soft_cross_entropy_accepts_float16_logits():
    logits = torch.tensor(
        [[1.0, 2.0, 10.0, -1.0]],
        dtype=torch.float16,
        requires_grad=True,
    )
    targets = torch.tensor([[0.25, 0.75, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    loss = masked_soft_cross_entropy(
        logits=logits,
        target_probs=targets,
        legal_mask=mask,
        temperature=1.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()

