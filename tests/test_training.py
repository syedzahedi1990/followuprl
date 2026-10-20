import copy
import math

import pytest
import torch

from paper2.training.finite_grpo import (
    clipped_objective, group_advantages, mixed_group_probability,
    policy_logprobs, training_step,
)
from paper2.training.toy import TinyPolicy


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_analytic_one_update_gradient_has_known_sign_despite_zero_scalar_loss():
    logits = torch.zeros(1, 8, requires_grad=True)
    _, logq = policy_logprobs(logits)
    actions = torch.tensor([[0, 1]])
    old = logq.detach().gather(-1, actions)
    advantages = group_advantages(torch.tensor([[1.0, 0.0]]))
    loss = clipped_objective(logq, actions, old, advantages).mean()
    assert abs(loss.item()) < 1e-8
    loss.backward()
    assert torch.allclose(logits.grad, torch.tensor([[-0.5, 0.5, 0, 0, 0, 0, 0, 0]]))


def test_population_reward_std_and_equal_groups():
    advantages = group_advantages(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0, 0, 0, 0], [1, 1, 1, 1]]))
    assert advantages[0, 0].item() == pytest.approx(math.sqrt(3), rel=1e-6)
    assert advantages[0, 1].item() == pytest.approx(-1 / math.sqrt(3), rel=1e-6)
    assert torch.equal(advantages[1:], torch.zeros(2, 4))
    assert advantages.requires_grad is False
    assert mixed_group_probability(0.5, 8) == pytest.approx(254 / 256)


def test_exploration_likelihood_and_gradient_use_same_mixture():
    logits = torch.tensor([[3.0, 1.0, 0, 0, 0, 0, -1, -2]], requires_grad=True)
    logp, logq = policy_logprobs(logits, exploration_epsilon=0.2)
    assert torch.allclose(logq.exp(), 0.8 * logp.exp() + 0.2 / 8)
    actions = torch.tensor([[0, 7]])
    old = logq.detach().gather(-1, actions)
    loss = clipped_objective(logq, actions, old, torch.tensor([[1.0, -1.0]])).mean()
    grad = torch.autograd.grad(loss, logits)[0]
    reference_logits = logits.detach().clone().requires_grad_()
    q = 0.8 * reference_logits.softmax(-1) + 0.2 / 8
    reference = -(q.log()[0, 0] - q.log()[0, 7]) / 2
    expected = torch.autograd.grad(reference, reference_logits)[0]
    assert torch.allclose(grad, expected)


def test_carried_adam_moves_on_graph_connected_zero_task_gradient():
    policy = TinyPolicy(101)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0)
    generator = torch.Generator().manual_seed(101)
    warm = training_step(policy, ["first", "second"], lambda i, a: float(a in {"0", "1", "2", "3"}),
                         optimizer, generator=generator, group_size=128)
    assert warm.metrics["task_gradient_norm"] > 0
    carried = training_step(policy, ["first", "second"], lambda i, a: 0.0,
                            optimizer, generator=generator, group_size=8)
    assert carried.metrics["task_gradient_norm"] == 0
    assert carried.metrics["parameter_update_norm"] > 0
    assert carried.metrics["all_zero_fraction"] == 1
    assert carried.metrics["optimizer_step_min"] == 2
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in policy.parameters())
    fresh = torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0)
    reset = training_step(policy, ["first"], lambda i, a: 0, fresh,
                          generator=generator, group_size=8)
    assert reset.metrics["parameter_update_norm"] == 0


def test_training_step_samples_one_distribution_and_records_original_rewards():
    policy = TinyPolicy(1)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0)
    result = training_step(policy, ["a", "b", "c"], lambda i, a: float(a == str(i)),
                           optimizer, generator=torch.Generator().manual_seed(5),
                           microbatch_size=2, group_size=32, exploration_epsilon=0.1,
                           families=["belief", "world", "arithmetic"], diagnose_family_gradients=True)
    assert result.metrics["sampled_actions"] == 96
    assert result.metrics["optimizer_step_max"] == 1
    assert "affected_gradient_norm" in result.metrics and "protected_gradient_norm" in result.metrics
    for row in result.records:
        assert sum(row["p"]) == pytest.approx(1, abs=1e-6)
        assert sum(row["q"]) == pytest.approx(1, abs=1e-6)
        for action, probability in zip(row["actions"], row["behaviour_logprobs"]):
            assert probability == pytest.approx(math.log(row["q"][action]), abs=1e-6)


def test_invalid_reward_and_hyperparameters_rejected():
    policy = TinyPolicy(1)
    optimizer = torch.optim.AdamW(policy.parameters())
    with pytest.raises(ValueError, match="binary rewards"):
        training_step(policy, ["x"], lambda i, a: 2, optimizer,
                      generator=torch.Generator().manual_seed(1))
    with pytest.raises(ValueError):
        policy_logprobs(torch.zeros(1, 8), exploration_epsilon=1)
    with pytest.raises(ValueError):
        policy_logprobs(torch.zeros(1, 9))
