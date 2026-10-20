"""CPU checks of the real Transformers and PEFT path using a tiny random Qwen."""
import copy

import pytest
import torch

from paper2.training.policy import ACTIONS, HFActionPolicy, PolicyValidationError


class CharacterTokenizer:
    """A test-only tokenizer with an explicit chat boundary and single characters."""
    chat_template = "test-only: <user>{{content}}<assistant>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 2 for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<u>" + messages[0]["content"] + "</u><a>"


@pytest.fixture
def policy():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, attention_dropout=0.2,
    )
    config._attn_implementation = "eager"
    model = get_peft_model(
        Qwen2ForCausalLM(config),
        LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                   target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM),
    )
    return HFActionPolicy(
        model, CharacterTokenizer(), max_prompt_tokens=128,
        identity={"model_id": "random-tiny-qwen-test-only"},
    )


def test_real_qwen_last_position_and_left_padding(policy):
    prompts = ["short?", "A much longer prompt with five moves."]
    policy.eval()
    with torch.no_grad():
        together = policy.action_logits(prompts)
        separate = torch.cat([policy.action_logits([p]) for p in prompts])
    assert together.shape == (2, 8)
    torch.testing.assert_close(together, separate, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(together.softmax(-1).sum(-1), torch.ones(2))
    inputs = policy.prepare_inputs(prompts)
    assert inputs["attention_mask"][0, 0] == 0
    assert inputs["position_ids"][0, -1] == len(policy.validate_prompt(prompts[0])[0]) - 1
    assert policy.metadata["action_token_ids"] == [ord(a) + 2 for a in ACTIONS]
    with torch.no_grad():
        full = policy.model(**inputs, use_cache=False).logits[:, -1]
    torch.testing.assert_close(together, full[:, policy.metadata["action_token_ids"]])


def test_adapter_gradients_and_frozen_base(policy):
    assert all(
        ("lora_" in name) == parameter.requires_grad
        for name, parameter in policy.named_parameters()
    )
    before = {
        name: value.detach().clone()
        for name, value in policy.named_parameters() if not value.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=0.01, weight_decay=0
    )
    policy.train()
    logits = policy.action_logits(["where?", "longer where?"])
    (-logits.log_softmax(-1)[:, 0].mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    optimizer.step()
    for name, value in policy.named_parameters():
        if not value.requires_grad:
            assert torch.equal(value, before[name])
    first = policy.action_logits(["where?"])
    second = policy.action_logits(["where?"])
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_trainable_checkpoint_roundtrip_and_name_rejection(policy):
    state = policy.trainable_state_dict()
    with torch.no_grad():
        for value in policy.parameters():
            if value.requires_grad:
                value.add_(1)
    policy.load_trainable_state_dict(state)
    for name, value in policy.named_parameters():
        if value.requires_grad:
            torch.testing.assert_close(value.cpu(), state[name])
    with pytest.raises(PolicyValidationError, match="names/order"):
        policy.load_trainable_state_dict(dict(reversed(list(state.items()))))
    invalid = copy.deepcopy(state)
    invalid[next(iter(invalid))] = torch.zeros(1)
    with pytest.raises(PolicyValidationError, match="shape"):
        policy.load_trainable_state_dict(invalid)


def test_overlong_prompt_is_rejected_without_truncation(policy):
    with pytest.raises(PolicyValidationError, match="must not be truncated"):
        policy.action_logits(["x" * 129])


def test_multitoken_or_boundary_mutating_action_is_rejected(policy):
    old_encode = policy.tokenizer.encode
    policy.tokenizer.encode = lambda text, **kw: (
        old_encode(text, **kw) + [222] if text.endswith("?") else old_encode(text, **kw)
    )
    with pytest.raises(PolicyValidationError, match="append one token"):
        policy.action_logits(["where"])


def test_invalid_model_revision_fails_before_model_download():
    from paper2.training.policy import load_policy
    with pytest.raises(PolicyValidationError, match="exact 40-character SHA"):
        load_policy({"model": {"id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "main"}})


def test_preflight_generation_backward_and_padding_path(policy, monkeypatch):
    import paper2.training.policy as policy_module
    monkeypatch.setattr(policy_module, "load_policy", lambda config: policy)
    result = policy_module.preflight_policy({}, ["where?", "where after a longer history?"])
    assert result["status"] == "passed"
    assert result["generation_tokens"] == 4
    assert result["adapter_gradient_norm"] > 0
    assert result["padded_unpadded_max_abs_error"] < 1e-5
    assert result["peak_cuda_allocated_bytes"] is None
    assert result["output_interface"]["prompt_count"] == 2
    assert all(0 < mass < 1 for mass in result["output_interface"]["full_vocabulary_action_mass"])
    assert all(p.grad is None for p in policy.parameters())


def test_correct_restricted_ranking_does_not_imply_valid_global_output(policy, monkeypatch):
    policy.validate_prompt("boundary fixture")
    action_ids = policy.metadata["action_token_ids"]
    logits = torch.full((2, 256), -10.0)
    logits[0, action_ids[2]] = 4.0
    logits[0, 200] = 12.0  # An out-of-action token dominates the correct digit.
    logits[1, action_ids[5]] = 13.0
    logits[1, 200] = 7.0
    monkeypatch.setattr(policy, "_last_position_logits", lambda prompts: logits)
    result = policy.output_interface_diagnostics(["first", "second"], ["2", "5"])
    assert result["eight_action_ranking_correct"] == [True, True]
    assert result["full_vocabulary_first_token_correct"] == [False, True]
    assert result["global_argmax_is_valid_digit"] == [False, True]
    assert result["full_vocabulary_valid_digit_mass"][0] < 0.001
    expected = logits.softmax(-1)[:, action_ids[:6]].sum(-1).tolist()
    assert result["full_vocabulary_valid_digit_mass"] == pytest.approx(expected)
