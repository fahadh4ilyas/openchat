"""Tests for conversation template tokenization and weight assignment.

Uses Qwen/Qwen3.5-0.8B via AutoProcessor for multimodal model types (qwen3_5),
and AutoTokenizer for text-only types (qwen3).  Requires TRANSFORMERS_OFFLINE=1.
"""

import os
import pytest

# Prevent network calls during tokenizer init (_patch_mistral_regex)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoProcessor, AutoTokenizer

from ochat.config import MODEL_CONFIG_MAP, Conversation


MODEL_35 = "Qwen/Qwen3.5-0.8B"   # multimodal, needs AutoProcessor


@pytest.fixture(scope="module")
def tokenizer_35():
    """Underlying tokenizer from Qwen3.5-0.8B AutoProcessor."""
    proc = AutoProcessor.from_pretrained(MODEL_35)
    return proc.tokenizer


# -- Common assertions --------------------------------------------------------

def _assert_no_special_token_leakage(tokens, weights):
    assert len(tokens) == len(weights), f"Token/weight length mismatch: {len(tokens)} vs {len(weights)}"
    assert len(tokens) > 0, "Empty token sequence"


def _assert_weight_ranges(weights, expected_set):
    unique = set(weights)
    for w in unique:
        assert w in expected_set, f"Unexpected weight {w}; expected one of {expected_set}"


# -- qwen3_5 (V3.2 template) --------------------------------------------------

class TestQwen35V32Template:
    """V3.2-style ConversationTemplate with <|end_of_turn|> delimiter."""

    pytestmark = pytest.mark.cpu
    MODEL_TYPE = "qwen3_5"
    EOT = "<|end_of_turn|>"

    @pytest.fixture(scope="class")
    def template(self, tokenizer_35):
        config = MODEL_CONFIG_MAP[self.MODEL_TYPE]
        return config.conversation_template(tokenizer=tokenizer_35)

    # -- empty condition (most common for SFT/DPO/ORPO) -----------------------

    def test_empty_condition_single_turn(self, template, tokenizer_35):
        """Single turn with empty condition — no class prefix in output."""
        conv = Conversation(
            condition="",
            items=[
                {"role": "user", "content": "Hello", "weight": 0.0},
                {"role": "assistant", "content": "Hi!", "weight": 1.0},
            ],
        )
        tokens_all, weights_all = template.tokenize_conversations([conv], inference=False)
        tokens, weights = tokens_all[0], weights_all[0]

        decoded = tokenizer_35.decode(tokens)
        # Empty condition means just "User:" / "Assistant:" without class prefix
        assert "User:Hello" in decoded or " User: Hello" in decoded
        assert "Assistant:Hi!" in decoded or " Assistant: Hi!" in decoded
        _assert_no_special_token_leakage(tokens, weights)
        _assert_weight_ranges(weights, {0.0, 1.0})
        # Weights: user prefix = 0, user content = 0, asst prefix = 0, asst content = 1.0
        assert 1.0 in weights
        assert 0.0 in weights

    def test_empty_condition_multi_turn(self, template, tokenizer_35):
        """Multi-turn with empty condition and mixed weights."""
        conv = Conversation(
            condition="",
            items=[
                {"role": "user", "content": "Q1", "weight": 0.0},
                {"role": "assistant", "content": "A1", "weight": 0.3},
                {"role": "user", "content": "Q2", "weight": 0.0},
                {"role": "assistant", "content": "A2", "weight": 1.0},
            ],
        )
        tokens_all, weights_all = template.tokenize_conversations([conv], inference=False)
        tokens, weights = tokens_all[0], weights_all[0]

        _assert_no_special_token_leakage(tokens, weights)
        _assert_weight_ranges(weights, {0.0, 0.3, 1.0})
        assert any(w == 0.3 for w in weights)
        assert any(w == 1.0 for w in weights)

        decoded = tokenizer_35.decode(tokens)
        assert decoded.count(self.EOT) >= 4  # EOT after each message

    # -- with condition (C-RLFT) ---------------------------------------------

    def test_condition_single_turn(self, template, tokenizer_35):
        """Single turn with C-RLFT condition prefix."""
        conv = Conversation(
            condition="GPT4",
            items=[
                {"role": "user", "content": "Hello", "weight": 0.0},
                {"role": "assistant", "content": "Hi!", "weight": 1.0},
            ],
        )
        tokens_all, weights_all = template.tokenize_conversations([conv], inference=False)
        tokens, weights = tokens_all[0], weights_all[0]

        decoded = tokenizer_35.decode(tokens)
        assert "GPT4 User:" in decoded
        assert "GPT4 Assistant:" in decoded
        _assert_no_special_token_leakage(tokens, weights)

    # -- system message ------------------------------------------------------

    def test_system_message(self, template, tokenizer_35):
        """System message with empty condition."""
        conv = Conversation(
            system="You are helpful.",
            condition="",
            items=[
                {"role": "user", "content": "Hi", "weight": 0.0},
                {"role": "assistant", "content": "Hello!", "weight": 1.0},
            ],
        )
        tokens_all, weights_all = template.tokenize_conversations([conv], inference=False)
        tokens, weights = tokens_all[0], weights_all[0]

        decoded = tokenizer_35.decode(tokens)
        assert "You are helpful." in decoded
        # System should appear before first user message
        sys_pos = decoded.find("You are helpful.")
        first_eot = decoded.find(self.EOT)
        assert sys_pos < first_eot

    # -- special token content -----------------------------------------------

    def test_special_token_in_content(self, template, tokenizer_35):
        """EOT-like text in user content should not break parsing."""
        conv = Conversation(
            condition="",
            items=[
                {"role": "user", "content": "What is <|end_of_turn|>?", "weight": 0.0},
                {"role": "assistant", "content": "It is a special marker.", "weight": 1.0},
            ],
        )
        tokens_all, weights_all = template.tokenize_conversations([conv], inference=False)
        tokens, weights = tokens_all[0], weights_all[0]

        decoded = tokenizer_35.decode(tokens)
        # The text should include both the literal tokens and the structural EOTs
        assert self.EOT in decoded

    # -- batching ------------------------------------------------------------

    def test_batched_conversations(self, template, tokenizer_35):
        """Multiple conversations in one call."""
        convs = [
            Conversation(
                condition="",
                items=[
                    {"role": "user", "content": "Hi", "weight": 0.0},
                    {"role": "assistant", "content": "Hey!", "weight": 1.0},
                ],
            ),
            Conversation(
                condition="GPT4",
                items=[
                    {"role": "user", "content": "Question", "weight": 0.0},
                    {"role": "assistant", "content": "Answer", "weight": 0.5},
                ],
            ),
        ]
        tokens_all, weights_all = template.tokenize_conversations(convs, inference=False)
        assert len(tokens_all) == 2
        for tokens, weights in zip(tokens_all, weights_all):
            _assert_no_special_token_leakage(tokens, weights)


# -- qwen3_5_chatml (ChatML template) -----------------------------------------

class TestQwen35ChatMLTemplate:
    """ChatMLConversationTemplate with <|im_start|>/<|im_end|> delimiters."""

    pytestmark = pytest.mark.cpu
    MODEL_TYPE = "qwen3_5_chatml"
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    @pytest.fixture(scope="class")
    def template(self, tokenizer_35):
        config = MODEL_CONFIG_MAP[self.MODEL_TYPE]
        return config.conversation_template(tokenizer=tokenizer_35)

    def test_empty_condition(self, template, tokenizer_35):
        """ChatML with empty condition — roles without class prefix."""
        conv = Conversation(
            condition="",
            items=[
                {"role": "user", "content": "Hello", "weight": 0.0},
                {"role": "assistant", "content": "Hi!", "weight": 1.0},
            ],
        )
        tokens_all, weights_all = template.tokenize_conversations([conv], inference=False)
        tokens, weights = tokens_all[0], weights_all[0]

        decoded = tokenizer_35.decode(tokens)
        assert self.IM_START in decoded
        assert self.IM_END in decoded
        assert "user\nHello" in decoded or "user Hello" in decoded
        assert "assistant\nHi!" in decoded or "assistant Hi!" in decoded
        _assert_no_special_token_leakage(tokens, weights)
        _assert_weight_ranges(weights, {0.0, 1.0})

    def test_with_condition(self, template, tokenizer_35):
        """ChatML with condition — roles include class prefix."""
        conv = Conversation(
            condition="GPT4 correct",
            items=[
                {"role": "user", "content": "Hello", "weight": 0.0},
                {"role": "assistant", "content": "Hi!", "weight": 1.0},
            ],
        )
        tokens_all, weights_all = template.tokenize_conversations([conv], inference=False)
        tokens, weights = tokens_all[0], weights_all[0]

        decoded = tokenizer_35.decode(tokens)
        assert "GPT4 correct user" in decoded
        assert "GPT4 correct assistant" in decoded

    def test_multi_turn_mixed_weights(self, template, tokenizer_35):
        """ChatML multi-turn with varying weights."""
        conv = Conversation(
            condition="",
            items=[
                {"role": "user", "content": "Q1", "weight": 0.0},
                {"role": "assistant", "content": "A1", "weight": 0.5},
                {"role": "user", "content": "Q2", "weight": 0.0},
                {"role": "assistant", "content": "A2", "weight": 1.0},
            ],
        )
        tokens_all, weights_all = template.tokenize_conversations([conv], inference=False)
        tokens, weights = tokens_all[0], weights_all[0]

        decoded = tokenizer_35.decode(tokens)
        im_start_count = decoded.count(self.IM_START)
        assert im_start_count == 4, f"Expected 4 {self.IM_START} blocks, got {im_start_count}"
        _assert_weight_ranges(weights, {0.0, 0.5, 1.0})

    def test_system_message(self, template, tokenizer_35):
        """ChatML with system prompt."""
        conv = Conversation(
            system="You are helpful.",
            condition="",
            items=[
                {"role": "user", "content": "Hi", "weight": 0.0},
                {"role": "assistant", "content": "Hello!", "weight": 1.0},
            ],
        )
        tokens_all, weights_all = template.tokenize_conversations([conv], inference=False)
        tokens, weights = tokens_all[0], weights_all[0]

        decoded = tokenizer_35.decode(tokens)
        assert "system" in decoded
        assert "You are helpful." in decoded

# -- model type registry ------------------------------------------------------

@pytest.mark.cpu
def test_model_type_registry():
    """Required model types exist in MODEL_CONFIG_MAP."""
    required = [
        "qwen3_5", "qwen3_5_chatml",
        "llama", "mistral", "mixtral",
        "qwen2", "qwen3",
        "gemma", "gemma2", "deepseekv2",
    ]
    for t in required:
        assert t in MODEL_CONFIG_MAP, f"Model type '{t}' not found"


@pytest.mark.cpu
def test_conversation_validation():
    """Conversation model construction and defaults."""
    conv = Conversation(
        items=[
            {"role": "user", "content": "Hello", "weight": 1.0},
            {"role": "assistant", "content": "Hi", "weight": 0.5},
        ],
    )
    assert conv.items[0].role == "user"
    assert conv.items[0].weight == 1.0
    assert conv.system == ""
    assert conv.condition == ""

    # No weights (inference)
    conv2 = Conversation(
        items=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    )
    assert conv2.items[0].weight is None


@pytest.mark.cpu
def test_templates_produce_different_output(tokenizer_35):
    """V3.2 and ChatML produce different token sequences."""
    conv = Conversation(
        condition="GPT4",
        items=[
            {"role": "user", "content": "Hello", "weight": 0.0},
            {"role": "assistant", "content": "Hi!", "weight": 1.0},
        ],
    )
    t1 = MODEL_CONFIG_MAP["qwen3_5"].conversation_template(tokenizer=tokenizer_35)
    t2 = MODEL_CONFIG_MAP["qwen3_5_chatml"].conversation_template(tokenizer=tokenizer_35)
    tok1, _ = t1.tokenize_conversations([conv], inference=False)
    tok2, _ = t2.tokenize_conversations([conv], inference=False)
    assert tok1[0] != tok2[0], "V3.2 and ChatML should differ"
