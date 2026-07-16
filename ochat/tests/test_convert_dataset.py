"""Tests for ochat.data.convert_dataset — SFT OpenAI → Conversation conversion.

Validates the multi-turn split, substring dedup, and ChatML round-trip logic
using Qwen/Qwen3.5-0.8B in offline mode.
"""

import os
import json
import tempfile
import pytest

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ochat.config.conversation_template import (
    ConversationOpenAI, Conversation, Message
)

# ChatML regex used by convert_dataset.py
CHATML_PATTERN = r"<\|im_start\|>([^\n]+)\n(.*?)(?:<\|im_end\|>(?=\s*(?:<\|im_start\|>|$)))"


# -- ConversationOpenAI parsing -----------------------------------------------

class TestConversationOpenAI:
    """Test ConversationOpenAI validation and parsing."""
    pytestmark = pytest.mark.cpu

    def test_simple_text_message(self):
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        conv = ConversationOpenAI.model_validate(data)
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        # content is a list of TextContentPart Pydantic models
        assert conv.messages[0].content[0].text == "Hello"
        assert conv.messages[1].role == "assistant"
        assert conv.messages[1].content[0].text == "Hi there!"

    def test_with_weights(self):
        data = {
            "messages": [
                {"role": "user", "content": "Q", "weight": 0.0},
                {"role": "assistant", "content": "A", "weight": 1.0},
            ]
        }
        conv = ConversationOpenAI.model_validate(data)
        assert conv.messages[0].weight == 0.0
        assert conv.messages[1].weight == 1.0

    def test_with_tools(self):
        data = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "NYC"}},
                        }
                    ],
                },
                {"role": "tool", "content": "Sunny, 72F"},
            ]
        }
        conv = ConversationOpenAI.model_validate(data)
        assert conv.messages[0].tool_calls is not None
        assert conv.messages[0].tool_calls[0].function.name == "get_weather"
        assert conv.messages[1].role == "tool"


# -- ChatML regex -------------------------------------------------------------

class TestChatMLParsing:
    """Test the ChatML regex used by convert_dataset to parse round-tripped text."""
    pytestmark = pytest.mark.cpu

    def test_simple_user_assistant(self):
        text = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>"
        matches = __import__("re").findall(CHATML_PATTERN, text, __import__("re").DOTALL)
        assert len(matches) == 2
        assert matches[0] == ("user", "Hello")
        assert matches[1] == ("assistant", "Hi!")

    def test_with_system(self):
        text = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHey!<|im_end|>"
        import re
        matches = re.findall(CHATML_PATTERN, text, re.DOTALL)
        assert len(matches) == 3
        assert matches[0] == ("system", "You are helpful.")
        assert matches[1] == ("user", "Hi")
        assert matches[2] == ("assistant", "Hey!")

    def test_multiline_content(self):
        text = "<|im_start|>assistant\nLine 1\nLine 2\nLine 3<|im_end|>"
        import re
        matches = re.findall(CHATML_PATTERN, text, re.DOTALL)
        assert len(matches) == 1
        assert matches[0] == ("assistant", "Line 1\nLine 2\nLine 3")

    def test_condition_in_role(self):
        text = "<|im_start|>GPT4 correct user\nHello<|im_end|>\n<|im_start|>GPT4 correct assistant\nHi!<|im_end|>"
        import re
        matches = re.findall(CHATML_PATTERN, text, re.DOTALL)
        assert len(matches) == 2
        assert matches[0] == ("GPT4 correct user", "Hello")
        assert matches[1] == ("GPT4 correct assistant", "Hi!")


# -- Conversation model -------------------------------------------------------

class TestConversationModel:
    """Test Conversation model construction and serialization."""
    pytestmark = pytest.mark.cpu

    def test_basic(self):
        conv = Conversation(
            condition="",
            items=[
                Message(role="user", content="Hello", weight=0.0),
                Message(role="assistant", content="Hi", weight=1.0),
            ],
        )
        assert conv.condition == ""
        assert conv.system == ""
        assert len(conv.items) == 2

    def test_serialization(self):
        conv = Conversation(
            system="You are helpful.",
            condition="",
            items=[
                Message(role="user", content="Q", weight=0.0),
                Message(role="assistant", content="A", weight=1.0),
            ],
        )
        d = conv.model_dump(exclude_none=True)
        assert d["system"] == "You are helpful."
        assert d["condition"] == ""
        assert len(d["items"]) == 2

    def test_empty_condition_excluded(self):
        conv = Conversation(
            condition="",
            items=[
                Message(role="user", content="Q", weight=0.0),
                Message(role="assistant", content="A", weight=1.0),
            ],
        )
        d = conv.model_dump(exclude_none=True)
        # Empty condition should still be in dump since exclude_none only skips None
        assert d["condition"] == ""


# -- Full conversion pipeline (in-process) ------------------------------------

@pytest.fixture(scope="module")
def qwen35_processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")


class TestSFTConversionPipeline:
    """End-to-end SFT conversion: OpenAI JSONL → OpenChat Conversation JSONL."""
    pytestmark = pytest.mark.cpu

    MODEL_TYPE = "qwen3_5_chatml"
    MODEL_PATH = "Qwen/Qwen3.5-0.8B"

    def _run_convert(self, qwen35_processor, lines, tmpdir):
        """Run the conversion pipeline in-process and return output lines."""
        from ochat.data.convert_dataset import process_batch
        import argparse

        args = argparse.Namespace(
            model_type=self.MODEL_TYPE,
            model_path=self.MODEL_PATH,
            use_json_repair=False,
        )
        results = process_batch(
            job_id=0,
            batch=lines,
            args=args,
            out_dir=str(tmpdir),
        )
        return results

    def test_single_turn(self, qwen35_processor, tmpdir):
        """Single user+assistant turn with weights."""
        lines = [json.dumps({
            "messages": [
                {"role": "user", "content": "Hello", "weight": 0.0},
                {"role": "assistant", "content": "Hi there!", "weight": 1.0},
            ]
        })]
        results = self._run_convert(qwen35_processor, lines, tmpdir)
        assert len(results) == 1

        conv = json.loads(results[0])
        items = conv["items"]
        assert len(items) == 2
        assert items[0]["role"] == "user"
        assert items[0]["content"] == "Hello"
        assert items[0]["weight"] == 0.0
        assert items[1]["role"] == "assistant"
        # Qwen3.5 chat template may wrap in <think> tags
        assert "Hi there!" in items[1]["content"]
        assert items[1]["weight"] == 1.0
        assert conv.get("condition", "") == ""

    def test_multi_turn_split(self, qwen35_processor, tmpdir):
        """Multi-turn: each weighted assistant turn becomes a separate output."""
        lines = [json.dumps({
            "messages": [
                {"role": "user", "content": "Q1", "weight": 0.0},
                {"role": "assistant", "content": "A1", "weight": 0.5},
                {"role": "user", "content": "Q2", "weight": 0.0},
                {"role": "assistant", "content": "A2", "weight": 1.0},
            ]
        })]
        results = self._run_convert(qwen35_processor, lines, tmpdir)
        # Should produce 2 conversations (one per weighted assistant turn)
        assert len(results) == 2

        # First output: Q1+A1 with weight 0.5
        conv1 = json.loads(results[0])
        assert conv1["items"][-1]["weight"] == 0.5
        assert "A1" in conv1["items"][-1]["content"]

        # Second output: Q1+A1+Q2+A2 with weight 1.0
        conv2 = json.loads(results[1])
        assert conv2["items"][-1]["weight"] == 1.0
        assert "A2" in conv2["items"][-1]["content"]

    def test_no_weighted_turn(self, qwen35_processor, tmpdir):
        """All weights 0 → no output."""
        lines = [json.dumps({
            "messages": [
                {"role": "user", "content": "Hello", "weight": 0.0},
                {"role": "assistant", "content": "Hi!", "weight": 0.0},
            ]
        })]
        results = self._run_convert(qwen35_processor, lines, tmpdir)
        assert len(results) == 0

    def test_system_message(self, qwen35_processor, tmpdir):
        """System message preserved in output."""
        lines = [json.dumps({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello", "weight": 0.0},
                {"role": "assistant", "content": "Hi!", "weight": 1.0},
            ]
        })]
        results = self._run_convert(qwen35_processor, lines, tmpdir)
        assert len(results) == 1
        conv = json.loads(results[0])
        assert conv["system"] == "You are helpful."
        assert len(conv["items"]) == 2

    def test_empty_condition(self, qwen35_processor, tmpdir):
        """Empty condition → no class prefix in output."""
        lines = [json.dumps({
            "messages": [
                {"role": "user", "content": "Hello", "weight": 0.0},
                {"role": "assistant", "content": "Hi!", "weight": 1.0},
            ]
        })]
        results = self._run_convert(qwen35_processor, lines, tmpdir)
        conv = json.loads(results[0])
        # Items should have empty condition (no class prefix)
        assert conv.get("condition", "") == ""

    def test_batch_mixed(self, qwen35_processor, tmpdir):
        """Multiple conversations in one batch."""
        lines = [
            json.dumps({
                "messages": [
                    {"role": "user", "content": "Q1", "weight": 0.0},
                    {"role": "assistant", "content": "A1", "weight": 1.0},
                ]
            }),
            json.dumps({
                "messages": [
                    {"role": "user", "content": "Q2", "weight": 0.0},
                    {"role": "assistant", "content": "A2", "weight": 1.0},
                ]
            }),
        ]
        results = self._run_convert(qwen35_processor, lines, tmpdir)
        assert len(results) == 2
