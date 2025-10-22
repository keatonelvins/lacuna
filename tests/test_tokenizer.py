from pathlib import Path
from types import SimpleNamespace
from transformers import AutoTokenizer

from lacuna.config import DataConfig, ModelConfig
from lacuna.data import get_tokenizer, encode


def test_add_eos_token():
    """Test if adding a new eos token configures the tokenizer correctly."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    assert tokenizer.eos_token == "<|endoftext|>"
    assert tokenizer.eos_token_id == 151643

    mock_config = SimpleNamespace(
        model=ModelConfig(name="Qwen/Qwen3-0.6B-Base"),
        data=DataConfig(eos_token="<|im_end|>"),
    )

    tokenizer = get_tokenizer(mock_config)

    assert tokenizer.eos_token == "<|im_end|>"
    assert tokenizer.eos_token_id == 151645

    text = """And blood-black nothingness began to spin
A system of cells interlinked within
Cells interlinked within cells interlinked
Within one stem. And dreadfully distinct
Against the dark, a tall white fountain played."""

    results = encode({"text": [text]}, tokenizer, "text")
    maybe_eos_token = results["input_ids"][0][-1]

    assert maybe_eos_token == 151645


def test_encode_messages():
    """Test encode function with messages column."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    mock_config = SimpleNamespace(
        model=ModelConfig(name="Qwen/Qwen3-0.6B"),
        data=DataConfig(column="messages", chat_template=Path("tests/test.jinja").read_text()),
    )

    tokenizer = get_tokenizer(mock_config)

    messages = [
        {"role": "user", "content": "What's it like to hold the hand of someone you love?"},
        {"role": "assistant", "content": "Within cells interlinked"},
    ]

    expected_text = """<|im_start|>user
What's it like to hold the hand of someone you love?<|im_end|>
<|im_start|>assistant
Within cells interlinked<|im_end|>
"""

    results = encode({"messages": [messages]}, tokenizer, "messages", template=True)

    input_ids = results["input_ids"][0]
    assistant_masks = results["assistant_masks"][0]

    text = tokenizer.decode(input_ids)
    assert text == expected_text

    assert len(input_ids) == len(assistant_masks)

    user_tokens = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assistant_tokens = [
        1,  # Within
        1,  # cells
        1,  # inter
        1,  # linked
        1,  # <|im_end|>
    ]
    trailing_newline = [0]
    expected_masks = user_tokens + assistant_tokens + trailing_newline

    assert assistant_masks == expected_masks
