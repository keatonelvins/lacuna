import pyarrow as pa
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from lacuna.utils import pack_bfd
from lacuna.data import LacunaDataset, get_tokenizer, _encode


def test_bfd_packing():
    input_ids = [
        [1],
        [2, 2],
        [3, 3, 3],
        [4, 4, 4, 4],
        [6, 6, 6, 6, 6, 6],
    ]
    assistant_masks = [
        [1],
        [0, 1],
        [1, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]

    table = pa.Table.from_arrays([pa.array(input_ids), pa.array(assistant_masks)], names=["input_ids", "assistant_masks"])

    packed = pack_bfd(table, seq_len=5)

    expected_packed_input = [
        [6, 6, 6, 6, 6],
        [4, 4, 4, 4, 1],
        [3, 3, 3, 2, 2],
    ]
    expected_packed_pos = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 0],
        [0, 1, 2, 0, 1],
    ]
    expected_packed_masks = [
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
    ]

    assert packed["input_ids"].to_pylist() == expected_packed_input
    assert packed["position_ids"].to_pylist() == expected_packed_pos
    assert packed["assistant_masks"].to_pylist() == expected_packed_masks


def test_encode_text():
    """Test _encode method with text column appends eos token."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    mock_config = SimpleNamespace(
        model=SimpleNamespace(name="Qwen/Qwen3-0.6B-Base"),
        data=SimpleNamespace(column="text", chat_template=None, eos_token=None),
    )

    tokenizer = get_tokenizer(mock_config)

    text = "And then black night. That blackness was sublime. I felt distributed through space and time"

    results = _encode({"text": [text]}, tokenizer, "text")
    maybe_eos_token = results["input_ids"][0][-1]
    assert maybe_eos_token == tokenizer.eos_token_id


def test_add_eos_token():
    """Test if adding a new eos token configures the tokenizer correctly."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    assert tokenizer.eos_token == "<|endoftext|>"
    assert tokenizer.eos_token_id == 151643

    mock_config = SimpleNamespace(
        model=SimpleNamespace(name="Qwen/Qwen3-0.6B-Base"),
        data=SimpleNamespace(column="text", chat_template=None, eos_token="<|im_end|>"),
    )

    tokenizer = get_tokenizer(mock_config)

    assert tokenizer.eos_token == "<|im_end|>"
    assert tokenizer.eos_token_id == 151645

    text = """And blood-black nothingness began to spin
A system of cells interlinked within
Cells interlinked within cells interlinked
Within one stem. And dreadfully distinct
Against the dark, a tall white fountain played."""

    results = _encode({"text": [text]}, tokenizer, "text")
    maybe_eos_token = results["input_ids"][0][-1]

    assert maybe_eos_token == tokenizer.eos_token_id


def test_encode_messages():
    """Test _encode function with messages column."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    mock_config = SimpleNamespace(
        model=SimpleNamespace(name="Qwen/Qwen3-0.6B"),
        data=SimpleNamespace(column="messages", chat_template=Path("tests/test.jinja").read_text(), eos_token=None),
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

    results = _encode({"messages": [messages]}, tokenizer, "messages")

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


def test_dataset_cache_reuse():
    """Test that LacunaDataset reuses cache when built with same config."""
    config = SimpleNamespace(
        model=SimpleNamespace(name="Qwen/Qwen3-0.6B-Base"),
        data=SimpleNamespace(
            datasets=["keatone/TinierStories"],
            split="train[:10]",
            column="text",
            num_workers=1,
            chat_template=None,
            eos_token=None,
            stream=False,
            map_batch_size=4,
            pack_batch_size=10,
        ),
    )
    config.trainer = SimpleNamespace(
        batch_size=4,
        seq_len=128,
        seed=42,
    )

    dataset1 = LacunaDataset(config)
    fingerprint1 = config.data.fingerprint
    dataset2 = LacunaDataset(config)
    fingerprint2 = config.data.fingerprint

    assert fingerprint1 == fingerprint2
    assert len(dataset1._dataset) == len(dataset2._dataset)

    config.trainer.seq_len = 256
    dataset3 = LacunaDataset(config)
    fingerprint3 = config.data.fingerprint

    assert fingerprint1 != fingerprint3
    assert len(dataset3._dataset) != len(dataset1._dataset)
