import pyarrow as pa
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from lacuna.utils import pack_bfd
from lacuna.data import LacunaDataset


def test_bfd_packing():
    input_ids = [
        [1],
        [2, 2],
        [3, 3, 3],
        [4, 4, 4, 4],
    ]
    assistant_masks = [
        [1],
        [0, 1],
        [1, 0, 1],
        [0, 1, 0, 1],
    ]

    table = pa.Table.from_arrays([pa.array(input_ids), pa.array(assistant_masks)], names=["input_ids", "assistant_masks"])

    packed = pack_bfd(table, seq_len=5)

    expected_packed_input = [
        [4, 4, 4, 4, 1],
        [3, 3, 3, 2, 2],
    ]
    expected_packed_pos = [
        [0, 1, 2, 3, 0],
        [0, 1, 2, 0, 1],
    ]
    expected_packed_masks = [
        [0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
    ]

    assert packed["input_ids"].to_pylist() == expected_packed_input
    assert packed["position_ids"].to_pylist() == expected_packed_pos
    assert packed["assistant_masks"].to_pylist() == expected_packed_masks


def test_encode_messages():
    """Test _encode method with messages column."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    mock_config = SimpleNamespace(data=SimpleNamespace(column="messages", chat_template=Path("tests/test.jinja").read_text()))

    dataset = object.__new__(LacunaDataset)
    dataset.config = mock_config
    dataset.tokenizer = tokenizer

    messages = [
        {"role": "user", "content": "What's it like to hold the hand of someone you love?"},
        {"role": "assistant", "content": "Within cells interlinked"},
    ]

    expected_text = """<|im_start|>user
What's it like to hold the hand of someone you love?<|im_end|>
<|im_start|>assistant
Within cells interlinked<|im_end|>
"""

    results = dataset._encode({"messages": [messages]})

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
