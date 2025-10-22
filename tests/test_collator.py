import torch
import pyarrow as pa

from lacuna.data import pack, build_inputs


def test_packing():
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

    table = pa.Table.from_arrays(
        [pa.array(input_ids), pa.array(assistant_masks)], names=["input_ids", "assistant_masks"]
    )

    packed = pack(table, seq_len=5)

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


def test_packing_drop():
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

    table = pa.Table.from_arrays(
        [pa.array(input_ids), pa.array(assistant_masks)], names=["input_ids", "assistant_masks"]
    )

    packed = pack(table, seq_len=4, truncate=False)

    expected_packed_input = [[4, 4, 4, 4], [3, 3, 3, 1], [2, 2]]
    expected_packed_pos = [[0, 1, 2, 3], [0, 1, 2, 0], [0, 1]]
    expected_packed_masks = [[0, 1, 0, 1], [1, 0, 1, 1], [0, 1]]

    assert packed["input_ids"].to_pylist() == expected_packed_input
    assert packed["position_ids"].to_pylist() == expected_packed_pos
    assert packed["assistant_masks"].to_pylist() == expected_packed_masks


def test_packing_truncate():
    input_ids = [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
    ]
    assistant_masks = [
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ]

    table = pa.Table.from_arrays(
        [pa.array(input_ids), pa.array(assistant_masks)], names=["input_ids", "assistant_masks"]
    )

    packed = pack(table, seq_len=9, context_len=3, truncate=True)

    expected_packed_input = [
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ]
    expected_packed_pos = [
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
    ]
    expected_packed_masks = [
        [1, 1, 1, 0, 1, 0, 1, 0, 1],
    ]

    assert packed["input_ids"].to_pylist() == expected_packed_input
    assert packed["position_ids"].to_pylist() == expected_packed_pos
    assert packed["assistant_masks"].to_pylist() == expected_packed_masks


def test_build_inputs():
    """Test padding is applied and labels are masked correctly."""
    batch = {
        "input_ids": torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]),
        "position_ids": torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]),
        "assistant_masks": torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
    }

    out_batch = build_inputs(batch, pad_id=0, pad_to=4)

    expected_batch = {
        "input_ids": torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 0, 0]),
        "labels": torch.tensor([-100, -100, 2, -100, 3, 3, -100, -100, -100, -100, -100, -100]),
        "position_ids": torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 0]),
    }

    assert torch.all(out_batch["input_ids"] == expected_batch["input_ids"])
    assert torch.all(out_batch["labels"] == expected_batch["labels"])
    assert torch.all(out_batch["position_ids"] == expected_batch["position_ids"])
