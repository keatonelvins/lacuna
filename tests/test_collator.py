import pyarrow as pa

from lacuna.utils import pack_bfd


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


def test_bfd_packing_drop():
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

    packed = pack_bfd(table, seq_len=4, truncate=False)

    expected_packed_input = [[4, 4, 4, 4], [3, 3, 3, 1], [2, 2]]
    expected_packed_pos = [[0, 1, 2, 3], [0, 1, 2, 0], [0, 1]]
    expected_packed_masks = [[0, 1, 0, 1], [1, 0, 1, 1], [0, 1]]

    assert packed["input_ids"].to_pylist() == expected_packed_input
    assert packed["position_ids"].to_pylist() == expected_packed_pos
    assert packed["assistant_masks"].to_pylist() == expected_packed_masks


def test_bfd_packing_truncate():
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

    table = pa.Table.from_arrays([pa.array(input_ids), pa.array(assistant_masks)], names=["input_ids", "assistant_masks"])

    packed = pack_bfd(table, seq_len=9, context_len=3, truncate=True)

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
