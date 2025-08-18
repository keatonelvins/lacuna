"""Tests for DataCollator."""

import torch

from lacuna.data import DataCollator


class TestDataCollator:
    """Test DataCollator for both packing and padding modes."""

    def setup_method(self):
        """Setup test data and collator."""
        self.pad_token_id = 99999

    def test_collator_with_packing(self):
        """Test that packing mode concatenates sequences without padding."""
        collator = DataCollator(
            pad_token_id=self.pad_token_id,
            packing=True,
        )

        examples = [
            {"input_ids": [1, 2, 3], "labels": [2, 3, 4]},
            {"input_ids": [5, 6], "labels": [6, 7]},
        ]

        result = collator(examples)

        # Should concatenate all sequences into single batch entry
        expected_input_ids = torch.tensor([[1, 2, 3, 5, 6]])
        expected_labels = torch.tensor([[2, 3, 4, 6, 7]])
        expected_position_ids = torch.tensor([[0, 1, 2, 0, 1]])

        torch.testing.assert_close(result["input_ids"], expected_input_ids)
        torch.testing.assert_close(result["labels"], expected_labels)
        torch.testing.assert_close(result["position_ids"], expected_position_ids)

        # Should NOT have attention_mask in packing mode (handled by flash attention)
        assert "attention_mask" not in result

    def test_collator_without_packing(self):
        """Test that non-packing mode pads sequences to max length."""
        collator = DataCollator(
            pad_token_id=self.pad_token_id,
            packing=False,
        )

        examples = [
            {"input_ids": [1, 2, 3], "labels": [2, 3, 4]},
            {"input_ids": [5, 6], "labels": [6, 7]},
        ]

        result = collator(examples)

        # Should pad to max length (3)
        expected_input_ids = torch.tensor([[1, 2, 3], [5, 6, self.pad_token_id]])
        expected_labels = torch.tensor([[2, 3, 4], [6, 7, -100]])
        expected_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        expected_position_ids = torch.tensor([[0, 1, 2], [0, 1, 0]])

        torch.testing.assert_close(result["input_ids"], expected_input_ids)
        torch.testing.assert_close(result["labels"], expected_labels)
        torch.testing.assert_close(result["attention_mask"], expected_attention_mask)
        torch.testing.assert_close(result["position_ids"], expected_position_ids)

    def test_collator_with_packed_seq_lengths(self):
        """Test collator with pre-packed sequences that have seq_lengths."""
        collator = DataCollator(
            pad_token_id=self.pad_token_id,
            packing=True,
        )

        # Simulate packed data from BFD algorithm
        examples = [
            {
                "input_ids": [1, 2, 3, 4, 5],  # Two sequences packed: [1,2,3] + [4,5]
                "labels": [2, 3, 4, 5, 6],
                "seq_lengths": [3, 2],  # Lengths of the two packed sequences
            }
        ]

        result = collator(examples)

        expected_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        expected_labels = torch.tensor([[2, 3, 4, 5, 6]])
        expected_position_ids = torch.tensor([[0, 1, 2, 0, 1]])  # Restart positions

        torch.testing.assert_close(result["input_ids"], expected_input_ids)
        torch.testing.assert_close(result["labels"], expected_labels)
        torch.testing.assert_close(result["position_ids"], expected_position_ids)
