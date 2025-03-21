"""
Test script for OperatorTransformer_1d models.

This script tests both the original and modified 1D transformer models
to ensure they work correctly with different input shapes.
"""

import torch
import numpy as np
from models.transformer_flash_attention import OperatorTransformer_1d


def test_1d_transformer():
    """Test the 1D transformer with 1D data."""
    print("Testing 1D OperatorTransformer...")

    # Initialize model
    model = OperatorTransformer_1d(
        in_channel=1,
        out_channel=1,
        pos_dim=1,
        latent_dim=128,
        num_heads=4,
        depth_enc=4,
        depth_dec=4,
        scale=10,
        self_per_cross_attn=1,
        height=64,  # sequence length
    )

    # Create dummy 1D data: [batch_size, channels, sequence_length]
    batch_size = 2
    data = torch.randn(batch_size, 1, 64)

    # Create time steps
    t = torch.tensor([0.3, 0.7])

    # Test forward pass
    print("Testing forward pass...")
    output = model(data, t)
    print(f"Output shape: {output.shape}")

    # Check output shape
    expected_shape = (batch_size, 1, 64)  # [batch, channels, sequence]
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )

    print("1D OperatorTransformer test passed!")
    return True


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("Starting OperatorTransformer_1d tests...")

    # Test model
    try:
        test_1d_transformer()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
