"""
Test script for BM2Transformer model.

This script tests both the 1D and 2D versions of the BM2Transformer model
to ensure they work correctly with different input shapes.
"""

import torch
import numpy as np
from models.bm2_transformer import init_bm2_model


def test_2d_transformer():
    """Test the 2D BM2Transformer model with 2D data."""
    print("Testing 2D BM2Transformer...")

    # Initialize model with default parameters (2D)
    model = init_bm2_model(
        in_channel=1,
        out_channel=1,
        pos_dim=2,
        latent_dim=128,
        num_heads=4,
        depth_enc=4,
        depth_dec=4,
        scale=10,
        self_per_cross_attn=1,
        height=32,
        dim=2,
    )

    # Create dummy 2D data: [batch_size, channels, height, width]
    batch_size = 2
    data = torch.randn(batch_size, 1, 32, 32)

    # Create time steps
    t = torch.tensor([0.3, 0.7])

    # Test forward direction
    print("Testing forward direction...")
    output_fwd = model(data, t, direction="fwd")
    print(f"Output shape (forward): {output_fwd.shape}")

    # Test backward direction
    print("Testing backward direction...")
    output_bwd = model(data, t, direction="bwd")
    print(f"Output shape (backward): {output_bwd.shape}")

    # Check output shapes
    assert output_fwd.shape == data.shape, (
        f"Expected {data.shape}, got {output_fwd.shape}"
    )
    assert output_bwd.shape == data.shape, (
        f"Expected {data.shape}, got {output_bwd.shape}"
    )

    print("2D BM2Transformer test passed!")
    return True


def test_1d_transformer():
    """Test the 1D BM2Transformer model with 1D data."""
    print("Testing 1D BM2Transformer...")

    # Initialize model with 1D parameters
    model = init_bm2_model(
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
        dim=1,
    )

    # Create dummy 1D data: [batch_size, channels, sequence_length]
    batch_size = 2
    data = torch.randn(batch_size, 1, 64)

    # Create time steps
    t = torch.tensor([0.3, 0.7])

    # Test forward direction
    print("Testing forward direction...")
    output_fwd = model(data, t, direction="fwd")
    print(f"Output shape (forward): {output_fwd.shape}")

    # Test backward direction
    print("Testing backward direction...")
    output_bwd = model(data, t, direction="bwd")
    print(f"Output shape (backward): {output_bwd.shape}")

    # Check output shapes
    assert output_fwd.shape == data.shape, (
        f"Expected {data.shape}, got {output_fwd.shape}"
    )
    assert output_bwd.shape == data.shape, (
        f"Expected {data.shape}, got {output_bwd.shape}"
    )

    print("1D BM2Transformer test passed!")
    return True


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("Starting BM2Transformer tests...")

    # Test both models
    try:
        test_2d_transformer()
        test_1d_transformer()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
