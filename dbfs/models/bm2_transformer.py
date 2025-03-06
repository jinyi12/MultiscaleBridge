import torch
import torch.nn as nn
from einops import rearrange

from .transformer_flash_attention import (
    OperatorTransformer,
    FinalLayer,
    make_grid,
    positionalencoding1d,
)


class BM2Transformer(OperatorTransformer):
    """
    BM2Transformer: A transformer-based model for the BM² algorithm that handles
    both forward and backward drifts with shared backbone weights.

    This model extends OperatorTransformer to:
    1. Support direction conditioning (forward/backward)
    2. Use separate output heads for forward and backward directions
    3. Properly handle gradient flow when using direction-specific paths

    The model maintains a single backbone but specializes the outputs
    based on the direction, allowing for more effective forward/backward
    drift estimation while sharing most parameters.
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        pos_dim,
        latent_dim,
        num_heads,
        depth_enc,
        depth_dec,
        scale,
        self_per_cross_attn,
        height,
    ):
        """
        Initialize the BM2Transformer model.

        Args:
            in_channel: Number of input channels
            out_channel: Number of output channels
            pos_dim: Dimension of positional encodings
            latent_dim: Dimension of latent representations
            num_heads: Number of attention heads
            depth_enc: Depth of encoder (unused in this implementation but kept for compatibility)
            depth_dec: Depth of decoder
            scale: Scale parameter for Gaussian Fourier features
            self_per_cross_attn: Number of self-attention blocks per cross-attention block
            height: Height dimension of the spatial grid
        """
        super().__init__(
            in_channel,
            out_channel,
            pos_dim,
            latent_dim,
            num_heads,
            depth_enc,
            depth_dec,
            scale,
            self_per_cross_attn,
            height,
        )

        # Add direction embedding
        self.direction_embedding = nn.Embedding(2, latent_dim)

        # Create separate output heads for forward and backward directions
        # Keep the original final_layer as the forward head for compatibility
        self.forward_head = self.final_layer
        self.backward_head = FinalLayer(latent_dim, out_channel)

        # Initialize the direction embedding and backward head
        nn.init.normal_(self.direction_embedding.weight, std=0.02)
        self._initialize_backward_head()

    def _initialize_backward_head(self):
        """Initialize the backward head with the same initialization as the forward head"""
        nn.init.constant_(self.backward_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.backward_head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.backward_head.linear.weight, 0)
        nn.init.constant_(self.backward_head.linear.bias, 0)

    def forward(self, data, t, direction="fwd", input_pos=None, output_pos=None):
        """
        Forward pass through the BM2Transformer.

        Args:
            data: Input tensor of shape [batch_size, channels, height, width]
            t: Time step tensor of shape [batch_size]
            direction: Direction flag, either "fwd" or "bwd"
            input_pos: Optional input positions, if None, uses a generated grid
            output_pos: Optional output positions (unused in this implementation)

        Returns:
            output: Processed output tensor of shape [batch_size, out_channels, height, width]
        """
        # Convert direction to tensor indices
        dir_idx = 0 if direction == "fwd" else 1
        dir_idx = torch.tensor([dir_idx] * t.shape[0], device=t.device)
        dir_emb = self.direction_embedding(dir_idx)

        # Rearrange input data to have channels as the last dimension
        data = rearrange(data, "b c h w -> b h w c")
        latent_dim = self.hidden_size

        # Step 1: Generate timestep embeddings with direction conditioning
        t_emb_base = self.time_embedder(t)
        # Add direction information to timestep embedding
        t_emb = t_emb_base + dir_emb

        batch_size, h, w, data_chans = data.shape
        device = data.device

        # Step 2: Generate or use provided position encodings
        if input_pos is None:
            rel_pos = make_grid((h, w)).to(device)
            inp_pos = rearrange(rel_pos, "h w c -> (h w) c").repeat(batch_size, 1, 1)
        else:
            inp_pos = input_pos
            rel_pos = None

        # Step 3: Apply Gaussian Fourier Feature Transform to positions
        context = self.gfft(rearrange(inp_pos, "b n c -> (b n) c"))
        context = rearrange(context, "(b n) c -> b n c", b=batch_size)

        # Step 4: Concatenate position features with input data
        proj_context = []
        proj_context.append(data)
        combined_context = torch.concat([context] + proj_context, dim=-1)

        # Step 5: Project combined context to expected hidden dimension
        context = self.context_projection(combined_context)

        # Step 6: Generate query embeddings with direction conditioning
        queries = self.query_embedder(inp_pos, t_emb)

        # Step 7: Process through transformer blocks with optional checkpointing
        use_checkpointing = self.training and torch.is_grad_enabled()

        for i, block in enumerate(self.decoder_blocks):
            if isinstance(
                block, nn.Module
            ):  # Basic check to ensure it's a valid module
                if hasattr(
                    block, "forward_impl"
                ):  # Custom naming for modules with special forward implementations
                    if use_checkpointing:
                        queries = torch.utils.checkpoint.checkpoint(
                            self.ckpt_wrapper(block),
                            queries,
                            context if hasattr(block, "cross_attn") else None,
                            t_emb,
                            use_reentrant=False,
                        )
                    else:
                        if hasattr(block, "cross_attn"):
                            queries = block(queries, context, t_emb)
                        else:
                            queries = block(queries, t_emb)
                else:  # Default handling
                    if use_checkpointing:
                        if (
                            len(block._forward_pre_hooks) + len(block._forward_hooks)
                            > 0
                        ):
                            # Handle modules with hooks differently
                            queries = (
                                block(queries, context, t_emb)
                                if "context" in block.forward.__code__.co_varnames
                                else block(queries, t_emb)
                            )
                        else:
                            if "context" in block.forward.__code__.co_varnames:
                                queries = torch.utils.checkpoint.checkpoint(
                                    self.ckpt_wrapper(block),
                                    queries,
                                    context,
                                    t_emb,
                                    use_reentrant=False,
                                )
                            else:
                                queries = torch.utils.checkpoint.checkpoint(
                                    self.ckpt_wrapper(block),
                                    queries,
                                    t_emb,
                                    use_reentrant=False,
                                )
                    else:
                        if "context" in block.forward.__code__.co_varnames:
                            queries = block(queries, context, t_emb)
                        else:
                            queries = block(queries, t_emb)

        # Step 8: Generate final output through the appropriate head based on direction
        if direction == "fwd":
            z = self.forward_head(queries, t_emb)
        else:
            z = self.backward_head(queries, t_emb)

        # Reshape output to expected format [batch, channels, height, width]
        z_out = rearrange(z, "b (h w) c -> b c h w", h=h, w=w)
        return z_out


class BM2Transformer1D(OperatorTransformer):
    """
    BM2Transformer1D: A 1D version of the BM2Transformer for handling 1D data sequences.

    This is similar to the BM2Transformer but adapted for 1D data.
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        pos_dim,
        latent_dim,
        num_heads,
        depth_enc,
        depth_dec,
        scale,
        self_per_cross_attn,
        height,
    ):
        """
        Initialize the BM2Transformer1D model.

        Args:
            in_channel: Number of input channels
            out_channel: Number of output channels
            pos_dim: Dimension of positional encodings
            latent_dim: Dimension of latent representations
            num_heads: Number of attention heads
            depth_enc: Depth of encoder (unused but kept for compatibility)
            depth_dec: Depth of decoder
            scale: Scale parameter for Gaussian Fourier features
            self_per_cross_attn: Number of self-attention blocks per cross-attention block
            height: Length of the sequence
        """
        super().__init__(
            in_channel,
            out_channel,
            pos_dim,
            latent_dim,
            num_heads,
            depth_enc,
            depth_dec,
            scale,
            self_per_cross_attn,
            height,
        )

        # Add direction embedding
        self.direction_embedding = nn.Embedding(2, latent_dim)

        # Create separate output heads for forward and backward directions
        # Keep the original final_layer as the forward head for compatibility
        self.forward_head = self.final_layer
        self.backward_head = FinalLayer(latent_dim, out_channel)

        # Initialize the direction embedding and backward head
        nn.init.normal_(self.direction_embedding.weight, std=0.02)
        self._initialize_backward_head()

    def _initialize_backward_head(self):
        """Initialize the backward head with the same initialization as the forward head"""
        nn.init.constant_(self.backward_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.backward_head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.backward_head.linear.weight, 0)
        nn.init.constant_(self.backward_head.linear.bias, 0)

    def forward(self, data, t, direction="fwd", input_pos=None, output_pos=None):
        """
        Forward pass through the BM2Transformer1D.

        Args:
            data: Input tensor of shape [batch_size, channels, sequence_length]
            t: Time step tensor of shape [batch_size]
            direction: Direction flag, either "fwd" or "bwd"
            input_pos: Optional input positions, if None, uses a generated grid
            output_pos: Optional output positions (unused in this implementation)

        Returns:
            output: Processed output tensor of shape [batch_size, out_channels, sequence_length]
        """
        # Convert direction to tensor indices
        dir_idx = 0 if direction == "fwd" else 1
        dir_idx = torch.tensor([dir_idx] * t.shape[0], device=t.device)
        dir_emb = self.direction_embedding(dir_idx)

        # Rearrange input data to have channels as the last dimension
        data = rearrange(data, "b c l -> b l c")
        latent_dim = self.hidden_size

        # Step 1: Generate timestep embeddings with direction conditioning
        t_emb_base = self.time_embedder(t)
        # Add direction information to timestep embedding
        t_emb = t_emb_base + dir_emb

        batch_size, seq_len, data_chans = data.shape
        device = data.device

        # Step 2: Generate or use provided position encodings (1D in this case)
        if input_pos is None:
            # For 1D, we use standard positional encoding
            pos_encoding = positionalencoding1d(latent_dim, seq_len).to(device)
            inp_pos = pos_encoding.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            inp_pos = input_pos

        # Step 3: Apply Gaussian Fourier Feature Transform to positions
        context = self.gfft(rearrange(inp_pos, "b n c -> (b n) c"))
        context = rearrange(context, "(b n) c -> b n c", b=batch_size)

        # Step 4: Concatenate position features with input data
        proj_context = []
        proj_context.append(data)
        combined_context = torch.concat([context] + proj_context, dim=-1)

        # Step 5: Project combined context to expected hidden dimension
        context = self.context_projection(combined_context)

        # Step 6: Generate query embeddings with direction conditioning
        queries = self.query_embedder(inp_pos, t_emb)

        # Step 7: Process through transformer blocks with optional checkpointing
        use_checkpointing = self.training and torch.is_grad_enabled()

        for i, block in enumerate(self.decoder_blocks):
            if isinstance(block, nn.Module):
                if hasattr(block, "forward_impl"):
                    if use_checkpointing:
                        queries = torch.utils.checkpoint.checkpoint(
                            self.ckpt_wrapper(block),
                            queries,
                            context if hasattr(block, "cross_attn") else None,
                            t_emb,
                            use_reentrant=False,
                        )
                    else:
                        if hasattr(block, "cross_attn"):
                            queries = block(queries, context, t_emb)
                        else:
                            queries = block(queries, t_emb)
                else:
                    if use_checkpointing:
                        if (
                            len(block._forward_pre_hooks) + len(block._forward_hooks)
                            > 0
                        ):
                            queries = (
                                block(queries, context, t_emb)
                                if "context" in block.forward.__code__.co_varnames
                                else block(queries, t_emb)
                            )
                        else:
                            if "context" in block.forward.__code__.co_varnames:
                                queries = torch.utils.checkpoint.checkpoint(
                                    self.ckpt_wrapper(block),
                                    queries,
                                    context,
                                    t_emb,
                                    use_reentrant=False,
                                )
                            else:
                                queries = torch.utils.checkpoint.checkpoint(
                                    self.ckpt_wrapper(block),
                                    queries,
                                    t_emb,
                                    use_reentrant=False,
                                )
                    else:
                        if "context" in block.forward.__code__.co_varnames:
                            queries = block(queries, context, t_emb)
                        else:
                            queries = block(queries, t_emb)

        # Step 8: Generate final output through the appropriate head based on direction
        if direction == "fwd":
            z = self.forward_head(queries, t_emb)
        else:
            z = self.backward_head(queries, t_emb)

        # Reshape output to expected format [batch, channels, sequence_length]
        z_out = rearrange(z, "b l c -> b c l")
        return z_out


def init_bm2_model(
    in_channel=2,
    out_channel=1,
    pos_dim=256,
    latent_dim=256,
    num_heads=4,
    depth_enc=6,
    depth_dec=2,
    scale=1,
    self_per_cross_attn=1,
    height=256,
    dim=2,  # Default is 2D
):
    """
    Initialize a BM2 model for either 1D or 2D data.

    Args:
        in_channel: Number of input channels
        out_channel: Number of output channels
        pos_dim: Dimension of positional encodings
        latent_dim: Dimension of latent representations
        num_heads: Number of attention heads
        depth_enc: Depth of encoder
        depth_dec: Depth of decoder
        scale: Scale parameter for Gaussian Fourier features
        self_per_cross_attn: Number of self-attention blocks per cross-attention block
        height: Height/length dimension
        dim: Dimensionality of data (1 or 2)

    Returns:
        A BM2Transformer or BM2Transformer1D model
    """
    if dim == 1:
        return BM2Transformer1D(
            in_channel=in_channel,
            out_channel=out_channel,
            pos_dim=pos_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            depth_enc=depth_enc,
            depth_dec=depth_dec,
            scale=scale,
            self_per_cross_attn=self_per_cross_attn,
            height=height,
        )
    else:
        return BM2Transformer(
            in_channel=in_channel,
            out_channel=out_channel,
            pos_dim=pos_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            depth_enc=depth_enc,
            depth_dec=depth_dec,
            scale=scale,
            self_per_cross_attn=self_per_cross_attn,
            height=height,
        )
