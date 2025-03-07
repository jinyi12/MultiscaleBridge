import torch
import torch.nn as nn
from einops import rearrange, repeat

from models.transformer_flash_attention import (
    OperatorTransformer,
    OperatorTransformer_1d,
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
        # Rearrange input data to have channels as the last dimension
        data = rearrange(data, "b c h w -> b h w c")

        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype

        # Convert direction to tensor indices
        dir_idx = 0 if direction == "fwd" else 1
        dir_idx = torch.tensor([dir_idx] * t.shape[0], device=t.device)
        dir_emb = self.direction_embedding(dir_idx)

        latent_dim = self.hidden_size

        # Step 1: Generate timestep embeddings with direction conditioning
        t_emb_base = self.time_embedder(t)
        # Add direction information to timestep embedding
        t_emb = t_emb_base + dir_emb

        batch_size, h, w, data_chans = data.shape
        device = data.device

        # Step 2: Generate or use provided position encodings
        if input_pos is None:
            grid = make_grid(axis[0])
            grid = repeat(grid, "b hw c -> (repeat b) hw c", repeat=b)
            grid = grid.to(data.device)
            input_pos = output_pos = self.gfft(grid)
        else:
            raise NotImplementedError("Not implemented")

        # # Step 3: Apply Gaussian Fourier Feature Transform to positions
        # context = self.gfft(rearrange(inp_pos, "b n c -> (b n) c"))
        # context = rearrange(context, "(b n) c -> b n c", b=batch_size)
        context = input_pos

        # Step 4: Combine positional encoding with projected input data
        flattened_data = rearrange(data, "b ... c -> b (...) c")
        input_emb = self.x_embedder(flattened_data) + context

        # Step 5: Generate query embeddings with direction conditioning
        queries = self.query_embedder(output_pos, t_emb)

        # Step 6: make repeated random latents
        x = repeat(self.latents, "n d -> b n d", b=b)
        c = t_emb

        # Step 7: Process through transformer blocks with optional checkpointing
        use_checkpointing = self.training and torch.is_grad_enabled()

        # Process through encoder blocks
        for block in self.enc_block:
            cross_attn, attns = block[0], block[-1]
            x = (
                torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(cross_attn), x, input_emb, c
                )
                if self.training
                else cross_attn(x, input_emb, c)
            )

            for l in range(len(attns)):
                x = (
                    torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(attns[l]), x, c)
                    if self.training
                    else attns[l](x, c)
                )

        # Process through decoder blocks
        for block in self.dec_block:
            cross_attn, attns = block[0], block[-1]
            queries = (
                torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(cross_attn), queries, x, c
                )
                if self.training
                else cross_attn(queries, x, c)
            )

            for l in range(len(attns)):
                queries = (
                    torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(attns[l]), queries, c
                    )
                    if self.training
                    else attns[l](queries, c)
                )

        # Step 8: Generate final output through the appropriate head based on direction
        if direction == "fwd":
            z = self.forward_head(queries, t_emb)
        else:
            z = self.backward_head(queries, t_emb)

        # Reshape output to expected format [batch, channels, height, width]
        z_out = rearrange(z, "b (h w) c -> b c h w", h=h, w=w)
        return z_out


class BM2Transformer1D(OperatorTransformer_1d):
    """
    BM2Transformer1D: A 1D version of the BM2Transformer for handling 1D data sequences.

    This model extends OperatorTransformer_1d to support direction conditioning
    (forward/backward) by injecting an additional learned direction embedding into
    the timestep embeddings. The backbone logic (input embedding, positional encoding,
    encoder/decoder processing, and final projection) is identical to that of
    OperatorTransformer_1d in transformer.py, ensuring consistency in the data flow.
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
            in_channel: Number of input channels.
            out_channel: Number of output channels.
            pos_dim: Dimension of positional encodings.
            latent_dim: Dimension of latent representations.
            num_heads: Number of attention heads.
            depth_enc: Depth of encoder.
            depth_dec: Depth of decoder.
            scale: (Unused) Scale parameter.
            self_per_cross_attn: Number of self-attention blocks per cross-attention block.
            height: Length of the sequence (used for positional encoding).
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

        # --- Direction Conditioning ---
        # Add a direction embedding to distinguish forward ("fwd") and backward ("bwd") directions.
        self.direction_embedding = nn.Embedding(2, latent_dim)

        # Create separate final output heads for forward and backward directions.
        # The forward head is kept identical to the parent's final_layer for compatibility.
        self.forward_head = self.final_layer
        self.backward_head = FinalLayer(latent_dim, out_channel)

        # Initialize the direction embedding and backward head.
        nn.init.normal_(self.direction_embedding.weight, std=0.02)
        self._initialize_backward_head()

    def _initialize_backward_head(self):
        """Initialize the backward head with the same initialization as the forward head."""
        nn.init.constant_(self.backward_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.backward_head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.backward_head.linear.weight, 0)
        nn.init.constant_(self.backward_head.linear.bias, 0)

    def forward(self, data, t, direction="fwd", input_pos=None, output_pos=None):
        """
        Forward pass through the BM2Transformer1D.

        Args:
            data: Input tensor of shape [batch_size, channels, sequence_length].
            t: Time step tensor of shape [batch_size].
            direction: String flag ("fwd" or "bwd") for forward or backward diffusion.
            input_pos: Optional input positions (unused; positional encodings are generated internally).
            output_pos: Optional output positions (ignored if not provided).

        Returns:
            z_out: Output tensor of shape [batch_size, out_channels, sequence_length].
        """
        # Convert direction flag to tensor indices and embed:
        dir_idx = 0 if direction == "fwd" else 1
        dir_idx = torch.tensor([dir_idx] * t.shape[0], device=t.device)
        dir_emb = self.direction_embedding(dir_idx)

        # Rearrange input data to have channels as the last dimension.
        data = rearrange(data, "b c l -> b l c")
        batch_size, seq_length, _ = data.shape

        latent_dim = self.hidden_size  # latent_dim

        # Step 1: Generate timestep embeddings with directional conditioning.
        t_emb_base = self.time_embedder(t)
        t_emb = t_emb_base + dir_emb

        # Step 2: Create 1D positional encodings.
        pe = positionalencoding1d(latent_dim, seq_length).to(data.device)
        # The positional encoding is repeated across the batch.
        position_encodings = rearrange(pe, "n d -> 1 n d").repeat(batch_size, 1, 1)

        # Step 3: Combine input data with positional encoding.
        # The x_embedder projects the input data to the latent dimension.
        input_emb = self.x_embedder(data) + position_encodings

        # Step 4: (Optional) For the 1D backbone in transformer.py the queries are usually created
        # by simply cloning the input embeddings. (In some variants, self.q_embedder is used.)
        queries = input_emb.clone()

        # Step 5: Initialize latent representation.
        x = repeat(self.latents, "n d -> b n d", b=batch_size)
        c = t_emb

        # Step 6: Process through encoder blocks.
        for block in self.enc_block:
            cross_attn, attns = block[0], block[-1]
            x = (
                torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(cross_attn), x, input_emb, c
                )
                if self.training
                else cross_attn(x, input_emb, c)
            )
            for attn in attns:
                x = (
                    torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(attn), x, c)
                    if self.training
                    else attn(x, c)
                )

        # Step 7: Process through decoder blocks.
        for block in self.dec_block:
            cross_attn, attns = block[0], block[-1]
            queries = (
                torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(cross_attn), queries, x, c
                )
                if self.training
                else cross_attn(queries, x, c)
            )
            for attn in attns:
                queries = (
                    torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(attn), queries, c
                    )
                    if self.training
                    else attn(queries, c)
                )

        # Step 8: Compute final output through the appropriate head.
        if direction == "fwd":
            z = self.forward_head(queries, t_emb)
        else:
            z = self.backward_head(queries, t_emb)

        # Step 9: Rearrange output to shape [batch, channels, sequence_length] and return.
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
