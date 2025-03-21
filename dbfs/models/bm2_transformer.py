import torch
import torch.nn as nn
from einops import rearrange, repeat

from .transformer_flash_attention import (
    OperatorTransformer,
    OperatorTransformer_1d,
    FinalLayer,
    make_grid,
    positionalencoding1d,
)

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

allow_ops_in_compiled_graph()


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
        in_axis,
        out_axis,
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
        use_checkpointing=True,  # whether to use checkpointing for forward pass
    ):
        """
        Initialize the BM2Transformer model.

        Args:
            in_axis: Number of input axes, e.g. 2 for 2D data
            out_axis: Number of output axes, e.g. 2 for 2D data
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
            in_axis,
            out_axis,
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
        self.locals = (
            in_axis,
            out_axis,
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

        # # Add direction embedding
        # self.direction_embedding = nn.Embedding(2, latent_dim)

        # # Conditioning projection, takes the direction embedding and the timestep embedding and concatenates them
        # # and projects them to the latent dimension
        # self.conditioning_projection = nn.Linear(2 * latent_dim, latent_dim)

        # Create separate output heads for forward and backward directions
        # Keep the original final_layer as the forward head for compatibility
        self.forward_head = self.final_layer
        self.backward_head = FinalLayer(latent_dim, out_channel)

        # Initialize the backward head
        self._initialize_backward_head()

        self.use_checkpointing = use_checkpointing

    def maybe_checkpoint(self, module, *args):
        """Helper method to optionally apply checkpointing based on configuration"""
        if self.training and self.use_checkpointing and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(module), *args)
        else:
            return module(*args)

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

        assert len(axis) == self.input_axis, (
            "input data must have the right number of axis"
        )

        # Convert direction to tensor indices
        # dir_idx = 0 if direction == "fwd" else 1
        # dir_idx = torch.tensor([dir_idx] * t.shape[0], device=t.device)
        # dir_emb = self.direction_embedding(dir_idx)

        # Step 1: Generate timestep embeddings with direction conditioning
        t_emb_base = self.time_embedder(t)

        # Project the direction embedding and the timestep embedding and concatenate them
        # and project them to the latent dimension
        # conditioning = self.conditioning_projection(
        #     torch.cat([dir_emb, t_emb_base], dim=-1)
        # )

        conditioning = t_emb_base

        batch_size, h, w, data_chans = data.shape
        device = data.device

        # Step 2: Generate or use provided position encodings
        if input_pos is None:
            grid = make_grid(axis[0])
            grid = repeat(grid, "b hw c -> (repeat b) hw c", repeat=b)
            grid = grid.to(data.device)
            # # Step 3: Apply Gaussian Fourier Feature Transform to positions
            input_pos = output_pos = self.gfft(grid)
        else:
            # Apply Gaussian Fourier Feature Transform to positions
            input_pos = output_pos = self.gfft(input_pos)

        context = input_pos

        # Step 4: Combine positional encoding with projected input data
        flattened_data = rearrange(data, "b ... c -> b (...) c")
        # print("flattened_data.shape", flattened_data.shape)
        input_emb = self.x_embedder(flattened_data) + context

        # Step 5: Generate query embeddings with direction conditioning
        queries = self.query_embedder(output_pos, conditioning)

        # Step 6: make repeated random latents
        x = repeat(self.latents, "n d -> b n d", b=b)
        c = conditioning

        # Step 7: Process through transformer blocks with optional checkpointing
        # Process through encoder blocks
        for block in self.enc_block:
            cross_attn, attns = block[0], block[-1]
            x = self.maybe_checkpoint(cross_attn, x, input_emb, c)

            for l in range(len(attns)):
                x = self.maybe_checkpoint(attns[l], x, c)

        # Process through decoder blocks
        for block in self.dec_block:
            cross_attn, attns = block[0], block[-1]
            queries = self.maybe_checkpoint(cross_attn, queries, x, c)

            for l in range(len(attns)):
                queries = self.maybe_checkpoint(attns[l], queries, c)

        # Step 8: Generate final output forward and backward
        z_forward = self.forward_head(queries, conditioning)
        z_backward = self.backward_head(queries, conditioning)

        # Reshape output to expected format [batch, channels, height, width]
        z_forward_out = rearrange(z_forward, "b (h w) c -> b c h w", h=h, w=w)
        z_backward_out = rearrange(z_backward, "b (h w) c -> b c h w", h=h, w=w)
        return z_forward_out, z_backward_out


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
        self.locals = (
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

        # Conditioning projection, takes the direction embedding and the timestep embedding and concatenates them
        # and projects them to the latent dimension
        self.conditioning_projection = nn.Linear(2 * latent_dim, latent_dim)

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

        # Project the direction embedding and the timestep embedding and concatenate them
        # and project them to the latent dimension
        conditioning = self.conditioning_projection(
            torch.cat([dir_emb, t_emb_base], dim=-1)
        )

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
        c = conditioning

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
            z = self.forward_head(queries, conditioning)
        else:
            z = self.backward_head(queries, conditioning)

        # Step 9: Rearrange output to shape [batch, channels, sequence_length] and return.
        z_out = rearrange(z, "b l c -> b c l")
        return z_out
