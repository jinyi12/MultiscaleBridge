import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from functools import wraps
import numpy as np
from einops import rearrange, repeat
from torch.amp import autocast, GradScaler


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        nonlocal cache
        if not _cache:
            return f(*args, **kwargs)
        if cache is not None:
            return cache
        result = f(*args, **kwargs)
        cache = result
        return result

    return cached_fn


def make_grid(res):
    gridx = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    gridx = gridx.reshape(1, res, 1, 1).repeat([1, 1, res, 1])
    gridy = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    gridy = gridy.reshape(1, 1, res, 1).repeat([1, res, 1, 1])
    grid = torch.cat((gridx, gridy), dim=-1).reshape(1, -1, 2)
    return grid


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale)

    def forward(self, x):
        assert x.dim() == 3 and x.size(2) == self._num_input_channels, (
            f"Expected input shape is [batch_size, num_input_channels]: [{x.shape[0]}, {self._num_input_channels}], but got {x.shape}"
        )

        batches, num_of_points, channels = x.shape
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, "b n c -> (b n) c")

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, "(b n) c -> b n c", b=batches)

        x = 2 * np.pi * x

        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        # Cache for frequently used timesteps
        self.embedding_cache = {}
        self.max_cache_size = 1000  # Limit cache size to prevent memory issues

    @staticmethod
    def _compute_timestep_embedding(t, dim, max_period=10000):
        """
        Raw computation of sinusoidal timestep embeddings without caching.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings with caching for efficiency.

        During training with diffusion models, we often use the same timesteps
        repeatedly, so caching makes sense. During inference, timesteps change
        but there are usually fewer of them.

        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # Only cache scalar timesteps
        if t.shape[0] == 1 or (t[0] == t).all():
            # Use the first element as key since all values are the same
            key = (float(t[0].item()), dim, max_period, t.device)

            if key in self.embedding_cache:
                # If all timesteps are the same and we have it cached, return cached value
                cached_embedding = self.embedding_cache[key]
                if cached_embedding.shape[0] != t.shape[0]:
                    return cached_embedding.repeat(t.shape[0], 1)
                return cached_embedding

            # Compute embedding
            embedding = self._compute_timestep_embedding(t, dim, max_period)

            # Manage cache size
            if len(self.embedding_cache) >= self.max_cache_size:
                # Simple strategy: clear the entire cache when it gets too big
                self.embedding_cache = {}

            # Cache the result
            self.embedding_cache[key] = embedding
            return embedding
        else:
            # For varied timesteps, don't use cache
            return self._compute_timestep_embedding(t, dim, max_period)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class QuerieEmbedder(nn.Module):
    """
    Embeds target grids into vector representations.
    """

    def __init__(self, hidden_size, pos_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            act_layer=approx_gelu,
            drop=0,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

        self.proj = nn.Linear(pos_dim, hidden_size, bias=True)

    def forward(self, grid, c):
        grid = self.proj(grid)

        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
        queries = grid + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm1(grid), shift_mlp, scale_mlp)
        )
        return queries


class FlashAttention(nn.Module):
    """
    FlashAttention: Efficient multi-head self-attention using PyTorch's Flash Attention implementation.

    This module implements high-performance attention with:
    1. Memory-efficient scaled dot-product attention using PyTorch's built-in FlashAttention-2
    2. Automatic fallback to a chunked attention implementation when Flash Attention isn't available
    3. Multi-head attention mechanisms with parallel computation

    Flash Attention optimizes memory usage by avoiding materializing the full attention matrix,
    making it suitable for processing longer sequences or using larger batch sizes.
    """

    def __init__(self, query_dim, context_dim=None, num_heads=8, dropout=0.0):
        """
        Initialize the FlashAttention module.

        Args:
            query_dim: Dimension of the query input
            context_dim: Dimension of the context input (defaults to query_dim)
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.dim_head = int(query_dim // num_heads)
        inner_dim = query_dim
        context_dim = default(context_dim, query_dim)

        self.scale = self.dim_head**-0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=True)

        self.dropout = dropout
        self.to_out = nn.Linear(inner_dim, query_dim)

        # Check if flash attention is available
        self.has_flash_attn = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        """
        Apply self-attention to the input tensor.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, query_dim]

        Returns:
            output: Self-attention output of shape [batch_size, sequence_length, query_dim]
        """
        h = self.heads

        # Project inputs to queries, keys, and values
        q = self.to_q(x)
        context = x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Reshape for multi-head attention
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # Use PyTorch's Flash Attention if available, otherwise use memory-efficient implementation
        if self.has_flash_attn:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn_output = self.memory_efficient_attention(q, k, v)

        # Reshape and project to output dimension
        out = rearrange(attn_output, "b h n d -> b n (h d)")
        return self.to_out(out)


class FlashCrossAttention(nn.Module):
    """
    FlashCrossAttention: Efficient multi-head cross-attention using PyTorch's Flash Attention implementation.

    This module implements cross-attention between queries and contexts with:
    1. Memory-efficient scaled dot-product attention using PyTorch's built-in FlashAttention-2
    2. Automatic fallback to a chunked attention implementation when Flash Attention isn't available
    3. Multi-head attention mechanisms with parallel computation

    Cross-attention allows the model to attend to a context tensor based on query representations,
    essential for many transformer architectures that need to process conditional information.
    """

    def __init__(self, query_dim, context_dim=None, num_heads=8, dropout=0.0):
        """
        Initialize the FlashCrossAttention module.

        Args:
            query_dim: Dimension of the query input
            context_dim: Dimension of the context input (defaults to query_dim)
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.dim_head = int(query_dim // num_heads)
        inner_dim = query_dim
        context_dim = default(context_dim, query_dim)

        self.scale = self.dim_head**-0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=True)

        self.dropout = dropout
        self.to_out = nn.Linear(inner_dim, query_dim)

        # Check if flash attention is available
        self.has_flash_attn = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, context=None):
        """
        Apply cross-attention between query and context tensors.

        Args:
            x: Query tensor of shape [batch_size, query_sequence_length, query_dim]
            context: Optional context tensor of shape [batch_size, context_sequence_length, context_dim]
                     If None, defaults to self-attention using x

        Returns:
            output: Cross-attention output of shape [batch_size, query_sequence_length, query_dim]
        """
        h = self.heads

        # Project queries from input
        q = self.to_q(x)
        # Default context to input if not provided (self-attention)
        context = default(context, x)
        # Project context to keys and values
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Reshape for multi-head attention
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # Use PyTorch's Flash Attention if available, otherwise use memory-efficient implementation
        if self.has_flash_attn:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn_output = self.memory_efficient_attention(q, k, v)

        # Reshape and project to output dimension
        out = rearrange(attn_output, "b h n d -> b n (h d)")
        return self.to_out(out)


class AttnBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning, using Flash Attention.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = FlashAttention(hidden_size, num_heads=num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class CrossAttnBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning, using Flash Attention.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = FlashCrossAttention(
            hidden_size, num_heads=num_heads, **block_kwargs
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 8 * hidden_size, bias=True)
        )

    def forward(self, queries, context, c):
        (
            shift_msa_q,
            scale_msa_q,
            shift_msa_c,
            scale_msa_c,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(8, dim=1)
        queries = queries + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(queries), shift_msa_q, scale_msa_q),
            modulate(self.norm2(context), shift_msa_c, scale_msa_c),
        )
        queries = queries + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(queries), shift_mlp, scale_mlp)
        )
        return queries


class FinalLayer(nn.Module):
    """
    The final layer of the model.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class OperatorTransformer(nn.Module):
    """
    OperatorTransformer: A transformer-based model for processing multidimensional data.

    This model implements a transformer architecture with:
    1. Gaussian Fourier Feature mapping for positional encoding
    2. Self-attention and cross-attention mechanisms with Flash Attention
    3. Time step conditioning for dynamic control
    4. Query-based processing through latent space

    The model processes input data and positions through a series of attention blocks,
    allowing for complex transformations of spatial/temporal input signals.
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
        Initialize the OperatorTransformer model.

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
            height: Height dimension of the spatial grid
        """
        super().__init__()

        self.locals = [
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
        ]

        self.num_heads = num_heads
        cross_heads = num_heads
        self.input_axis = in_channel
        self.hidden_size = latent_dim

        # Gaussian Fourier Feature Transform for 2D positions
        self.gfft = GaussianFourierFeatureTransform(2, latent_dim // 2, scale=scale)
        self.latents = nn.Parameter(torch.randn(latent_dim, latent_dim))

        # Create embedder for input data (called x_embedder in transformer.py)
        self.x_embedder = nn.Linear(out_channel, latent_dim, bias=True)

        # Rename for consistency with transformer.py
        self.time_embedder = TimestepEmbedder(latent_dim)
        self.query_embedder = QuerieEmbedder(latent_dim, latent_dim)

        # Restructure to match the encoder-decoder block structure in transformer.py
        self.enc_block = nn.ModuleList([])
        for _ in range(depth_enc):
            enc_attns = nn.ModuleList(
                [AttnBlock(latent_dim, num_heads) for _ in range(self_per_cross_attn)]
            )
            self.enc_block.append(
                nn.ModuleList([CrossAttnBlock(latent_dim, cross_heads), enc_attns])
            )

        self.dec_block = nn.ModuleList([])
        for _ in range(depth_dec):
            dec_attns = nn.ModuleList(
                [AttnBlock(latent_dim, num_heads) for _ in range(self_per_cross_attn)]
            )
            self.dec_block.append(
                nn.ModuleList([CrossAttnBlock(latent_dim, cross_heads), dec_attns])
            )

        self.final_layer = FinalLayer(latent_dim, out_channel)
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize all model weights with appropriate distributions"""

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embedder weights
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Initialize query embedder
        nn.init.constant_(self.query_embedder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.query_embedder.adaLN_modulation[-1].bias, 0)

        # Zero-out adaLN modulation layers in encoder blocks
        for block in self.enc_block:
            cross_attn, attns = block[0], block[-1]
            nn.init.constant_(cross_attn.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(cross_attn.adaLN_modulation[-1].bias, 0)
            for l in range(len(attns)):
                nn.init.constant_(attns[l].adaLN_modulation[-1].weight, 0)
                nn.init.constant_(attns[l].adaLN_modulation[-1].bias, 0)

        # Zero-out adaLN modulation layers in decoder blocks
        for block in self.dec_block:
            cross_attn, attns = block[0], block[-1]
            nn.init.constant_(cross_attn.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(cross_attn.adaLN_modulation[-1].bias, 0)
            for l in range(len(attns)):
                nn.init.constant_(attns[l].adaLN_modulation[-1].weight, 0)
                nn.init.constant_(attns[l].adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def ckpt_wrapper(self, module):
        """Wrapper for checkpoint functionality to enable gradient checkpointing"""

        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(self, data, t, input_pos=None, output_pos=None):
        """
        Forward pass through the OperatorTransformer.

        Args:
            data: Input tensor of shape [batch_size, channels, height, width]
            t: Time step tensor of shape [batch_size]
            input_pos: Optional input positions, if None, uses a generated grid
            output_pos: Optional output positions (unused in this implementation)

        Returns:
            output: Processed output tensor of shape [batch_size, out_channels, height, width]
        """
        # Step 1: Rearrange input data to have channels as the last dimension
        data = rearrange(data, "b c h w -> b h w c")

        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == 2, "input data must have the right number of dimensions"

        # Step 2: Flatten spatial dimensions
        data = rearrange(data, "b ... c -> b (...) c")

        # Step 3: Generate or use provided position encodings
        if input_pos is None:
            grid = make_grid(axis[0])
            grid = repeat(grid, "b hw c -> (repeat b) hw c", repeat=b)
            grid = grid.to(data.device)
            input_pos = output_pos = self.gfft(grid)
        else:
            # Use provided positions
            input_pos = input_pos
            output_pos = output_pos if output_pos is not None else input_pos

        # Step 4: Apply x_embedder to data and add positional encoding
        input_emb = self.x_embedder(data) + input_pos
        t_emb = self.time_embedder(t)

        # Step 5: Generate queries from output positions
        queries = self.query_embedder(output_pos, t_emb)

        # Step 6: Initialize latent representation
        x = repeat(self.latents, "n d -> b n d", b=b)
        c = t_emb

        # Step 7: Process through encoder blocks
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

        # Step 8: Process through decoder blocks
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

        # Step 9: Generate final output through the final layer
        out = self.final_layer(queries, c)

        # Step 10: Reshape output to expected format [batch, channels, height, width]
        out = rearrange(out, "b (h w) c -> b c h w", h=axis[0], w=axis[1])
        return out


@cache_fn
def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(
                d_model
            )
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class OperatorTransformer_1d(nn.Module):
    """
    OperatorTransformer_1d: A transformer-based model for processing 1D data sequences.

    This model implements a transformer architecture specialized for 1D data with:
    1. Gaussian Fourier Feature mapping for positional encoding
    2. Self-attention and cross-attention mechanisms with Flash Attention
    3. Time step conditioning for dynamic control
    4. Sequence-aware processing through latent space

    The model processes 1D input sequences using positional encoding and
    attention mechanisms, allowing for complex transformations of sequential data.
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
        Initialize the OperatorTransformer_1d model.

        Args:
            in_channel: Number of input channels
            out_channel: Number of output channels
            pos_dim: Dimension of positional encodings
            latent_dim: Dimension of latent representations
            num_heads: Number of attention heads
            depth_enc: Depth of encoder (unused in this implementation)
            depth_dec: Depth of decoder
            scale: Scale parameter for Gaussian Fourier features (unused now)
            self_per_cross_attn: Number of self-attention blocks per cross-attention block
            height: Length of the sequence (used for positional encoding)
        """
        super().__init__()

        self.locals = [
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
        ]

        self.num_heads = num_heads
        cross_heads = num_heads
        self.input_axis = in_channel
        self.hidden_size = latent_dim
        # GFFT is no longer used for positional encoding
        self.height = height

        # Add input embedder similar to transformer.py
        self.x_embedder = nn.Linear(in_channel, latent_dim)

        # Project combined context (time embedding + input data) to the latent dimension
        self.context_projection = nn.Linear(latent_dim + in_channel, latent_dim)

        in_sizes = [latent_dim * 2, latent_dim, in_channel]

        self.query_embedder = QuerieEmbedder(latent_dim, pos_dim)
        self.time_embedder = TimestepEmbedder(latent_dim)

        # Build the decoder blocks: self-attention followed by cross-attention
        cross_blocks = []
        for i in range(depth_dec):
            blocks = []
            for j in range(self_per_cross_attn):
                blocks.extend([AttnBlock(latent_dim, num_heads)])
            blocks.extend([CrossAttnBlock(latent_dim, cross_heads)])
            cross_blocks.extend(blocks)
        self.decoder_blocks = nn.ModuleList(cross_blocks)

        self.final_layer = FinalLayer(latent_dim, out_channel)
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize all model weights with appropriate distributions"""

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize context projection layer
        nn.init.normal_(self.context_projection.weight, std=0.02)
        nn.init.constant_(self.context_projection.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Initialize query embedder MLP:
        nn.init.normal_(self.query_embedder.mlp_pos_emb[0].weight, std=0.02)
        nn.init.normal_(self.query_embedder.mlp_pos_emb[2].weight, std=0.02)
        nn.init.normal_(self.query_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.query_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.decoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def ckpt_wrapper(self, module):
        """Wrapper for checkpoint functionality to enable gradient checkpointing"""

        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(self, data, t, input_pos=None, output_pos=None):
        """
        Forward pass through the OperatorTransformer_1d.

        Args:
            data: Input tensor of shape [batch_size, channels, sequence_length]
            t: Time step tensor of shape [batch_size]
            input_pos: Optional input positions (unused in this implementation)
            output_pos: Optional output positions (unused in this implementation)

        Returns:
            output: Processed output tensor of shape [batch_size, out_channels, sequence_length]
        """
        # Rearrange data from [batch, channels, sequence] to [batch, sequence, channels]
        data = rearrange(data, "b c l -> b l c")
        latent_dim = self.hidden_size

        # Step 1: Generate timestep embeddings for conditioning
        t_emb = self.time_embedder(t)
        batch_size, seq_length, feature_size = data.shape
        device = data.device

        # Step 2: Create positional encodings
        # Generate 1D positional encodings similar to transformer.py
        pe = positionalencoding1d(latent_dim, seq_length).to(device)
        position_encodings = rearrange(pe, "n d -> 1 n d").repeat(batch_size, 1, 1)

        # Step 3: Combine data with positional encoding
        # Apply input embedding and add positional encoding
        input_emb = self.x_embedder(data) + position_encodings

        # Step 4: Set up queries with positional encoding
        queries = input_emb.clone()

        # Step 5: Process through transformer blocks with optional checkpointing
        # Use gradient checkpointing for transformer blocks during training
        use_checkpointing = self.training and torch.is_grad_enabled()

        for i, block in enumerate(self.decoder_blocks):
            if isinstance(block, AttnBlock):
                # Self-attention refines the query representation
                if use_checkpointing:
                    queries = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block),
                        queries,
                        t_emb,
                        use_reentrant=False,
                    )
                else:
                    queries = block(queries, t_emb)
            elif isinstance(block, CrossAttnBlock):
                # Cross-attention incorporates contextual information
                if use_checkpointing:
                    queries = torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block),
                        queries,
                        input_emb,  # Use input_emb as context instead of GFFT-processed
                        t_emb,
                        use_reentrant=False,
                    )
                else:
                    queries = block(queries, input_emb, t_emb)
            else:
                raise ValueError(f"Unknown block type: {type(block)}")

        # Final output processing
        z = self.final_layer(queries, t_emb)

        # Rearrange back to [batch, channels, sequence]
        z_out = rearrange(z, "b l c -> b c l")
        return z_out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def jit_compile_model(model):
    """
    Apply TorchScript JIT compilation to model for faster inference
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping JIT compilation")
        return model

    try:
        # Try to JIT compile the model
        model.eval()  # Set to eval mode
        # Use TorchScript to compile the model
        scripted_model = torch.jit.script(model)
        print("Successfully JIT compiled model")
        return scripted_model
    except Exception as e:
        print(f"JIT compilation failed: {e}")
        print("Falling back to original model")
        return model


if __name__ == "__main__":
    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = OperatorTransformer_1d(
        in_channel=1,
        out_channel=1,
        latent_dim=512,
        pos_dim=1,
        num_heads=4,
        depth_enc=6,
        depth_dec=1,
        scale=1,
        self_per_cross_attn=1,
        height=64,
    ).to(device)

    print("number of parameters is {}".format(count_parameters(model)))
    x = torch.rand(batch_size, 100, 1).to(device)
    t = torch.rand(
        batch_size,
    ).to(device)

    # Example of using mixed precision for inference
    with autocast(enabled=torch.cuda.is_available(), device_type=device):
        yy = model(x, t)
    print(yy.shape)

    # Example of how you would use it in training
    if torch.cuda.is_available():
        print("Mixed precision training example:")
        scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # In training loop:
        optimizer.zero_grad()
        with autocast(device_type=device):
            output = model(x, t)
            # loss = criterion(output, target)

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

    # Example of JIT compilation for inference
    print("\nTesting JIT compilation for inference:")
    model.eval()

    # Compile with TorchScript JIT for faster inference
    # Note: might not work for all models due to dynamic control flow
    try:
        compiled_model = jit_compile_model(model)
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            # Warm-up run
            _ = compiled_model(x, t)

            # Timed run
            start_time.record()
            y_compiled = compiled_model(x, t)
            end_time.record()

            torch.cuda.synchronize()
            print(
                f"JIT compiled model inference time: {start_time.elapsed_time(end_time):.3f} ms"
            )

            # Compare with original model
            start_time.record()
            with torch.no_grad():
                y_original = model(x, t)
            end_time.record()

            torch.cuda.synchronize()
            print(
                f"Original model inference time: {start_time.elapsed_time(end_time):.3f} ms"
            )

            # Verify outputs are the same
            if torch.allclose(y_compiled, y_original, rtol=1e-3, atol=1e-3):
                print("✓ JIT compilation successful - outputs match")
            else:
                print("✗ Warning: JIT compilation changed model outputs")
    except Exception as e:
        print(f"JIT compilation test failed: {e}")
