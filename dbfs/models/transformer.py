"""
References:
[1] https://github.com/lucidrains/perceiver-pytorch
[2] DIT: https://github.com/facebookresearch/DiT
[3] OFormer: https://github.com/BaratiLab/OFormer
"""

from math import pi, log, sqrt
from functools import wraps

import numpy as np

from typing import Tuple

import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(
            torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False
        )

    def forward(self, x):
        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, "b n c -> (b n) c")

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, "(b n) c -> b n c", b=batches)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn


def make_grid(res):
    gridx = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    gridx = gridx.reshape(1, res, 1, 1).repeat([1, 1, res, 1])
    gridy = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    gridy = gridy.reshape(1, 1, res, 1).repeat([1, res, 1, 1])
    grid = torch.cat((gridx, gridy), dim=-1).reshape(1, -1, 2)
    return grid


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

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -log(max_period)
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

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

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


class AttnBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, **block_kwargs)
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
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = CrossAttention(hidden_size, num_heads=num_heads, **block_kwargs)
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


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()

        dim_head = int(query_dim // num_heads)
        inner_dim = query_dim

        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)

        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, num_heads=8, dropout=0.0):
        super().__init__()

        dim_head = int(query_dim // num_heads)
        inner_dim = query_dim

        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x):
        h = self.heads

        q = self.to_q(x)

        context = x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        return self.to_out(out)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class OperatorTransformer(nn.Module):
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
        self.gfft = GaussianFourierFeatureTransform(2, latent_dim // 2, scale=scale)
        self.latents = nn.Parameter(torch.randn(latent_dim, latent_dim))

        self.x_embedder = nn.Linear(out_channel, latent_dim, bias=True)
        self.t_embedder = TimestepEmbedder(latent_dim)
        self.q_embedder = QuerieEmbedder(latent_dim, latent_dim)
        # self.q_embedder = nn.Linear(out_channel, latent_dim, bias=True)

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
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.enc_block:
            cross_attn, attns = block[0], block[-1]
            nn.init.constant_(cross_attn.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(cross_attn.adaLN_modulation[-1].bias, 0)
            for l in range(len(attns)):
                nn.init.constant_(attns[l].adaLN_modulation[-1].weight, 0)
                nn.init.constant_(attns[l].adaLN_modulation[-1].bias, 0)

        for block in self.dec_block:
            cross_attn, attns = block[0], block[-1]
            nn.init.constant_(cross_attn.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(cross_attn.adaLN_modulation[-1].bias, 0)
            for l in range(len(attns)):
                nn.init.constant_(attns[l].adaLN_modulation[-1].weight, 0)
                nn.init.constant_(attns[l].adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.q_embedder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.q_embedder.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(self, data, t, input_pos=None, output_pos=None):
        data = rearrange(data, "b c h w -> b h w c")

        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, (
            "input data must have the right number of axis"
        )

        data = rearrange(data, "b ... c -> b (...) c")

        if input_pos is None:
            grid = make_grid(axis[0])
            grid = repeat(grid, "b hw c -> (repeat b) hw c", repeat=b)
            grid = grid.to(data.device)
            input_pos = output_pos = self.gfft(grid)
        else:
            print("Not Implemented")

        input_emb = self.x_embedder(data) + input_pos
        t = self.t_embedder(t)

        queries = self.q_embedder(output_pos, t)
        # queries = self.q_embedder(data) + output_pos

        x = repeat(self.latents, "n d -> b n d", b=b)
        c = t

        for block in self.enc_block:
            cross_attn, attns = block[0], block[-1]
            x = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(cross_attn), x, input_emb, c
            )
            for l in range(len(attns)):
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(attns[l]), x, c)

        for block in self.dec_block:
            cross_attn, attns = block[0], block[-1]
            queries = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(cross_attn), queries, x, c
            )
            for l in range(len(attns)):
                queries = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(attns[l]), queries, c
                )

        # for block in self.enc_block:
        #     cross_attn, attns = block[0], block[-1]
        #     x = cross_attn(x, input_emb, c)
        #     for l in range(len(attns)):
        #         x = attns[l](x, c)

        # for block in self.dec_block:
        #     cross_attn, attns = block[0], block[-1]
        #     queries = cross_attn(queries, x, c)
        #     for l in range(len(attns)):
        #         queries = attns[l](queries, c)

        out = self.final_layer(queries, c)
        out = rearrange(out, "b (h w) c -> b c h w", h=axis[0], w=axis[1])

        return out


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
    div_term = torch.exp(
        (torch.arange(0, d_model, 2, dtype=torch.float) * -(log(10000.0) / d_model))
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class OperatorTransformer_1d(nn.Module):
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
        # self.gfft = GaussianFourierFeatureTransform(2, latent_dim//2, scale=scale)
        self.latents = nn.Parameter(torch.randn(latent_dim, latent_dim))

        self.x_embedder = nn.Linear(out_channel, latent_dim, bias=True)
        self.t_embedder = TimestepEmbedder(latent_dim)
        # self.q_embedder = QuerieEmbedder(latent_dim, latent_dim)
        # self.q_embedder = nn.Linear(out_channel, latent_dim, bias=True)

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
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.enc_block:
            cross_attn, attns = block[0], block[-1]
            nn.init.constant_(cross_attn.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(cross_attn.adaLN_modulation[-1].bias, 0)
            for l in range(len(attns)):
                nn.init.constant_(attns[l].adaLN_modulation[-1].weight, 0)
                nn.init.constant_(attns[l].adaLN_modulation[-1].bias, 0)

        for block in self.dec_block:
            cross_attn, attns = block[0], block[-1]
            nn.init.constant_(cross_attn.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(cross_attn.adaLN_modulation[-1].bias, 0)
            for l in range(len(attns)):
                nn.init.constant_(attns[l].adaLN_modulation[-1].weight, 0)
                nn.init.constant_(attns[l].adaLN_modulation[-1].bias, 0)

        # nn.init.constant_(self.q_embedder.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.q_embedder.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(self, data, t, input_pos=None, output_pos=None):
        # data = rearrange(data, 'b c h w -> b h w c')

        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype

        # data = rearrange(data, 'b ... c -> b (...) c')

        # if input_pos is None:
        #     grid = make_grid(axis[0])
        #     grid = repeat(grid, 'b hw c -> (repeat b) hw c', repeat=b)
        #     grid = grid.to(data.device)
        #     input_pos = output_pos = self.gfft(grid)
        # else:
        #     print("Not Implemented")

        embed = positionalencoding1d(self.hidden_size, data.size(1)).to(device)
        input_emb = self.x_embedder(data) + embed[None]
        t = self.t_embedder(t)

        queries = input_emb + embed[None]
        # queries = self.q_embedder(data) + output_pos

        x = repeat(self.latents, "n d -> b n d", b=b)
        c = t

        for block in self.enc_block:
            cross_attn, attns = block[0], block[-1]
            x = cross_attn(x, input_emb, c)
            for l in range(len(attns)):
                x = attns[l](x, c)

        for block in self.dec_block:
            cross_attn, attns = block[0], block[-1]
            queries = cross_attn(queries, x, c)
            for l in range(len(attns)):
                queries = attns[l](queries, c)

        out = self.final_layer(queries, c)
        # out = rearrange(out, 'b (h w) c -> b c h w', h=axis[0], w=axis[1])

        return out


if __name__ == "__main__":

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = "cuda:0"
    batch_size, h, w = 1, 64, 64
    model = OperatorTransformer_1d(
        in_channel=1,
        out_channel=1,
        latent_dim=512,
        pos_dim=512,
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
    yy = model(x, t)
    print(yy.shape)
