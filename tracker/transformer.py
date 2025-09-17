# Modified transformer from LocoTrack

import torch.nn.functional as F
import torch
import numpy as np
from torch import nn

def get_alibi_slope(num_heads, device="cuda"):
    x = (24) ** (1 / num_heads)
    return torch.tensor(
        [1 / x ** (i + 1) for i in range(num_heads)], device=device, dtype=torch.float32
    ).view(-1, 1, 1)

class MultiHeadAttention(nn.Module):
    """Multi-headed attention (MHA) module."""

    def __init__(
        self,
        num_heads,
        key_size,
        with_bias=True,
        value_size=None,
        model_size=None,
    ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads

        self.with_bias = with_bias

        self.query_proj = nn.Linear(
            num_heads * key_size, num_heads * key_size, bias=with_bias
        )
        self.key_proj = nn.Linear(
            num_heads * key_size, num_heads * key_size, bias=with_bias
        )
        self.value_proj = nn.Linear(
            num_heads * self.value_size, num_heads * self.value_size, bias=with_bias
        )
        self.final_proj = nn.Linear(
            num_heads * self.value_size, self.model_size, bias=with_bias
        )

    def forward(self, query, key, value):
        batch_size, sequence_length, _ = query.size()

        query_heads = self._linear_projection(
            query, self.key_size, self.query_proj
        )  # [T', H, Q=K]
        key_heads = self._linear_projection(
            key, self.key_size, self.key_proj
        )  # [T, H, K]
        value_heads = self._linear_projection(
            value, self.value_size, self.value_proj
        )  # [T, H, V]

        # TODO: add this bias?
        # device = query.device
        # bias_forward = get_alibi_slope(
        #     self.num_heads // 2, device=device
        # )  * get_relative_positions(sequence_length, device=device)
        # bias_forward = bias_forward + torch.triu(
        #     torch.full_like(bias_forward, -1e9), diagonal=1
        # )
        # bias_backward = get_alibi_slope(
        #     self.num_heads // 2, device=device
        # )  * get_relative_positions(sequence_length, reverse=True, device=device)
        # bias_backward = bias_backward + torch.tril(
        #     torch.full_like(bias_backward, -1e9), diagonal=-1
        # )
        # attn_bias = torch.cat([bias_forward, bias_backward], dim=0)

        attn = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            scale=1 / np.sqrt(self.key_size),
        )
        attn = attn.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)

        return self.final_proj(attn)  # [T', D']

    def _linear_projection(self, x, head_size, proj_layer):
        y = proj_layer(x)
        batch_size, sequence_length, _ = x.shape
        return y.reshape(
            (batch_size, sequence_length, self.num_heads, head_size)
        ).permute(0, 2, 1, 3)


class Transformer(nn.Module):
    """A transformer stack."""

    def __init__(
        self, num_heads, num_layers, attn_size, dropout_rate, widening_factor=4
    ):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": MultiHeadAttention(
                            num_heads, attn_size, model_size=attn_size * num_heads
                        ),
                        "dense": nn.Sequential(
                            nn.Linear(
                                attn_size * num_heads,
                                widening_factor * attn_size * num_heads,
                            ),
                            nn.GELU(),
                            nn.Linear(
                                widening_factor * attn_size * num_heads,
                                attn_size * num_heads,
                            ),
                        ),
                        "layer_norm1": nn.LayerNorm(attn_size * num_heads),
                        "layer_norm2": nn.LayerNorm(attn_size * num_heads),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_out = nn.LayerNorm(attn_size * num_heads)

    def forward(self, embeddings):
        h = embeddings
        for layer in self.layers:
            h_norm = layer["layer_norm1"](h)
            h_attn = layer["attn"](h_norm, h_norm, h_norm)
            h_attn = F.dropout(h_attn, p=self.dropout_rate, training=self.training)
            h = h + h_attn

            h_norm = layer["layer_norm2"](h)
            h_dense = layer["dense"](h_norm)
            h_dense = F.dropout(h_dense, p=self.dropout_rate, training=self.training)
            h = h + h_dense

        return self.ln_out(h)
    
class RefineTransformer(nn.Module):
    def __init__(self, input_channels, output_channels, dim=512, num_heads=8, num_layers=1):
        super().__init__()
        self.dim = dim

        self.transformer = Transformer(
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=dim // num_heads,
            dropout_rate=0.,
            widening_factor=4,
        )
        self.input_proj = nn.Linear(input_channels, dim)
        self.output_proj = nn.Linear(dim, output_channels)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.output_proj(x)
