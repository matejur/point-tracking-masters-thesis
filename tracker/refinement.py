import torch
from torch import nn
from tracker.utils.utils import extract_bilinear_regions
from tracker.transformer import RefineTransformer
from einops import rearrange


class CorrelationEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels_list, kernel_shapes, strides):
        super(CorrelationEmbedding, self).__init__()

        in_channels_list = [in_channels] + list(out_channels_list[:-1])
        layers = []
        for in_ch, out_ch, kernel_shape, stride in zip(
            in_channels_list, out_channels_list, kernel_shapes, strides
        ):
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_shape, stride=stride),
                    nn.GroupNorm(out_ch // 16, out_ch),
                    nn.ReLU(),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, query_regions, guess_regions):
        assert query_regions.ndim == 5, "Input should be of shape (B, N, C, H1, W1)"
        assert guess_regions.ndim == 5, "Input should be of shape (B, N, C, H2, W2)"

        corr_volume = torch.einsum(
            "bnchw, bncij -> bnhwij", query_regions, guess_regions
        )

        branch1 = rearrange(corr_volume, "b n h w i j -> (b n) (h w) i j")
        branch2 = rearrange(corr_volume, "b n h w i j -> (b n) (i j) h w")

        branch1 = self.net(branch1)
        branch2 = self.net(branch2)

        branch1 = torch.mean(branch1, dim=(2, 3))
        branch2 = torch.mean(branch2, dim=(2, 3))

        B = query_regions.shape[0]
        branch1 = rearrange(branch1, "(b n) c -> b n c", b=B)
        branch2 = rearrange(branch2, "(b n) c -> b n c", b=B)

        embedding = torch.cat([branch1, branch2], dim=-1)

        return embedding


def sinusoidal_embedding_table(length, dim):
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * -(torch.log(torch.tensor(10000.0)) / dim)
    )

    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class Refinement(nn.Module):
    def __init__(
        self,
        region_size=11,
        add_query_frame_token=True,
        add_positional_encoding=True,
    ):
        super(Refinement, self).__init__()
        self.region_size = region_size
        self.add_query_frame_token = add_query_frame_token
        self.add_positional_encoding = add_positional_encoding

        embedding_params = dict(
            in_channels=region_size * region_size,
            out_channels_list=[64, 128, 128],
            kernel_shapes=[3, 3, 2],
            strides=[1, 2, 2],
        )

        self.embedding_net = CorrelationEmbedding(**embedding_params)

        embedding_channels = embedding_params["out_channels_list"][-1] * 2
        self.refinement_token = nn.Parameter(torch.randn(1, 1, 1, embedding_channels))
        self.ground_truth_token = nn.Parameter(torch.randn(1, 1, embedding_channels))

        self.register_buffer(
            "positional_embedding",
            sinusoidal_embedding_table(16, embedding_channels)
        )

        self.refine_transformer = RefineTransformer(
            input_channels=embedding_channels,
            output_channels=embedding_channels,
            num_heads=6,
            num_layers=3,
            dim=384,
        )

        self.pos_occ_net = nn.Linear(embedding_channels, 3)

    def forward(self, query_coords, guess_coords, desc1, desc2, frame_deltas):

        assert query_coords.ndim == 4, "query_coords should be of shape (B, A, N, 2)"
        assert guess_coords.ndim == 3, "guess_coords should be of shape (B, N, 2)"
        assert desc1.ndim == 5, "desc1 should be of shape (B, A, H, W, C)"
        assert desc2.ndim == 5, "desc2 should be of shape (B, A, H, W, C)"
        assert frame_deltas.ndim == 2, "frame_deltas should be of shape (B, A)"
        
        frame_deltas[frame_deltas > 15] = 15

        B, A, N, _ = query_coords.shape
        guess_coords_expanded = guess_coords.view(B, 1, N, 2).expand(B, A, N, 2)

        assert query_coords.shape == guess_coords_expanded.shape, (
            "query_coords and guess_coords should have the same shape"
        )
        assert desc1.shape == desc2.shape, "desc1 and desc2 should have the same shape"
        assert query_coords.shape[0] == desc1.shape[0], "Batch size mismatch"
        assert query_coords.shape[1] == desc1.shape[1], "Number of anchors mismatch"
        assert frame_deltas.shape[0] == desc1.shape[0], "Batch size mismatch"
        assert frame_deltas.shape[1] == desc1.shape[1], "Number of anchors mismatch"

        frame_deltas = torch.concatenate(
            [
                frame_deltas,
                torch.zeros(
                    (B, 1), dtype=frame_deltas.dtype, device=frame_deltas.device
                ),
            ],
            dim=1,
        )

        H, W = desc1.shape[2:4]

        desc1 = desc1.view(B * A, H, W, -1)
        desc2 = desc2.view(B * A, H, W, -1)
        query_coords = query_coords.reshape(B * A, N, 2)
        guess_coords_expanded = guess_coords_expanded.reshape(B * A, N, 2)

        query_region = extract_bilinear_regions(desc1, query_coords, self.region_size)
        guess_region = extract_bilinear_regions(
            desc2, guess_coords_expanded, self.region_size
        )

        embeddings = self.embedding_net(query_region, guess_region)

        tokens = embeddings.view(B, A, N, -1)

        if self.add_query_frame_token:
            # Let the transformer know that the first token is special, always 100% correct
            tokens[:, 0] += self.ground_truth_token

        refinement_tokens = self.refinement_token.expand(
            tokens.shape[0], 1, tokens.shape[2], self.refinement_token.shape[-1]
        )
        tokens = torch.cat([tokens, refinement_tokens], dim=1)

        if self.add_positional_encoding:
            pe = self.positional_embedding[frame_deltas].unsqueeze(2)
            tokens = tokens + pe

        tokens = rearrange(tokens, "b a n c -> (b n) a c")
        tokens = self.refine_transformer(tokens)
        tokens = rearrange(tokens, "(b n) a c -> b a n c", b=B)

        embeddings = tokens[:, -1]
        
        pos_occ = self.pos_occ_net(embeddings)

        pos_correction = pos_occ[:, :, :2]
        refined_coords = guess_coords + pos_correction

        occ_logits = pos_occ[:, :, 2]

        return refined_coords, occ_logits
