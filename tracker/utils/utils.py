import torch
import mediapy
import torch.nn.functional as F


def extract_bilinear_regions(
    features: torch.Tensor, coords: torch.Tensor, kernel_size: int
) -> torch.Tensor:
    assert features.ndim == 4, "features should be of shape (B, H, W, C)"
    assert coords.ndim == 3, "coords should be of shape (B, N, 2)"
    assert coords.shape[0] == features.shape[0], "Batch size mismatch"

    region = neighbor_grid(coords, kernel_size)

    B, N, Kh, Kw, _ = region.shape

    features = features.permute(0, 3, 1, 2)
    region = region.float()
    region = (
        2 * (region / (torch.tensor(features.shape[2:], device=region.device) - 1)) - 1
    )
    region = region.reshape(B, N, Kh * Kw, 2)

    out = F.grid_sample(
        features,
        region,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    return out.permute(0, 2, 1, 3).reshape(B, N, -1, Kh, Kw)


def neighbor_grid(centers, size):
    N = size // 2
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-N, N + 1), torch.arange(-N, N + 1), indexing="ij"
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).to(centers.device)
    grid = grid + centers[:, :, None, None, :]

    return grid


def subpixel_kernel_similarity(desc1, desc2, pts, kernel_size=1):
    assert pts.ndim == 3, "pts should be of shape (B, N, 2)"
    assert desc1.ndim == 4, "desc1 should be of shape (B, H, W, C)"
    assert desc2.ndim == 4, "desc2 should be of shape (B, H, W, C)"
    assert desc1.shape[0] == desc2.shape[0], "Batch size mismatch"
    assert desc1.shape[0] == pts.shape[0], "Batch size mismatch"

    kernels = extract_bilinear_regions(desc1, pts, kernel_size)

    B, H, W, C = desc1.shape
    N = pts.shape[1]

    half_kernel = kernel_size // 2
    x = desc2.permute(0, 3, 1, 2).reshape(1, B * C, H, W)
    w = kernels.reshape(B * N, C, kernel_size, kernel_size)

    sims = (
        F.conv2d(
            x,
            w,
            padding=half_kernel,
            groups=B,
        )
        / (half_kernel * 2 + 1) ** 2
    ).reshape(B, N, H, W)

    return sims


def softargmax_peaks(sims, temp=20, threshold=5):
    assert sims.ndim == 4, "sims should be of shape (B, N, H, W)"
    B, N, H, W = sims.shape

    softmaxed = sims.reshape(B, N, H * W)
    softmaxed = torch.nn.functional.softmax(softmaxed * temp, dim=-1)
    softmaxed = softmaxed.view(B, N, H, W)

    sims_flat = sims.view(B, N, -1)
    argmax = torch.argmax(sims_flat, dim=-1)

    pos = torch.stack(
        (
            argmax % W,
            argmax // W,
        ),
        dim=-1,
    )

    grid_y, grid_x = torch.meshgrid(
        torch.arange(-threshold, threshold + 1),
        torch.arange(-threshold, threshold + 1),
        indexing="ij",
    )
    coords = torch.stack((grid_x, grid_y), dim=-1).to(pos.device)
    coords = coords + pos[:, :, None, None, :]

    x = coords[:, :, :, :, 0]
    y = coords[:, :, :, :, 1]

    outside_x = torch.logical_or(x < 0, x >= W)
    outside_y = torch.logical_or(y < 0, y >= H)
    outside = torch.logical_or(outside_x, outside_y)

    x = torch.clamp(x, 0, W - 1)
    y = torch.clamp(y, 0, H - 1)
    batch_idx = torch.arange(B, device=pos.device)[:, None, None, None]
    channel_idx = torch.arange(N, device=pos.device)[None, :, None, None]

    regions = softmaxed[batch_idx, channel_idx, y, x]
    regions[outside] = 0

    weighted_sum = torch.sum(
        regions[..., None] * coords,
        dim=(2, 3),
    )

    weights = torch.maximum(
        torch.sum(regions[..., None], dim=(2, 3)),
        torch.tensor(1e-12, device=softmaxed.device),
    )

    return weighted_sum / weights

def efficient_forward(mast3r, anchor_images, target_image):
    """
    This function encodes all images only once
    Default forward pass would encode target image A times (once for each anchor image)
    """
    assert anchor_images.ndim == 5, "anchor_images should be of shape (B, A, C, H, W)"
    assert target_image.ndim == 4, "target_image should be of shape (B, C, H, W)"

    B, A, C, H, W = anchor_images.shape

    anchor_images_batches = anchor_images.reshape(B * A, C, H, W)
    images_to_encode = torch.cat((anchor_images_batches, target_image), dim=0)

    feats, pos = mast3r.encode_images(images_to_encode)

    anchor_feats = feats[: B * A]
    anchor_pos = pos[: B * A]
    target_feats = feats[B * A :]
    target_pos = pos[B * A :]

    target_feats = target_feats.repeat_interleave(A, dim=0)
    target_pos = target_pos.repeat_interleave(A, dim=0)

    return mast3r.decode_and_heads(
        anchor_feats,
        anchor_pos,
        target_feats,
        target_pos,
        torch.tensor(anchor_images.shape[-2:])[None].repeat(B, 1),
    )


def resize_for_model(video, query_points, out_resolution):
    assert video.ndim == 4, "video must be a 4D tensor (T, H, W, C)"
    assert query_points.ndim == 2 and query_points.shape[1] == 3, (
        "query_points must be a 2D tensor (N, 3), in the format [t, y, x]"
    )

    # Resize query points
    query_points = query_points.copy()
    query_points[:, 1:] = (
        query_points[:, 1:] / torch.tensor(video.shape[1:3])
    ) * torch.tensor(out_resolution)

    # Resize video
    video = mediapy.resize_video(
        video,
        out_resolution,
    )

    return video, query_points

def revert_resize(traj, model_resolution, in_resolution):
    traj = traj.clone()
    traj = (
        traj / torch.tensor(model_resolution, device=traj.device)
    ) * torch.tensor(in_resolution, device=traj.device)
    return traj

def infere_pairs(model, encoded_video, encoded_pos, anchors, target_frame, shape):
    A = len(anchors)
    anchor_feats = encoded_video[:, anchors]
    anchor_pos = encoded_pos[:, anchors]

    target_feats = encoded_video[:, target_frame].unsqueeze(1).repeat(1, A, 1, 1)
    target_pos = encoded_pos[:, target_frame].unsqueeze(1).repeat(1, A, 1, 1)

    B, A = anchor_feats.shape[:2]
    target_feats = target_feats.reshape(B * A, *target_feats.shape[2:])
    target_pos = target_pos.reshape(B * A, *target_pos.shape[2:])
    anchor_feats = anchor_feats.reshape(B * A, *anchor_feats.shape[2:])
    anchor_pos = anchor_pos.reshape(B * A, *anchor_pos.shape[2:])

    with torch.inference_mode():
        return model.decode_and_heads(
            anchor_feats,
            anchor_pos,
            target_feats,
            target_pos,
            torch.tensor(shape)[None].repeat(B, 1),
        )