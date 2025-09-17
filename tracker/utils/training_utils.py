import torch
from typing import NamedTuple
import torch.nn.functional as F

from tracker.utils.utils import (
    subpixel_kernel_similarity,
    softargmax_peaks,
    infere_pairs,
)


class TrainingPoints(NamedTuple):
    anchor_points: torch.Tensor
    target_points: torch.Tensor
    target_occluded: torch.Tensor
    valid: torch.Tensor


def get_points(all_points, occluded, anchor_frames, target_frame) -> TrainingPoints:
    anchor_points = all_points[:, anchor_frames]
    target_points = all_points[:, target_frame]
    target_occluded = occluded[:, target_frame]

    visible_in_anchors = ~occluded[:, anchor_frames]
    valid = visible_in_anchors.all(dim=1)

    return TrainingPoints(
        anchor_points=anchor_points,
        target_points=target_points,
        target_occluded=target_occluded,
        valid=valid,
    )

class TrainData(NamedTuple):
    huber_loss: torch.Tensor
    occ_loss: torch.Tensor
    guess_error: torch.Tensor
    refined_error: torch.Tensor


def forward_video(mast3r, refiner, batch, max_anchors):
    # Encode all video frames - the encoder is always the same
    video = batch["video"]
    B, N, C, H, W = video.shape
    all_images = video.reshape(-1, *video.shape[2:]).cuda()

    with torch.inference_mode():
        encoded, pos = mast3r.encode_images(all_images)

    encoded_video = encoded.reshape(B, N, *encoded.shape[1:])
    encoded_pos = pos.reshape(B, N, *pos.shape[1:])

    gt_points = batch["points"].cuda()
    gt_occluded = batch["occluded"].cuda()

    predicted_points = gt_points.clone()

    # Go through the video as the tracker would
    for target_frame in range(1, N):
        num_anchors = min(max_anchors, target_frame)

        num_random_anchors = num_anchors - 1

        anchor_candidates = torch.arange(1, target_frame)
        normalized = anchor_candidates / target_frame
        weights = normalized.pow(2)  # Bias a bit towards the nearer frames

        if weights.sum() == 0:
            weights = torch.ones_like(weights)
        else:
            weights = weights / weights.sum()  # Normalize to sum to 1

        samples = (
            torch.multinomial(weights, num_random_anchors, replacement=False)
            if num_random_anchors > 0
            else torch.tensor([])
        )
        samples = samples.sort()[0].long()

        anchor_frames = torch.cat(
            [
                torch.tensor([0]),
                anchor_candidates[samples],
            ]
        ).long()

        # Get all needed pixel descriptors
        pred1, pred2 = infere_pairs(
            mast3r,
            encoded_video,
            encoded_pos,
            anchor_frames,
            target_frame,
            [256, 256],
        )

        desc1 = pred1["desc"]
        desc2 = pred2["desc"]

        training_points = get_points(
            predicted_points, gt_occluded, anchor_frames, target_frame
        )

        # Calculate the position guess
        anchor_points = training_points.anchor_points.reshape(B * num_anchors, -1, 2)
        sims = subpixel_kernel_similarity(desc1, desc2, anchor_points, kernel_size=3)
        anchor_points = anchor_points.reshape(B, num_anchors, -1, 2)

        sims = sims.reshape(B, num_anchors, -1, H, W)
        sims = sims.sum(dim=1)
        position_guess = softargmax_peaks(sims)

        refined_coords, occ_pred = refiner(
            anchor_points,
            position_guess,
            desc1.reshape(B, num_anchors, H, W, -1),
            desc2.reshape(B, num_anchors, H, W, -1),
            target_frame - anchor_frames.repeat(B, 1),
        )

        valid_and_visible = training_points.valid & ~training_points.target_occluded
        valid_target = training_points.target_points[valid_and_visible]
        valid_refined = refined_coords[valid_and_visible]

        huber_loss = F.huber_loss(
            valid_refined,
            valid_target,
            delta=3.5,
            reduction="mean",
        )

        target_occluded = training_points.target_occluded[training_points.valid]
        occ_pred = occ_pred[training_points.valid]

        # occlusion loss
        occluded_sum = target_occluded.sum()
        visible_sum = target_occluded.numel() - occluded_sum

        occ_loss = F.binary_cross_entropy_with_logits(
            occ_pred,
            target_occluded.to(occ_pred.dtype),
            reduction="mean",
            pos_weight=visible_sum / occluded_sum.clamp(min=1.0),
        )

        # The model should learn on its own predictions - noisy data!
        predicted_points[:, target_frame][valid_and_visible] = valid_refined.detach()

        valid_guess = position_guess[valid_and_visible]
        guess_error = torch.norm(
            valid_guess - valid_target,
            dim=-1,
        ).mean()
        refined_error = torch.norm(
            valid_refined - valid_target,
            dim=-1,
        ).mean()

        yield TrainData(
            huber_loss=huber_loss,
            occ_loss=occ_loss,
            guess_error=guess_error,
            refined_error=refined_error,
        )
