import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm.auto import trange
from tracker.utils import utils
from enum import Enum

from .utils import path_to_mast3r  # noqa: F401
import mast3r.utils.path_to_dust3r  # noqa: F401

def last_n_visible_plus_query(num_anchors, occluded, t):
    occluded_before = occluded[:, :t]

    N, _ = occluded_before.shape
    anchor_frames = torch.full((N, num_anchors), -1, dtype=torch.int64)

    for i in range(N):
        visible = (occluded_before[i] == 0).nonzero(as_tuple=False).squeeze(-1)
        if visible.numel() == 0:
            continue
        
        if num_anchors == 1:
            anchors = visible[[0]]
        else:
            first = visible[[0]]
            others = visible[-(num_anchors - 1) :]
            anchors = torch.cat((first, others))

        unique_anchors = torch.unique(anchors)
        num = unique_anchors.numel()
        anchor_frames[i, -num:] = unique_anchors

    return anchor_frames

def last_n_visible(num_anchors, occluded, t):
    occluded_before = occluded[:, :t]

    N, _ = occluded_before.shape
    anchor_frames = torch.full((N, num_anchors), -1, dtype=torch.int64)

    for i in range(N):
        visible = (occluded_before[i] == 0).nonzero(as_tuple=False).squeeze(-1)
        if visible.numel() == 0:
            continue
        anchors = visible[-num_anchors:]
        unique_anchors = torch.unique(anchors)
        num = unique_anchors.numel()
        anchor_frames[i, -num:] = unique_anchors

    return anchor_frames

class AnchorSelectionStrategy(Enum):
    LAST_N_VISIBLE = "last_n_visible"
    LAST_N_VISIBLE_PLUS_QUERY = "last_n_visible_plus_query"

class Tracker:
    def __init__(
        self,
        mast3r,
        refiner,
        num_anchors=4,
        anchor_selection_strategy=AnchorSelectionStrategy.LAST_N_VISIBLE_PLUS_QUERY,
        use_refined_coords=True,
        refiner_iterations=1,
        processing_img_size=512,
        display_progress=False,
        max_batch_size=4,
    ):
        self.mast3r = mast3r
        self.refiner = refiner
        self.num_anchors = num_anchors
        self.use_refined_coords = use_refined_coords
        self.refiner_iterations = refiner_iterations

        self.anchor_selection_strategy = anchor_selection_strategy

        if isinstance(self.anchor_selection_strategy, str):
            self.anchor_selection_strategy = AnchorSelectionStrategy(
                self.anchor_selection_strategy
            )
        

        self.display_progress = display_progress
        self.max_batch_size = max_batch_size
        self.processing_img_size = processing_img_size

        self.norm = T.Compose(
            [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def _prepare(self, video, query_points):
        self.video_size = video.shape[1:3]

        video, query_points = utils.resize_for_model(
            video, query_points, (self.processing_img_size, self.processing_img_size)
        )
        video = torch.stack([self.norm(frame) for frame in video]).cuda()

        video_features = []
        pos = []

        for i in trange(
            0,
            len(video),
            self.max_batch_size,
            desc=f"Encoding video in batches of {self.max_batch_size}",
            disable=not self.display_progress,
        ):
            batch = video[i : i + self.max_batch_size]

            with torch.inference_mode():
                feats_batch, pos_batch = self.mast3r.encode_images(batch)

            video_features.append(feats_batch)
            pos.append(pos_batch)

        video_features = torch.cat(video_features, dim=0).unsqueeze(0)
        pos = torch.cat(pos, dim=0).unsqueeze(0)

        return video_features, pos, query_points
    
    def _get_anchor_frames(self, occluded, t):
        match self.anchor_selection_strategy:
            case AnchorSelectionStrategy.LAST_N_VISIBLE:
                return last_n_visible(self.num_anchors, occluded, t)
            case AnchorSelectionStrategy.LAST_N_VISIBLE_PLUS_QUERY:
                return last_n_visible_plus_query(self.num_anchors, occluded, t)
            case _:
                raise ValueError(
                    f"Unknown anchor selection strategy: {self.anchor_selection_strategy}"
                )

    def track(self, video, query_points, gt_occluded=None):
        assert video.ndim == 4, "video must be a 4D tensor (T, H, W, C)"
        assert query_points.ndim == 2 and query_points.shape[1] == 3, (
            "query_points must be a 2D tensor (N, 3), in the format [t, y, x]"
        )

        # For statistics
        self.image_pair_inferences = 0

        N = len(query_points)
        T = len(video)

        features, pos, query_points = self._prepare(video, query_points)

        query_points = torch.from_numpy(query_points).cuda().float()
        t = query_points[:, 0].int()
        y = query_points[:, 1]
        x = query_points[:, 2]

        # Some evaluations are done without the refinement module, so we rely on ground truth occlusion
        use_gt_occluded = gt_occluded is not None
        if use_gt_occluded:
            print("Using ground truth occlusion!!!")
            occluded = torch.from_numpy(gt_occluded).cuda().bool()
        else:
            occluded = torch.ones((N, T)).cuda().bool()
            occluded[torch.arange(N), t] = False

        point_predictions = torch.zeros((N, T, 2)).float().cuda()
        point_predictions[torch.arange(N), t, 0] = x
        point_predictions[torch.arange(N), t, 1] = y

        for current_frame_index in trange(
            1, T, desc="Tracking frames", disable=not self.display_progress
        ):
            anchor_frames = self._get_anchor_frames(
                occluded, current_frame_index
            )  # (N, N_ANCHORS)

            if use_gt_occluded:
                occluded_this_frame = gt_occluded[:, current_frame_index]
                anchor_frames[occluded_this_frame] = -1

            unique_anchors = torch.unique(anchor_frames[anchor_frames >= 0])

            if unique_anchors.numel() == 0:
                print("No anchors found. This can only happen when using gt_occluded.")
                continue

            self.image_pair_inferences += len(unique_anchors)

            # Get all needed descriptors for current frame prediction
            desc1 = []
            desc2 = []

            for i in range(0, len(unique_anchors), self.max_batch_size):
                batch_indices = unique_anchors[i : i + self.max_batch_size]

                pred1, pred2 = utils.infere_pairs(
                    self.mast3r,
                    features,
                    pos,
                    batch_indices,
                    current_frame_index,
                    [self.processing_img_size, self.processing_img_size],
                )

                desc1.append(pred1["desc"])
                desc2.append(pred2["desc"])

            desc1 = torch.cat(desc1, dim=0)
            desc2 = torch.cat(desc2, dim=0)

            unique_anchor_combinations = torch.unique(anchor_frames, dim=0)
            for anchor_combination in unique_anchor_combinations:
                point_mask = (anchor_frames == anchor_combination).all(dim=1)

                # Get only valid anchor, continue if all -1
                anchor_combination = anchor_combination[anchor_combination >= 0]
                if anchor_combination.numel() == 0:
                    continue

                anchor_points = point_predictions[point_mask][
                    :, anchor_combination
                ].permute(1, 0, 2)

                descriptor_indices = torch.searchsorted(
                    unique_anchors, anchor_combination
                )
                desc_anchors = desc1[descriptor_indices]
                desc_target = desc2[descriptor_indices]

                sims = utils.subpixel_kernel_similarity(
                    desc_anchors, desc_target, anchor_points
                )

                sims = sims.sum(dim=0)
                pos_guess = utils.softargmax_peaks(sims.unsqueeze(0))

                # In this case we do not need to run the refiner
                # Just save the guess position and continue
                if not self.use_refined_coords and use_gt_occluded:
                    point_predictions[point_mask, current_frame_index] = pos_guess.squeeze(0)
                    continue

                pos_refined = pos_guess.clone()
                for i in range(self.refiner_iterations):
                    with torch.inference_mode():
                        pos_refined, occ_logits = self.refiner(
                            anchor_points.unsqueeze(0),
                            pos_refined,
                            desc_anchors.unsqueeze(0),
                            desc_target.unsqueeze(0),
                            (current_frame_index - anchor_combination).unsqueeze(0),
                        )

                pos_refined = pos_refined.squeeze(0)
                occ_logits = occ_logits.squeeze(0)

                point_predictions[point_mask, current_frame_index] = (
                    pos_refined if self.use_refined_coords else pos_guess.squeeze(0)
                )

                if not use_gt_occluded:
                    occ = F.sigmoid(occ_logits)
                    occluded[point_mask, current_frame_index] = occ > 0.5

        # Convert predictions back to original video size
        point_predictions = utils.revert_resize(
            point_predictions,
            (self.processing_img_size, self.processing_img_size),
            self.video_size,
        )

        return point_predictions.cpu().numpy(), occluded.cpu().numpy()
