import glob
import os.path as osp
import torch
import numpy as np


from dust3r.utils.image import imread_cv2
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.datasets.base.easy_dataset import EasyDataset


class KubricSeq(EasyDataset):
    def __init__(self, root, split, num_tracks=256):
        super().__init__()

        self.video_paths = sorted(glob.glob(osp.join(root, split, "*")))
        self.num_tracks = num_tracks

        print(f"Found {len(self.video_paths)} videos in {osp.join(root, split)}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = sorted(glob.glob(osp.join(video_path, "rgbs", "*.jpg")))

        images = [imread_cv2(frame) for frame in frames]
        images = [ImgNorm(img) for img in images]
        video = torch.stack(images, dim=0)

        tracks = np.load(osp.join(video_path, "tracks.npz"))
        points = tracks["target_points"]
        occluded = tracks["occluded"]

        N, C, H, W = video.shape

        outside = (
            (points[:, :, 0] < 0)
            | (points[:, :, 0] >= W - 1)
            | (points[:, :, 1] < 0)
            | (points[:, :, 1] >= H - 1))
        
        occluded = np.logical_or(occluded, outside)

        visible_in_first = np.where(~occluded[:, 0])[0]

        # Sometimes the first frame has no visible points, rarely happens
        if len(visible_in_first) == 0:
            best = 0
            offset = 0
            for i in range(video.shape[0]):
                curr = (~occluded[:, i]).sum()
                if curr > best:
                    best = curr
                    offset = i
                if curr > 128:
                    break

            video = torch.roll(video, shifts=-offset, dims=0)
            points = np.roll(points, -offset, axis=1)
            occluded = np.roll(occluded, -offset, axis=1)
            visible_in_first = np.where(~occluded[:, 0])[0]

            print(f"Needed to roll, found {len(visible_in_first)} with offset {offset}", force=True)

        if len(visible_in_first) < self.num_tracks:
            # If there are not enough visible points, we can sample with replacement
            indices = np.random.choice(
                visible_in_first, self.num_tracks, replace=True
            )
        else:
            indices = np.random.choice(
                visible_in_first, self.num_tracks, replace=False
            )

        points = torch.tensor(points[indices, :], dtype=torch.float32).permute(1, 0, 2)
        occluded = torch.tensor(occluded[indices, :]).permute(1, 0)

        return {
            "video": video,
            "points": points,
            "occluded": occluded,
        }
