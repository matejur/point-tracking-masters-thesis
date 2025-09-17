import cv2
import colorsys
import mediapy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from tracker.utils.utils import softargmax_peaks


def gen_colors(n):
    hsv_tuples = [(x * 1.0 / n, 1, 0.8) for x in range(n)]
    rgb_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    return np.array(rgb_tuples) * 255


def plot_masks(video, masks, out_resolution=(1024, 1024)):
    out_video = video.copy()
    colors = gen_colors(masks.shape[0])

    for i, object in enumerate(masks):
        out_video[object > 0] = video[object > 0] * 0.2 + colors[i] * 0.8

    return mediapy.resize_video(out_video, out_resolution)


def plot_tracks(video, tracks, gt, visible, gt_visible, out_resolution=(1024, 1024), write_track_id=False):
    T, H, W, C = video.shape
    tracks = tracks * np.array(out_resolution) / np.array([W, H])

    if gt is not None:
        gt = gt * np.array(out_resolution) / np.array([W, H])

    out_video = mediapy.resize_video(video, out_resolution)

    for i, frame in enumerate(out_video):
        for track_idx in range(tracks.shape[0]):
            x, y = tracks[track_idx, i]

            color = (0, 100, 255)
            show_line = True
            if not gt_visible[track_idx, i]:
                if visible[track_idx, i]:
                    color = (255, 140, 0)
                    show_line = False
                else:
                    continue
                # continue
            else:
                if not visible[track_idx, i]:
                    color = (220, 20, 60)

            cv2.circle(
                frame,
                (int(x), int(y)),
                7,
                color,
                -1,
            )


            if gt is not None:
                gtx, gty = gt[track_idx, i]

                if gtx != 0 or gty != 0:
                    cv2.circle(
                        frame,
                        (int(gtx), int(gty)),
                        7,
                        (0, 200, 0),
                        -1,
                    )

                    if show_line:
                        cv2.line(
                            frame,
                            (int(gtx), int(gty)),
                            (int(x), int(y)),
                            (255, 0, 0),
                            2,
                        )

            if write_track_id:
                cv2.putText(
                    frame,
                    str(track_idx),
                    (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 255, 255),
                    2,
                )

    return out_video


def correlation_grid(
    video, correlations, trajecotries, target_points, point_id, zoom_size=None
):
    max_corr_maps = max([len(corrs) for corrs in correlations.values()])
    num_plots = max_corr_maps + 1
    num_frames = len(video)

    if zoom_size is None:
        center = (video[0].shape[1] // 2, video[0].shape[0] // 2)
    else:
        pts = target_points[point_id, :num_frames]
        pts[pts == 0] = np.nan
        center = np.nanmean(
            pts,
            axis=0,
        ).astype(int)

    if isinstance(zoom_size, int):
        size = (zoom_size, zoom_size)
    elif zoom_size is None:
        size = (video[0].shape[1], video[0].shape[0])

    tl = (center[0] - size[0] // 2, center[1] - size[1] // 2)
    br = (center[0] + size[0] // 2, center[1] + size[1] // 2)

    cols = np.ceil(np.sqrt(num_plots)).astype(int)
    rows = np.ceil(num_plots / cols).astype(int)

    original_backend = plt.get_backend()
    plt.switch_backend("agg")

    first_frame = np.nonzero(trajecotries)[1][0]

    out_video = []
    for frame_idx in range(first_frame, num_frames):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5 + 1))
        fig.subplots_adjust(left=0, right=1, top=0.98, bottom=0, wspace=0, hspace=0.1)

        axes = axes.flatten()
        axes[0].imshow(video[frame_idx][tl[1] : br[1], tl[0] : br[0]])
        axes[0].set_title(f"Frame {frame_idx}")

        history = correlations.get(frame_idx, {}).items()
        history = sorted(history, key=lambda x: -x[0])

        for i, (prev_frame_idx, corrs) in enumerate(history):
            delta = frame_idx - prev_frame_idx
            ax = axes[i + 1]
            correlation_map = corrs[[point_id]][:, tl[1] : br[1], tl[0] : br[0]]
            peak = softargmax_peaks(correlation_map)
            axes[0].scatter(
                peak[:, 0],
                peak[:, 1],
                c="y",
                s=100,
                alpha=0.7,
            )
            ax.imshow(correlation_map[0], vmin=0, vmax=1)
            ax.set_title(f"Delta {delta}")
            ax.scatter(
                *(target_points[point_id, frame_idx] - tl),
                c="g",
            )
            ax.scatter(
                peak[:, 0],
                peak[:, 1],
                c="r",
                s=10,
            )

        axes[0].scatter(
            *(target_points[point_id, frame_idx] - tl),
            c="g",
            s=100,
            alpha=0.7,
        )
        axes[0].scatter(
            *(trajecotries[point_id, frame_idx] - tl),
            c="r",
            s=100,
            alpha=0.7,
        )

        for ax in axes:
            ax.axis("off")

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.array(canvas.buffer_rgba())
        buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
        out_video.append(buf)
        plt.close(fig)

    plt.switch_backend(original_backend)

    return out_video


def save_video(path, video):
    mediapy.write_video(path, video, fps=24)
