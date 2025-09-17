import os
import json
import time
import torch
import argparse
import numpy as np
from pathlib import Path

from tracker.utils import path_to_mast3r  # noqa: F401
from mast3r.model import AsymmetricMASt3R
from tracker.refinement import Refinement
from tracker.tracker import Tracker
from benchmarks.tapvid.evaluation_datasets import (
    compute_tapvid_metrics,
    create_davis_dataset,
    create_kinetics_dataset,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tracker on TAPVID Davis.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["tapvid_davis", "tapvid_kinetics"],
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the file/folder.",
    )
    parser.add_argument(
        "--mast3r_weights",
        type=str,
        required=True,
        help="Path to the MAST3R weights file.",
    )
    parser.add_argument(
        "--refiner_weights",
        type=str,
        required=True,
        help="Path to the Refiner weights file.",
    )
    parser.add_argument(
        "--num_anchors",
        type=int,
        required=True,
        help="Number of anchors points.",
    )
    parser.add_argument(
        "--use_guess_coords",
        type=str2bool,
        required=True,
    )
    parser.add_argument(
        "--refiner_iter",
        type=int,
        required=True,
        help="Number of iterations for the refiner.",
    )
    parser.add_argument(
        "--display_progress",
        action="store_true",
        help="Display progress during tracking.",
    )
    parser.add_argument(
        "--query_mode",
        type=str,
        required=True,
        choices=["first", "strided"],
    )
    parser.add_argument(
        "--processing_size",
        type=int,
        required=True,
        help="Size to which the video frames will be resized for processing.",
    )
    parser.add_argument(
        "--anchor_selection_strategy",
        type=str,
        required=True,
        choices=["last_n_visible", "last_n_visible_plus_query", "only_query"],
    )
    parser.add_argument(
        "--add_query_frame_token",
        type=str2bool,
        required=True,
    )
    parser.add_argument(
        "--add_positional_encoding",
        type=str2bool,
        required=True,
    )
    parser.add_argument(
        "--region_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--use_gt_occlusion",
        type=str2bool,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the evaluation results.",
    )

    return parser.parse_args()


def load_models(args):
    mast3r = AsymmetricMASt3R.from_pretrained(args.mast3r_weights).eval().cuda()

    refiner = (
        Refinement(
            region_size=args.region_size,
            add_query_frame_token=args.add_query_frame_token,
            add_positional_encoding=args.add_positional_encoding,
        )
        .eval()
        .cuda()
    )
    refiner.load_state_dict(torch.load(args.refiner_weights))

    return mast3r, refiner


def experiment_folder(args):
    output_dir = Path(args.output_dir)

    output_dir /= args.dataset
    output_dir /= f"{args.processing_size}x{args.processing_size}_processing_size"
    output_dir /= args.query_mode
    output_dir /= "use_gt_occlusion" if args.use_gt_occlusion else "no_gt_occlusion"
    output_dir /= args.anchor_selection_strategy
    output_dir /= f"{args.num_anchors}_anchors"
    output_dir /= (
        "guess_coords"
        if args.use_guess_coords
        else f"refined_coords_{args.refiner_iter}_iter_{args.region_size}_region"
        + ("_no_token" if not args.add_query_frame_token else "")
        + ("_no_pos_enc" if not args.add_positional_encoding else "")
    )

    (output_dir / "trajs").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)

    return output_dir


def main():
    args = parse_args()

    output_folder = experiment_folder(args)
    print("Starting evaluation:", output_folder)

    if "davis" in args.dataset:
        dataset = create_davis_dataset(
            args.dataset_path,
            args.query_mode,
        )
    elif "kinetics" in args.dataset:
        dataset = create_kinetics_dataset(
            args.dataset_path,
            args.query_mode,
        )

    if args.anchor_selection_strategy == "only_query":
        args.num_anchors = 1

    mast3r, refiner = load_models(args)
    tracker = Tracker(
        mast3r,
        refiner,
        num_anchors=args.num_anchors,
        anchor_selection_strategy=args.anchor_selection_strategy,
        use_refined_coords=not args.use_guess_coords,
        refiner_iterations=args.refiner_iter,
        processing_img_size=args.processing_size,
        display_progress=args.display_progress,
        max_batch_size=8,
    )

    final_metrics = {}

    for i, video in enumerate(dataset):
        video_name = list(video.keys())[0]
        data = video[video_name]

        # Kinetics videos don't have names
        if "kinetics" in args.dataset:
            # evaluate only on first 100 videos
            if i == 100:
                break

            video_name += f"_{i}"

        if os.path.exists(output_folder / "metrics" / f"{video_name}.json"):
            print(f"Loading {video_name}, already evaluated.")

            with open(output_folder / "metrics" / f"{video_name}.json", "r") as f:
                metrics = json.load(f)

            final_metrics[video_name] = metrics

            continue

        print(f"Tracking video: {video_name}")

        video = ((data["video"] + 1) * 127.5).astype(np.uint8)[0]
        query_points = data["query_points"][0]
        gt_occluded = data["occluded"][0]

        start = time.time()
        trajs, occluded = tracker.track(
            video,
            query_points,
            gt_occluded=gt_occluded if args.use_gt_occlusion else None,
        )
        end = time.time()

        metrics = compute_tapvid_metrics(
            query_points=data["query_points"],
            gt_occluded=data["occluded"],
            gt_tracks=data["target_points"],
            pred_occluded=occluded[None],
            pred_tracks=trajs[None],
            query_mode="first",
        )

        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                metrics[k] = v[0]
                
        metrics["duration"] = end - start
        metrics["inferences"] = tracker.image_pair_inferences

        with open(output_folder / "metrics" / f"{video_name}.json", "w") as f:
            json.dump(metrics, f, indent=4)

        np.savez(
            output_folder / "trajs" / f"{video_name}.npz",
            trajs=trajs,
            occluded=occluded,
        )

        final_metrics[video_name] = metrics
        print(f"Metrics for {video_name}:")
        print(
            f"Time: {end - start:.2f}s,",
            f"Inferences: {tracker.image_pair_inferences},",
            f"AJ: {metrics['average_jaccard']:.2f},",
            f"AVG PTS: {metrics['average_pts_within_thresh']:.2f},",
            f"OA: {metrics['occlusion_accuracy']:.2f}\n",
        )

    OAs = [m["occlusion_accuracy"] for m in final_metrics.values()]
    AJs = [m["average_jaccard"] for m in final_metrics.values()]
    AVG_PTS = [m["average_pts_within_thresh"] for m in final_metrics.values()]
    durations = [m["duration"] for m in final_metrics.values()]
    inferences = [m["inferences"] for m in final_metrics.values()]

    final_metrics["average"] = {
        "occlusion_accuracy": np.mean(OAs),
        "average_jaccard": np.mean(AJs),
        "average_pts_within_thresh": np.mean(AVG_PTS),
        "average_time": np.mean(durations),
        "average_inferences": int(np.mean(inferences)),
    }

    print(f"Total evaluation time: {np.sum(durations):.2f}s")
    print(f"Total inferences: {int(np.sum(inferences))}")
    print("Average:")
    print(
        f"Time: {final_metrics['average']['average_time']:.2f}s,",
        f"Inferences: {final_metrics['average']['average_inferences']:.2f},",
        f"AJ: {final_metrics['average']['average_jaccard']:.2f},",
        f"AVG PTS: {final_metrics['average']['average_pts_within_thresh']:.2f},",
        f"OA: {final_metrics['average']['occlusion_accuracy']:.2f}\n",
    )

    final_metrics["others"] = {
        "mast3r_weights": args.mast3r_weights,
        "refiner_weights": args.refiner_weights,
        "total_inferences": int(np.sum(inferences)),
        "total_seconds": float(np.sum(durations)),
    }

    with open(output_folder / "results.json", "w") as f:
        json.dump(final_metrics, f, indent=4)


if __name__ == "__main__":
    main()
