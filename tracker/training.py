# Modified Mast3r/Dust3r training script
import torch
import numpy as np
import argparse
from torch import nn
from pathlib import Path
from typing import Sized
from collections import defaultdict
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

# This adds dynamic_mast3r, dust3r and croco to the path
from tracker.utils import path_to_mast3r  # noqa: F401
from mast3r.utils import path_to_dust3r  # noqa: F401
from dust3r.utils import path_to_croco  # noqa: F401
from dust3r.utils.misc import freeze_all_params

from tracker.refinement import Refinement  # noqa: F401 Used with eval
from dust3r.datasets import get_data_loader
from mast3r.model import AsymmetricMASt3R
from croco.utils import misc
from tracker.utils import training_utils


inf = float("inf")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--refiner", required=True, type=str)
    parser.add_argument("--num_anchors", required=True, type=int)
    parser.add_argument("--mast3r_weights", required=True, type=str)

    parser.add_argument("--train_dataset", required=True, type=str)
    parser.add_argument("--test_dataset", required=True, type=str)

    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--epochs", required=True, type=int)

    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--blr", default=1.5e-4, type=float)
    parser.add_argument("--min_lr", required=True, type=float)
    parser.add_argument("--warmup_epochs", required=True, type=int)

    parser.add_argument("--disable_cudnn_benchmark", action="store_true", default=False)
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--seed", default=0, type=int, help="Random seed")

    return parser.parse_args()


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ["Train", "Test"][test]
    print(f"Building {split} Data loader for dataset:", dataset)
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
    )

    print(f"{split} dataset length: ", len(loader))
    return loader


def load_mast3r(weights_path):
    model = AsymmetricMASt3R(
        pos_embed="RoPE100",
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(256, 256),
        head_type="catmlp+dpt",
        output_mode="pts3d+desc24",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
        two_confs=True,
        desc_conf_mode=("exp", 0, inf),
    )

    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=False)["model"]
    )

    return model


def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_loader = build_dataset(
        args.train_dataset, args.batch_size, args.num_workers, test=False
    )

    test_loader = build_dataset(
        args.test_dataset, args.batch_size, args.num_workers, test=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    print("Loading MASt3R model")
    mast3r = load_mast3r(args.mast3r_weights).to(device).eval()

    print("Creating refinement model")
    refiner = eval(args.refiner).to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    freeze_all_params(mast3r.modules())

    refiner_without_ddp = refiner
    if args.distributed:
        refiner = torch.nn.parallel.DistributedDataParallel(
            refiner,
            device_ids=[args.gpu],
            find_unused_parameters=True,
            static_graph=True,
        )
        refiner_without_ddp = refiner.module

    optimizer = torch.optim.AdamW(
        refiner_without_ddp.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    start_epoch = 0
    best_loss = inf
    resume_path = Path(args.output_dir) / "checkpoint.pth"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
        refiner.load_state_dict(ckpt["model"], strict=False)
        best_loss = ckpt.get("best_loss", inf)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            mast3r,
            refiner,
            train_loader,
            optimizer,
            epoch,
            log_writer,
            args,
        )
        torch.cuda.empty_cache()

        loss = test_one_epoch(
            mast3r,
            refiner,
            test_loader,
            epoch,
            log_writer,
            args,
        )
        torch.cuda.empty_cache()

        if loss < best_loss:
            print("Saving best refiner")
            best_loss = loss
            misc.save_on_master(
                refiner_without_ddp.state_dict(),
                Path(args.output_dir) / "refiner_best.pth",
            )

        # Save checkpoint
        misc.save_on_master(
            {
                "epoch": epoch,
                "model": refiner_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            },
            Path(args.output_dir) / "checkpoint.pth",
        )


def train_one_epoch(
    mast3r: nn.Module,
    refiner: nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    log_writer: SummaryWriter,
    args: argparse.Namespace,
):
    refiner.train()
    mast3r.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    for iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_every, f"Epoch: [{epoch}]")
    ):
        epoch_f = epoch + iter_step / len(data_loader)

        misc.adjust_learning_rate(optimizer, epoch_f, args)

        huber_losses = []
        occ_losses = []
        combined_losses = []
        refined_errors = []
        guess_errors = []

        for batch_data in training_utils.forward_video(
            mast3r, refiner, batch, args.num_anchors
        ):
            huber_loss = batch_data.huber_loss
            occ_loss = batch_data.occ_loss
            refined_error = batch_data.refined_error
            guess_error = batch_data.guess_error

            loss = huber_loss + occ_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            huber_losses.append(huber_loss.item())
            occ_losses.append(occ_loss.item())
            refined_errors.append(refined_error.item())
            guess_errors.append(guess_error.item())

            combined_losses.append(loss.item())

        huber_loss_mean = np.mean(huber_losses)
        occ_loss_mean = np.mean(occ_losses)
        loss_mean = np.mean(combined_losses)

        refined_error = np.mean(refined_errors)
        guess_error = np.mean(guess_errors)
        error_diff = refined_error - guess_error

        metric_logger.update(
            loss=loss_mean, occ_loss=occ_loss_mean, huber_loss=huber_loss_mean
        )
        metric_logger.update(
            refined_error=refined_error, guess_error=guess_error, error_diff=error_diff
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if ((iter_step + 1) % (args.print_every)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_mean)
            huber_loss_value_reduce = misc.all_reduce_mean(huber_loss_mean)
            occ_loss_value_reduce = misc.all_reduce_mean(occ_loss_mean)
            refined_error_reduce = misc.all_reduce_mean(refined_error)
            guess_error_reduce = misc.all_reduce_mean(guess_error)
            error_diff_reduce = misc.all_reduce_mean(error_diff)

            if log_writer is None:
                continue
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar("train/loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar(
                "train/huber_loss", huber_loss_value_reduce, epoch_1000x
            )
            log_writer.add_scalar("train/occ_loss", occ_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("train/guess_error", guess_error_reduce, epoch_1000x)
            log_writer.add_scalar(
                "train/refined_error", refined_error_reduce, epoch_1000x
            )
            log_writer.add_scalar("train/error_diff", error_diff_reduce, epoch_1000x)
            log_writer.add_scalar(
                "train/lr", optimizer.param_groups[0]["lr"], epoch_1000x
            )

    metric_logger.synchronize_between_processes()
    print("Averaged stats: ", metric_logger)


@torch.inference_mode()
def test_one_epoch(
    mast3r: nn.Module,
    refiner: nn.Module,
    data_loader: Sized,
    epoch: int,
    log_writer: SummaryWriter,
    args: argparse.Namespace,
):
    mast3r.eval()
    refiner.eval()

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))

    for iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_every, f"Test Epoch: [{epoch}]")
    ):
        epoch_f = epoch + iter_step / len(data_loader)

        huber_losses = []
        occ_losses = []
        combined_losses = []
        refined_errors = []
        guess_errors = []

        for batch_data in training_utils.forward_video(
            mast3r, refiner, batch, args.num_anchors
        ):
            huber_loss = batch_data.huber_loss
            occ_loss = batch_data.occ_loss
            refined_error = batch_data.refined_error
            guess_error = batch_data.guess_error

            huber_losses.append(huber_loss.item())
            occ_losses.append(occ_loss.item())
            refined_errors.append(refined_error.item())
            guess_errors.append(guess_error.item())

            loss = huber_loss + occ_loss
            combined_losses.append(loss.item())

        huber_loss_mean = np.mean(huber_losses)
        occ_loss_mean = np.mean(occ_losses)
        loss_mean = np.mean(combined_losses)

        refined_error = np.mean(refined_errors)
        guess_error = np.mean(guess_errors)
        error_diff = refined_error - guess_error

        metric_logger.update(
            loss=loss_mean,
            guess_error=guess_error,
            refined_error=refined_error,
            error_diff=error_diff,
            occ_loss=occ_loss_mean,
            huber_loss=huber_loss_mean,
        )

        guess_error_reduce = misc.all_reduce_mean(guess_error)
        refined_error_reduce = misc.all_reduce_mean(refined_error)
        occ_loss_reduce = misc.all_reduce_mean(occ_loss_mean)
        huber_loss_reduce = misc.all_reduce_mean(huber_loss_mean)
        loss_reduce = misc.all_reduce_mean(loss_mean)
        error_diff_reduce = misc.all_reduce_mean(error_diff)

        if log_writer is None:
            continue

        epoch_1000x = int(epoch_f * 1000)
        log_writer.add_scalar("test/guess_error", guess_error_reduce, epoch_1000x)
        log_writer.add_scalar("test/refined_error", refined_error_reduce, epoch_1000x)
        log_writer.add_scalar("test/error_diff", error_diff_reduce, epoch_1000x)
        log_writer.add_scalar("test/occ_loss", occ_loss_reduce, epoch_1000x)
        log_writer.add_scalar("test/huber_loss", huber_loss_reduce, epoch_1000x)
        log_writer.add_scalar("test/loss", loss_reduce, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats: ", metric_logger)

    return metric_logger.meters["loss"].median
