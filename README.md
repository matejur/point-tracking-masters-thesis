# Point tracking

This repository contains the source code for my master's thesis on point tracking using [MASt3R](https://github.com/naver/mast3r).

You can check out the final thesis [here](https://repozitorij.uni-lj.si/IzpisGradiva.php?id=174662&lang=slv).

## Get started

1. Recursively clone this repository

```bash
git clone git@github.com:matejur/point-tracking-master-thesis.git
```

2. We used the [uv](https://github.com/astral-sh/uv) package manager.
Run the following to download all dependencies.
```bash
uv sync
```

3. Optional, compile the cuda kernels for RoPE.
```bash
cd dust3r/croco/models/curope/
uv run python setup.py build_ext --inplace
cd ../../../../
```

## Checkpoint

Best checkpoints from my thesis can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1vlDBwc7zoqEiIJPe_d4fXaMHROsn7pmo?usp=drive_link).

## Training

If you wish to train the tracker yourself you need:
- DynMASt3R weights from the link above, or train it yourself with the [dynamic_mast3r](/dynamic_mast3r/) submodule
- Panning MOVI-e from the [LocoTrack's repository](https://github.com/cvlab-kaist/locotrack)

Dataloaders require a slighlty different dataset structure, so run `dynamic_master/preprocess_kubric.py`. Check the script for setting required folders.

Finally, modify the provided `train.sh` script with necessary paths and run using `uv run bash train.sh`.

## Evaluation

Use the provided `evaluate.sh` script with `uv run bash evaluate.sh`.
Modify it with correct paths and desired settings.

## Acknowledgement

Please check out the original [MASt3R](https://github.com/naver/mast3r) and [LocoTrack](https://github.com/cvlab-kaist/locotrack) repositories.
