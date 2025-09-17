OUTPUT_DIR="<output_directory>"

mkdir -p "$OUTPUT_DIR"

cp "$0" "$OUTPUT_DIR/script.sh"

python train.py \
    --refiner "Refinement(region_size=11, add_query_frame_token=True, add_positional_encoding=True)" \
    --num_anchors 4 \
    --mast3r_weights "<dynamic_master_weights_path>" \
    --train_dataset "5000 @ KubricSeq(root='<kubric_root>', split='train', num_tracks=512)" \
    --test_dataset "250 @ KubricSeq(root='<kubric_root>', split='validation', num_tracks=512)"  \
    --lr 0.0001 \
    --min_lr 1e-6 \
    --warmup_epochs 2 \
    --batch_size 16 \
    --epochs 50 \
    --num_workers 8 \
    --disable_cudnn_benchmark \
    --output_dir "$OUTPUT_DIR"
