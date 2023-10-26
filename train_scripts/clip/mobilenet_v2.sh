NETWORK=mobilenet_v2
exp_dir=results/${NETWORK}_clip
mkdir -p "${exp_dir}"


torchrun --nproc_per_node=4 train_clip.py \
    --data-path '/path/to/imagenet/' \
    --workers 12 \
    --model ${NETWORK} \
    --output-dir "${exp_dir}" \
    --CLIP_text_path './imagenet_text_features/gpt3.pth' \
    --embed_size 512 \
    --epochs 300 \
    --lr 0.045 \
    --batch-size 180 \
    --wd 0.00004 \
    --lr-step-size 1 \
    --lr-gamma 0.98 \
2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"