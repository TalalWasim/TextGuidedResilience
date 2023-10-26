NETWORK=resnet50 # also resnet18, resnet34, resnet101, resnet152
exp_dir=results/${NETWORK}_clip
mkdir -p "${exp_dir}"

torchrun --nproc_per_node=4 train_clip.py \
    --data-path '/path/to/imagenet/' \
    --workers 12 \
    --model ${NETWORK} \
    --output-dir "${exp_dir}" \
    --batch-size 180 \
    --CLIP_text_path './imagenet_text_features/gpt3.pth' \
    --embed_size 512 \
2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"