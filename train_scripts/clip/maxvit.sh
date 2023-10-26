NETWORK=maxvit_t
exp_dir=results/${NETWORK}_clip
mkdir -p "${exp_dir}"

torchrun --nproc_per_node=4 train_clip.py\
    --data-path '/path/to/imagenet/' \
    --workers 12 \
    --model ${NETWORK} \
    --output-dir "${exp_dir}" \
    --CLIP_text_path './imagenet_text_features/gpt3.pth' \
    --embed_size 512 \
    --epochs 400 \
    --batch-size 1024 \
    --opt adamw \
    --lr 3e-3 \
    --weight-decay 0.05 \
    --lr-scheduler cosineannealinglr \
    --lr-min 1e-5 --lr-warmup-method linear \
    --lr-warmup-epochs 32 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.8 \
    --clip-grad-norm 1.0 \
    --interpolation bicubic \
    --auto-augment ta_wide \
    --policy-magnitude 15 \
    --model-ema \
    --val-resize-size 224 \
    --val-crop-size 224 \
    --train-crop-size 224 \
    --amp \
    --model-ema-steps 32 \
    --transformer-embedding-decay 0 \
    --sync-bn \
2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"