NETWORK=swin_v2_t # also swin_v2_s
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
    --batch-size 256 \
    --opt adamw \
    --lr 0.001 \
    --weight-decay 0.05 \
    --norm-weight-decay 0.0 \
    --bias-weight-decay 0.0 \
    --transformer-embedding-decay 0.0 \
    --lr-scheduler cosineannealinglr \
    --lr-min 0.00001 \
    --lr-warmup-method linear \
    --lr-warmup-epochs 20 \
    --lr-warmup-decay 0.01 \
    --amp \
    --label-smoothing 0.1 \
    --mixup-alpha 0.8 \
    --clip-grad-norm 5.0 \
    --cutmix-alpha 1.0 \
    --random-erase 0.25 \
    --interpolation bicubic \
    --auto-augment ta_wide \
    --model-ema \
    --ra-sampler \
    --ra-reps 4 \
    --val-resize-size 256 \
    --val-crop-size 256 \
    --train-crop-size 256 \
2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"