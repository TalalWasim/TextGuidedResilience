NETWORK=mobilenet_v2
exp_dir=results/${NETWORK}_baseline
mkdir -p "${exp_dir}"


torchrun --nproc_per_node=4 train_baseline.py \
    --data-path '/path/to/imagenet/' \
    --workers 12 \
    --model ${NETWORK} \
    --output-dir "${exp_dir}" \
    --epochs 300 \
    --lr 0.045 \
    --batch-size 180 \
    --wd 0.00004 \
    --lr-step-size 1 \
    --lr-gamma 0.98 \
2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"