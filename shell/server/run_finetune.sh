# /raidWorkspace/mingu/Data
# D:/Dropbox/Data
echo "Experiments Started"
GPUS=0

PRETRAINED_FILE_PRE="checkpoints/wm811k/supmoco/resnet18/"
PRETRAINED_FILE_SUF="/ckpt.last.pth.tar"

for HASH in "2021-10-27_12-57-58" "2021-10-27_12-57-58" "2021-10-27_12-57-58"
do
    PRETRAINED_FILE="${PRETRAINED_FILE_PRE}${HASH}${PRETRAINED_FILE_SUF}"

    python ./run_classification.py \
    --gpus $GPUS \
    --pretrained_file $PRETRAINED_FILE \
    --root /raidWorkspace/mingu/Data/wm811k \
    --data wm811k \
    --input_size 64 \
    --backbone_type resnet18 \
    --epochs 100 \
    --batch_size 512 \
    --num_workers 4 \
    --optimizer sgd \
    --learning_rate 0.03 \
    --weight_decay 0 \
    --cosine_warmup 0 \
    --cosine_cycles 1 \
    --cosine_min_lr 0.0 \
    --save_every 50 \
    --projector_dim 128 \
    --pretrained_task supmoco \
    --loss_function ce \
    --enable_wandb \
    --mixed_precision
done
echo "Finished."
