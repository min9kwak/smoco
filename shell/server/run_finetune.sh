# /raidWorkspace/mingu/Data
# D:/Dropbox/Data
echo "Experiments Started"
SERVER=dgx
GPUS=0

EPOCHS=100
OPTIMIZER=sgd
LEARNING_RATE=0.03

PRETRAINED_FILE_PRE="checkpoints/pet-moco-global/resnet/"

for HASH in "2021-10-27_12-57-58" "2021-10-27_12-57-58" "2021-10-27_12-57-58"
do
    PRETRAINED_DIR="${PRETRAINED_FILE_PRE}${HASH}"

    python ./run_finetune.py \
    --gpus $GPUS \
    --server $SERVER \
    --epochs $EPOCHS \
    --optimizer $OPTIMIZER \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0 \
    --cosine_warmup 0 \
    --cosine_cycles 1 \
    --cosine_min_lr 0.0 \
    --save_every 200 \
    --enable_wandb \
    --pretrained_dir $PRETRAINED_DIR \
    --freeze_bn \
    --balance
done
echo "Finished."
