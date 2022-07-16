# /raidWorkspace/mingu/Data
# D:/Dropbox/Data
echo "Experiments Started"
SERVER=workstation2
GPUS=00

EPOCHS=100
OPTIMIZER=adamw
LEARNING_RATE=0.0001

FINETUNE_TRANS=train

PRETRAINED_FILE_PRE="checkpoints/pet-moco/resnet/"
HASH="2022-06-11_15-53-34"

for HASH in "2022-06-11_15-53-34" "2022-06-11_15-53-34" "2022-06-11_15-53-34" "2022-06-11_15-53-34" "2022-06-11_15-53-34"
do
  for LEARNING_RATE in 0.0001
  do
    for EPOCHS in 100 50
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
      --save_every 2000 \
      --enable_wandb \
      --pretrained_dir $PRETRAINED_DIR \
      --balance \
      --finetune_trans $FINETUNE_TRANS
    done
  done
done
echo "Finished."
