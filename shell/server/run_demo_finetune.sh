# /raidWorkspace/mingu/Data
# D:/Dropbox/Data
echo "Experiments Started"
SERVER=dgx
GPUS=0

EPOCHS=10
OPTIMIZER=adamw
LEARNING_RATE=0.0001

FINETUNE_TRANS=test

PRETRAINED_FILE_PRE="checkpoints/pet-supmoco/resnet/"
HASH="2022-06-11_15-53-34"

for HASH in "2022-07-03_13-41-32" "2022-07-03_13-37-29" "2022-07-03_13-33-23" "2022-07-03_13-29-10" "2022-07-03_13-25-05" "2022-07-03_13-21-00" "2022-07-03_13-16-54" "2022-07-03_13-12-44" "2022-07-03_13-08-35" "2022-07-03_13-04-32"
do
  for LEARNING_RATE in 0.0001
  do
    for EPOCHS in 10
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
