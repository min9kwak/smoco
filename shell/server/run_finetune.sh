# /raidWorkspace/mingu/Data
# D:/Dropbox/Data
echo "Experiments Started"
SERVER=dgx
GPUS=3

EPOCHS=50
OPTIMIZER=sgd
LEARNING_RATE=0.03

PRETRAINED_FILE_PRE="checkpoints/pet-moco-global/resnet/"
HASH="2022-06-11_15-53-34"

for HASH in "2022-06-11_15-53-34" "2022-06-11_15-53-34" "2022-06-11_15-53-34" "2022-06-11_15-53-34" "2022-06-11_15-53-34"
do
  for LEARNING_RATE in 0.03 0.01
  do
    for EPOCHS in 50 100
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
  done
done
echo "Finished."
