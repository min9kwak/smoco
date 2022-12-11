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

for HASH in "2022-07-02_08-00-31" "2022-07-02_08-00-57" "2022-07-02_09-38-52" "2022-07-02_09-40-42" "2022-07-02_11-17-38" "2022-07-02_11-20-21" "2022-07-02_17-15-14" "2022-07-02_17-15-34" "2022-07-02_18-53-46" "2022-07-02_18-54-27"
do
  for LEARNING_RATE in 0.0001
  do
    for EPOCHS in 10
    do
      PRETRAINED_DIR="${PRETRAINED_FILE_PRE}${HASH}"
      python ./run_demo_finetune.py \
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
