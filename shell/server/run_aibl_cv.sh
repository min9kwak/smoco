echo "Experiments Started"
SERVER=workstation3
GPUS=5

EPOCHS=10
BATCH_SIZE=8
OPTIMIZER=adamw
LEARNING_RATE=0.00001

FINETUNE_TRANS="test"

RANDOM_STATE=2021
N_SPLITS=5
N_CV=0
TRAIN_MODE="train"

PRETRAINED_FILE_PRE="checkpoints/pet-supmoco_finetune/resnet/"


for HASH in "2023-07-11_17-17-03" "2023-07-11_17-15-39" "2023-07-11_17-12-00" "2023-07-11_17-10-44" "2023-07-11_17-06-59" "2023-07-11_17-06-00" "2023-07-11_17-03-30" "2023-07-11_17-02-29" "2023-07-11_16-59-54" "2023-07-11_16-59-02"
do
  for LEARNING_RATE in 0.00001
  do
    for N_CV in 0
    do
      PRETRAINED_DIR="${PRETRAINED_FILE_PRE}${HASH}"
      python ./run_aibl_cv.py \
      --gpus $GPUS \
      --server $SERVER \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
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
      --finetune_trans $FINETUNE_TRANS \
      --random_state $RANDOM_STATE \
      --n_splits $N_SPLITS \
      --n_cv $N_CV \
      --train_mode $TRAIN_MODE
    done
  done
done
echo "Finished."
