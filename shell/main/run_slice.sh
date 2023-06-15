echo "Experiments Started"
SERVER=main
GPUS=0

DATA_TYPE=pet
N_SPLITS=10
N_CV=0
IMAGE_SIZE=145
RANDOM_STATE=2021

INTENSITY=scale

EPOCHS=100
BATCH_SIZE=16
OPTIMIZER=adamw
LEARNING_RATE=0.0001

TRAIN_SLICES=sagittal
NUM_SLICES=5
SLICE_RANGE=0.15

for TRAIN_SLICES in random
do
	for EPOCHS in 100
	do
		python ./run_slice_classification.py \
		--gpus $GPUS \
		--server $SERVER \
		--data_type $DATA_TYPE \
		--root D:/data/ADNI \
		--data_info labels/data_info.csv \
		--mci_only \
		--n_splits $N_SPLITS \
		--n_cv $N_CV \
		--image_size $IMAGE_SIZE \
		--small_kernel \
		--random_state $RANDOM_STATE \
		--intensity $INTENSITY \
		--flip \
		--rotate \
		--prob 0.5 \
		--backbone_type resnet50 \
		--train_slices $TRAIN_SLICES \
		--num_slices $NUM_SLICES \
		--slice_range $SLICE_RANGE \
		--epochs $EPOCHS \
		--batch_size $BATCH_SIZE \
		--optimizer $OPTIMIZER \
		--learning_rate $LEARNING_RATE \
		--weight_decay 0.0001 \
		--cosine_warmup 0 \
		--cosine_cycles 1 \
		--cosine_min_lr 0.0 \
		--save_every 1000 \
		--enable_wandb \
		--balance
	done
done
echo "Finished."
