echo "Experiments Started"
SERVER=dgx
GPUS=0

DATA_TYPE=pet
N_SPLITS=10
N_CV=0
IMAGE_SIZE=72
RANDOM_STATE=2021

INTENSITY=scale

EPOCHS=100
BATCH_SIZE=8
OPTIMIZER=adamw
LEARNING_RATE=0.0001

BACKBONE_TYPE=resnet
ARCH=50

ALPHA=50
RAMP_UP=10


for RANDOM_STATE in 2021 2023 2025
do
	for ALPHA in 50 100
	do
		python ./run_swa.py \
		--gpus $GPUS \
		--server $SERVER \
		--data_type $DATA_TYPE \
		--root /raidWorkspace/mingu/Data/ADNI \
		--data_info labels/data_info.csv \
		--mci_only \
		--n_splits $N_SPLITS \
		--n_cv $N_CV \
		--image_size $IMAGE_SIZE \
		--random_state $RANDOM_STATE \
		--intensity $INTENSITY \
		--crop_size 64 \
		--small_kernel \
		--flip \
		--rotate \
		--prob 0.5 \
		--backbone_type $BACKBONE_TYPE \
		--arch $ARCH \
		--epochs $EPOCHS \
		--batch_size $BATCH_SIZE \
		--optimizer $OPTIMIZER \
		--learning_rate $LEARNING_RATE \
		--weight_decay 0.0001 \
		--cosine_warmup 0 \
		--cosine_cycles 1 \
		--cosine_min_lr 0.0 \
		--save_every 2000 \
		--semi \
		--semi_loss pi \
		--swa_learning_rate 0.0001 \
		--swa_start 0 \
		--mu 1 \
		--alpha $ALPHA \
		--ramp_up $RAMP_UP \
		--enable_wandb \
		--balance
	done
done
echo "Finished."
