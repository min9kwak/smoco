echo "Experiments Started"
SERVER=workstation
GPUS=0

DATA_TYPE=mri
RANDOM_STATE=2021

INTENSITY=scale

BATCH_SIZE=16
EPOCHS=100

OPTIMIZER=sgd

BACKBONE_TYPE=resnet
ARCH=50

SWA_LEARNING_RATE=0.0001
ALPHA=100.0

for LEARNING_RATE in 0.01
do
	for ALPHA in 500 1000
	do
		python ./run_swa.py \
		--gpus $GPUS \
		--server $SERVER \
		--data_type $DATA_TYPE \
		--root /raidWorkspace/mingu/Data/ADNI \
		--data_info labels/data_info.csv \
		--train_size 0.9 \
		--image_size 96 \
		--random_state $RANDOM_STATE \
		--intensity $INTENSITY \
		--rotate \
		--flip \
		--prob 0.2 \
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
		--save_every 20 \
		--enable_wandb \
		--semi \
		--balance \
		--swa_learning_rate $SWA_LEARNING_RATE \
		--swa_start 0 \
		--mu 1 \
		--alpha $ALPHA \
		--ramp_up 0
	done
done
echo "Finished."
