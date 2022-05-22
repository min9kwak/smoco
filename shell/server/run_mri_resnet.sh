echo "Experiments Started"
SERVER=dgx
GPUS=0

DATA_TYPE=mri
TRAIN_SIZE=0.8
RANDOM_STATE=2021

INTENSITY=normalize
BLUR_STD=0.1

BATCH_SIZE=16
EPOCHS=100

ARCH=18

for RANDOM_STATE in 2021 2022 2023
do
	for LEARNING_RATE in 0.0001
	do
		python ./run_mri_resnet.py \
		--gpus $GPUS \
		--server $SERVER \
		--root /raidWorkspace/mingu/Data/ADNI \
		--data_info labels/data_info.csv \
		--train_size $TRAIN_SIZE \
		--random_state $RANDOM_STATE \
		--intensity $INTENSITY \
		--rotate \
		--flip \
		--blur \
		--blur_std $BLUR_STD \
		--epochs $EPOCHS \
		--batch_size $BATCH_SIZE \
		--optimizer adamw \
		--learning_rate $LEARNING_RATE \
		--weight_decay 0.0001 \
		--cosine_warmup 0 \
		--cosine_cycles 1 \
		--cosine_min_lr 0.0 \
		--save_every 20 \
		--enable_wandb \
		--arch $ARCH
	done
done
echo "Finished."
