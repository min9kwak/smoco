echo "Experiments Started"
SERVER=dgx
GPUS=0

RANDOM_STATE=2021
BATCH_SIZE=16
EPOCHS=100

for ARCH in 18 50
do
	for LEARNING_RATE in 0.0001 0.0002
	do
		python ./run_mri_resnet.py \
		--gpus $GPUS \
		--server $SERVER \
		--root /raidWorkspace/mingu/Data/ADNI \
		--data_info labels/data_info.csv \
		--random_state $RANDOM_STATE \
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
