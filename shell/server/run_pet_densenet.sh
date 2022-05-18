echo "Experiments Started"
SERVER=dgx
GPUS=0

DATA_TYPE=pet

RANDOM_STATE=2021
BATCH_SIZE=16
EPOCHS=100

INIT_FEATUERS=32
GROWTH_RATE=16
BLOCK_CONFIG="6,12,24,16"

for RANDOM_STATE in 2021 2022 2023
do
	for LEARNING_RATE in 0.0001
	do
		python ./run_uni_densenet.py \
		--gpus $GPUS \
		--server $SERVER \
		--data_type $DATA_TYPE \
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
		--init_features $INIT_FEATUERS \
		--growth_rate $GROWTH_RATE \
		--block_config=$BLOCK_CONFIG \
		--bn_size 4 \
		--dropout_rate 0.0
	done
done
echo "Finished."
