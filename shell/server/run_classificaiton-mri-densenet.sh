echo "Experiments Started"
SERVER=workstation
GPUS=0

DATA_TYPE=mri
RANDOM_STATE=2021

INTENSITY=scale

BATCH_SIZE=16
EPOCHS=100

BACKBONE_TYPE=densenet
INIT_FEATUERS=32
GROWTH_RATE=32
BLOCK_CONFIG="6,12,24,16"


for RANDOM_STATE in 1 2 3 4 5
do
	for LEARNING_RATE in 0.0001
	do
		python ./run_classification.py \
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
		--init_features $INIT_FEATUERS \
		--growth_rate $GROWTH_RATE \
		--block_config=$BLOCK_CONFIG \
		--bn_size 4 \
		--dropout_rate 0.0 \
		--epochs $EPOCHS \
		--batch_size $BATCH_SIZE \
		--optimizer adamw \
		--learning_rate $LEARNING_RATE \
		--weight_decay 0.0001 \
		--cosine_warmup 0 \
		--cosine_cycles 1 \
		--cosine_min_lr 0.0 \
		--save_every 20 \
		--enable_wandb
	done
done
echo "Finished."
