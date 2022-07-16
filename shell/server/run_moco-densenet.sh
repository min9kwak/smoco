echo "Experiments Started"
SERVER=workstation2
GPUS=00

DATA_TYPE=pet
N_SPLITS=10
N_CV=0
IMAGE_SIZE=72
RANDOM_STATE=2021

INTENSITY=scale

EPOCHS=100
BATCH_SIZE=16
OPTIMIZER=adamw
LEARNING_RATE=0.0001

BACKBONE_TYPE=densenet
INIT_FEATURES=64
GROWTH_RATE=32
BLOCK_CONFIG="6,12,24,16"
DROPOUT_RATE=0.00

PROJECTOR_DIM=128
NUM_NEGATIVES=512
KEY_MOMENTUM=0.995

for RANDOM_STATE in 2021 2023 2025
do
	for LEARNING_RATE in 0.0001
	do
		python ./run_moco.py \
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
		--crop \
		--crop_size 64 \
		--small_kernel \
		--flip \
		--affine \
		--prob 0.5 \
		--backbone_type $BACKBONE_TYPE \
		--init_features $INIT_FEATURES \
		--growth_rate $GROWTH_RATE \
		--block_config=$BLOCK_CONFIG \
		--bn_size 4 \
		--dropout_rate $DROPOUT_RATE \
		--epochs $EPOCHS \
		--batch_size $BATCH_SIZE \
		--optimizer $OPTIMIZER \
		--learning_rate $LEARNING_RATE \
		--weight_decay 0.0001 \
		--cosine_warmup 0 \
		--cosine_cycles 1 \
		--cosine_min_lr 0.0 \
		--save_every 2000 \
		--enable_wandb \
		--projector_dim $PROJECTOR_DIM \
		--num_negatives $NUM_NEGATIVES \
		--key_momentum $KEY_MOMENTUM \
		--split_bn
	done
done
echo "Finished."
