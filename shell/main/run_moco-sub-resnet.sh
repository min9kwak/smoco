echo "Experiments Started"
SERVER=main
GPUS=0

DATA_TYPE=pet
IMAGE_SIZE=36
RANDOM_STATE=2021

INTENSITY=scale

BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=0.01

BACKBONE_TYPE=resnet
ARCH=50

PROJECTOR_DIM=128
NUM_NEGATIVES=1024
KEY_MOMENTUM=0.995

for RANDOM_STATE in 2021 2022 2023
do
	for BATCH_SIZE in 16 32
	do
		python ../../run_moco_sub.py \
		--gpus $GPUS \
		--server $SERVER \
		--data_type $DATA_TYPE \
		--root D:/data/ADNI \
		--data_info labels/data_info.csv \
		--mci_only \
		--train_size 0.9 \
		--image_size $IMAGE_SIZE \
		--segment left_hippocampus \
		--random_state $RANDOM_STATE \
		--intensity $INTENSITY \
		--rotate \
		--flip \
		--blur \
		--blur_std 0.1 \
		--backbone_type $BACKBONE_TYPE \
		--arch $ARCH \
		--epochs $EPOCHS \
		--batch_size $BATCH_SIZE \
		--optimizer sgd \
		--learning_rate $LEARNING_RATE \
		--weight_decay 0.0001 \
		--cosine_warmup 0 \
		--cosine_cycles 1 \
		--cosine_min_lr 0.0 \
		--save_every 100 \
		--enable_wandb \
		--projector_dim $PROJECTOR_DIM \
		--num_negatives $NUM_NEGATIVES \
		--key_momentum $KEY_MOMENTUM \
		--split_bn
	done
done
echo "Finished."
