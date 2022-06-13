echo "Experiments Started"
SERVER=workstation
GPUS=0

DATA_TYPE=pet
SEGMENT=global
IMAGE_SIZE=98
RANDOM_STATE=2021

INTENSITY=scale

BATCH_SIZE=16
EPOCHS=200

BACKBONE_TYPE=resnet
ARCH=50

LEARNING_RATE=0.03

PROJECTOR_DIM=128
NUM_NEGATIVES=1024
KEY_MOMENTUM=0.995

ALPHAS="1.0,1.0"
ALPHAS_MIN="1.0,0.0"
ALPHAS_DECAY_END="-1,-1"

for ALPHAS in "1.0,1.0" "1.0,0.5", "1.0,2.0"
do
	do
		python ./run_supmoco.py \
		--gpus $GPUS \
		--server $SERVER \
		--data_type $DATA_TYPE \
		--root /raidWorkspace/mingu/Data/ADNI \
		--data_info labels/data_info.csv \
		--mci_only \
		--train_size 0.9 \
		--segment $SEGMENT \
		--image_size $IMAGE_SIZE \
		--random_state $RANDOM_STATE \
		--intensity $INTENSITY \
		--rotate \
		--flip \
		--blur \
		--blur_std 0.1 \
		--prob 0.5 \
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
		--save_every 1000 \
		--enable_wandb \
		--projector_dim $PROJECTOR_DIM \
		--num_negatives $NUM_NEGATIVES \
		--key_momentum $KEY_MOMENTUM \
		--split_bn \
		--alphas=$ALPHAS \
		--alphas_min=$ALPHAS_MIN \
		--alphas_decay_end=$ALPHAS_DECAY_END
	done
done
echo "Finished."
