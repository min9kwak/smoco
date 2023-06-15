echo "Experiments Started"
SERVER=main
GPUS=0

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

BACKBONE_TYPE=resnet
ARCH=50

PROJECTOR_DIM=128
NUM_NEGATIVES=1024
KEY_MOMENTUM=0.995

ALPHAS="1.0,1.0"
ALPHAS_MIN="1.0,0.0"
ALPHAS_DECAY_END="-1,-1"
TOPK=5

for ALPHAS in "1.0,1.0"
do
  for RANDOM_STATE in 2021 2023 2025 2027 2029
	do
	  python ./run_supmoco.py \
		--gpus $GPUS \
		--server $SERVER \
		--data_type $DATA_TYPE \
		--root D:/data/ADNI \
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
		--enable_wandb \
		--projector_dim $PROJECTOR_DIM \
		--num_negatives $NUM_NEGATIVES \
		--key_momentum $KEY_MOMENTUM \
		--split_bn \
		--alphas=$ALPHAS \
		--alphas_min=$ALPHAS_MIN \
		--alphas_decay_end=$ALPHAS_DECAY_END \
		--topk $TOPK
	done
done
echo "Finished."
