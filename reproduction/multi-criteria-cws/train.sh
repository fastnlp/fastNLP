export EXP_NAME=release04
export NGPU=2
export PORT=9988
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$1

if [ -z "$DATA_DIR" ]
then
    DATA_DIR="./data"
fi

echo $CUDA_VISIBLE_DEVICES
cmd="
python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port $PORT\
    main.py \
    --word-embeddings cn-char-fastnlp-100d \
    --bigram-embeddings cn-bi-fastnlp-100d \
    --num-epochs 100 \
    --batch-size 256 \
    --seed 1234 \
    --task-name $EXP_NAME \
    --dataset $DATA_DIR \
    --freeze \
"
echo $cmd
eval $cmd
