if [ -z "$DATA_DIR" ] 
then
    DATA_DIR="./data"
fi

mkdir -vp $DATA_DIR

cmd="python -u ./data-prepare.py --sighan05 $1 --sighan08 $2 --data_path $DATA_DIR"
echo $cmd
eval $cmd

cmd="python -u ./data-process.py --data_path $DATA_DIR"
echo $cmd
eval $cmd
