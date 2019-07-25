#python -u train.py --task pos --ds conll --mode train --gpu 1 --lr 3e-4 --w_decay 2e-5 --lr_decay .95 --drop 0.3 --ep 25 --bsz 64 > conll_pos102.log 2>&1 &
#python -u train.py --task pos --ds ctb --mode train --gpu 1 --lr 3e-4 --w_decay 2e-5 --lr_decay .95 --drop 0.3 --ep 25 --bsz 64 > ctb_pos101.log 2>&1 &
python -u train.py --task cls --ds sst --mode train --gpu 0 --lr 1e-4 --w_decay 5e-5 --lr_decay 1.0 --drop 0.4 --ep 20 --bsz 64 > sst_cls.log &
#python -u train.py --task nli --ds snli --mode train --gpu 1 --lr 1e-4 --w_decay 1e-5 --lr_decay 0.9 --drop 0.4 --ep 120 --bsz 128 > snli_nli201.log &
#python -u train.py --task ner --ds conll --mode train --gpu 0 --lr 1e-4 --w_decay 1e-5 --lr_decay 0.9 --drop 0.4 --ep 120 --bsz 64 > conll_ner201.log &
