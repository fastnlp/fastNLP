

# Multi-Criteria-CWS

An implementation of [Multi-Criteria Chinese Word Segmentation with Transformer](http://arxiv.org/abs/1906.12035) with fastNLP.

## Dataset
### Overview
We use the same datasets listed in paper.
- sighan2005
  - pku
  - msr
  - as
  - cityu
- sighan2008
  - ctb
  - ckip
  - cityu (combined with data in sighan2005)
  - ncc
  - sxu

### Preprocess
First, download OpenCC to convert between Traditional Chinese and Simplified Chinese.
``` shell
pip install opencc-python-reimplemented
```
Then, set a path to save processed data, and run the shell script to process the data.
```shell
export DATA_DIR=path/to/processed-data
bash make_data.sh path/to/sighan2005 path/to/sighan2008
```
It would take a few minutes to finish the process.

## Model
We use transformer to build the model, as described in paper.

## Train
Finally, to train the model, run the shell script.
The `train.sh` takes one argument, the GPU-IDs to use, for example:
``` shell
bash train.sh 0,1
```
This command use GPUs with ID 0 and 1.

Note: Please refer to the paper for details of hyper-parameters. And modify the settings in `train.sh` to match your experiment environment.

Type 
``` shell
python main.py --help
```
to learn all arguments to be specified in training.

## Performance

Results on the test sets of eight CWS datasets with multi-criteria learning. 

| Dataset        | MSRA  | AS    | PKU   | CTB   | CKIP  | CITYU | NCC   | SXU   | Avg.  |
| -------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Original paper | 98.05 | 96.44 | 96.41 | 96.99 | 96.51 | 96.91 | 96.04 | 97.61 | 96.87 |
| Ours           | 96.92 | 95.71 | 95.65 | 95.96 | 96.00 | 96.09 | 94.61 | 96.64 | 95.95 |

