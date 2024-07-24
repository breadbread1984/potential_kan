# Introduction

this project is to predict potential from electron density

# Usage

## Install prerequisite packages

```shell
python3 -m pip install -r requirements.txt
```

## Generate dataset

generate trainset with the following command

```shell
python3 create_dataset.py --pkls /root/data/data/pkl_folder_CH_4/c2h6_0.0000.pkl /root/data/data/pkl_folder_CH_4/c2h6_-0.1000.pkl /root/data/data/pkl_folder_CH_4/c2h6_0.1000.pkl
```

generate valset with the following command

```shell
python3 create_dataset.py --pkls /root/data/data/pkl_folder_CH_4/c2h6_0.0500.pkl /root/data/data/pkl_folder_CH_4/c2h6_-0.0500.pkl --val
```

## Train model

```shell
torchrun --nproc_per_node <data/parallelism/number> --nnodes 1 --node_rank 0 --master_addr localhost --master_port <port num> train.py --trainset <path/to/train/npz> --valset <path/to/eval/npz> [--ckpt <path/to/checkpoint>] [--batch_size <batch size>] [--lr <learning rate>] [--workers <number of workers>] [--device (cpu|cuda)]
```

