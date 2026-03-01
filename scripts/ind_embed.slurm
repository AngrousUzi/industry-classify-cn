#!/bin/bash

#SBATCH --job-name=ind_classify    # 作业名
#SBATCH --partition=small               # cpu 队列
#SBATCH --nodes=1                         # 使用1个节点
#SBATCH --ntasks=1                        # 总任务数 1（Python主进程）
#SBATCH --cpus-per-task=4                # 每个任务分配2个CPU核心
#SBATCH --output=/dssg/home/acct-nanli/xxxxxxyz/IPO/logs/%x/%x_%j.out
#SBATCH --error=/dssg/home/acct-nanli/xxxxxxyz/IPO/logs/%x/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxxxxxyz@sjtu.edu.cn

cd /dssg/home/acct-nanli/xxxxxxyz/IPO/

type=$1
range=$2
if [ "$type" == "embed" ]; then
    echo "Running embedding script..."
elif [ "$type" == "filter" ]; then
    echo "Running filtering script..."
else
    echo "Invalid argument: $type. Please use 'embed' or 'filter'."
    exit 1
fi
if [ "$range" == "all" ]; then
    echo "Processing all data..."
    python -u  "industry-classify-cn/ind_${range}_${type}.py"
    python -u "industry-classify-cn/ind_all_similarity_quantile.py"
elif [ "$range" == "ipo" ]; then
    echo "Processing ipo data..."
    python -u  "industry-classify-cn/ind_${range}_${type}.py"
else
    echo "Invalid argument: $range. Please use 'all' or 'ipo'."
    exit 1
fi
