#!/bin/bash

LOG_DATE=/home/juhyung/data/longitudinal/log/log_genmask_2025_07_13

# echo "[INFO] Removing previous log directory: $LOG_DATE"
# rm -rf $LOG_DATE

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

PYTHON_PATH=~/.conda/envs/python310/bin/python

export DATA_ROOT="/fast_storage/juhyung/data/longitudinal"
export RUN_DIR=$LOG_DATE
export TRAIN_ITER=4
echo "[RUN] DATA_ROOT=$DATA_ROOT"
echo "[RUN] TRAIN_ITER=$TRAIN_ITER"
nohup $PYTHON_PATH train.py --gpu 0,1,2,3,4,5,6,7  --acs_num 0 --parallel_factor 4 --recon_net_chan 32 --train_batch 48 --using_consistency True --prior_key img1_reg --target_key img2 --target_mask_key mask2 --rotation_conf none_0 > /dev/null 2>&1  &


unset DATA_ROOT
unset TRAIN_ITER
unset RUN_DIR

sleep 20
echo "[INFO] Tail logs in: $LOG_DATE"

find $LOG_DATE -type f -name "*.log" -exec tail -n 100 -f {} +
