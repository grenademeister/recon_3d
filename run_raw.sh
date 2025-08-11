#!/bin/bash

LOG_DATE=/home/juhyung/data/longitudinal/log/log_recon_raw_2025_07_13

# echo "[INFO] Removing previous log directory: $LOG_DATE"
# rm -rf $LOG_DATE

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

PYTHON_PATH=~/.conda/envs/python310/bin/python

export DATA_ROOT="/home/juhyung/data/longitudinal/log/log_prediction_raw_2025_07_13/00001_test/results"
export RUN_DIR=$LOG_DATE
GPU=1
PRIOR_KEY="out"
TARGET_KEY="flair"
TARGET_MASK_KEY="mask_fl"
LONGITUDINAL_CHECKPOINTS="/home/juhyung/data/longitudinal/log/log_recon_2025_07_13/00000_train/checkpoints/checkpoint_best.ckpt"
nohup $PYTHON_PATH test_raw.py \
  --gpu $GPU \
  --longitudinal_checkpoints $LONGITUDINAL_CHECKPOINTS \
  --prior_key $PRIOR_KEY \
  --target_key $TARGET_KEY \
  --target_mask_key $TARGET_MASK_KEY \
  --use_meta True \
  > /dev/null 2>&1 &
sleep 5

PYTHON_PATH=~/.conda/envs/python310/bin/python


find $LOG_DATE -type f -name "*.log" -exec tail -n 100 -f {} +


