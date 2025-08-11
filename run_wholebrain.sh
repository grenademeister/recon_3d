#!/bin/bash

LOG_DATE=/home/juhyung/data/longitudinal/log/log_recon_wholebrain_2025_07_13

# echo "[INFO] Removing previous log directory: $LOG_DATE"
# rm -rf $LOG_DATE

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

PYTHON_PATH=~/.conda/envs/python310/bin/python

# export DATA_ROOT="/home/juhyung/data/longitudinal/data/data_flairt1t2_test_fastmri/test"
# export RUN_DIR=$LOG_DATE
# GPU=3
# PRIOR_KEY="none"
# TARGET_KEY="fl"
# TARGET_MASK_KEY="fl_mask"
# LONGITUDINAL_CHECKPOINTS="/home/juhyung/data/longitudinal/log/log_recon_2025_07_13/00001_train/checkpoints/checkpoint_best.ckpt"
# nohup $PYTHON_PATH test_wholebrain.py \
#   --gpu $GPU \
#   --longitudinal_checkpoints $LONGITUDINAL_CHECKPOINTS \
#   --prior_key $PRIOR_KEY \
#   --target_key $TARGET_KEY \
#   --target_mask_key $TARGET_MASK_KEY \
#   > /dev/null 2>&1 &
# sleep 5

# export DATA_ROOT="/home/juhyung/data/longitudinal/data/data_longitudinal_test_snuhosptial/test"
# export RUN_DIR=$LOG_DATE
# GPU=1
# PRIOR_KEY="img1_reg"
# TARGET_KEY="img2"
# TARGET_MASK_KEY="mask2"
# LONGITUDINAL_CHECKPOINTS="/home/juhyung/data/longitudinal/log/log_recon_2025_07_13/00002_train/checkpoints/checkpoint_best.ckpt"
# nohup $PYTHON_PATH test_wholebrain.py \
#   --gpu $GPU \
#   --longitudinal_checkpoints $LONGITUDINAL_CHECKPOINTS \
#   --prior_key $PRIOR_KEY \
#   --target_key $TARGET_KEY \
#   --target_mask_key $TARGET_MASK_KEY \
#   > /dev/null 2>&1 &

# unset DATA_ROOT
# unset RUN_DIR
# unset LONGITUDINAL_CHECKPOINTS
# sleep 5

# export DATA_ROOT="/home/juhyung/data/longitudinal/data/data_flairt1t2_test_fastmri/test"
# export RUN_DIR=$LOG_DATE
# GPU=4
# PRIOR_KEY="t2"
# TARGET_KEY="fl"
# TARGET_MASK_KEY="fl_mask"
# LONGITUDINAL_CHECKPOINTS="/home/juhyung/data/longitudinal/log/log_recon_2025_07_13/00003_train/checkpoints/checkpoint_best.ckpt"
# nohup $PYTHON_PATH test_wholebrain.py \
#   --gpu $GPU \
#   --longitudinal_checkpoints $LONGITUDINAL_CHECKPOINTS \
#   --prior_key $PRIOR_KEY \
#   --target_key $TARGET_KEY \
#   --target_mask_key $TARGET_MASK_KEY \
#   > /dev/null 2>&1 &
# sleep 5


export DATA_ROOT="/home/juhyung/data/longitudinal/log/log_prediction_wholebrain_2025_07_13/00002_test/raw"
export RUN_DIR=$LOG_DATE
GPU=0
PRIOR_KEY="out"
TARGET_KEY="fl"
TARGET_MASK_KEY="mask_fl"
LONGITUDINAL_CHECKPOINTS="/home/juhyung/data/longitudinal/log/log_recon_2025_07_13/00004_train/checkpoints/checkpoint_best.ckpt"
nohup $PYTHON_PATH test_wholebrain.py \
  --gpu $GPU \
  --longitudinal_checkpoints $LONGITUDINAL_CHECKPOINTS \
  --prior_key $PRIOR_KEY \
  --target_key $TARGET_KEY \
  --target_mask_key $TARGET_MASK_KEY \
  > /dev/null 2>&1 &
sleep 5

export DATA_ROOT="/home/juhyung/data/longitudinal/log/log_prediction_wholebrain_2025_07_13/00005_test/raw"
export RUN_DIR=$LOG_DATE
GPU=1
PRIOR_KEY="out"
TARGET_KEY="fl"
TARGET_MASK_KEY="mask_fl"
LONGITUDINAL_CHECKPOINTS="/home/juhyung/data/longitudinal/log/log_recon_2025_07_13/00004_train/checkpoints/checkpoint_best.ckpt"
nohup $PYTHON_PATH test_wholebrain.py \
  --gpu $GPU \
  --longitudinal_checkpoints $LONGITUDINAL_CHECKPOINTS \
  --prior_key $PRIOR_KEY \
  --target_key $TARGET_KEY \
  --target_mask_key $TARGET_MASK_KEY \
  > /dev/null 2>&1 &
sleep 5

export DATA_ROOT="/home/juhyung/data/longitudinal/log/log_prediction_wholebrain_2025_07_13/00008_test/raw"
export RUN_DIR=$LOG_DATE
GPU=2
PRIOR_KEY="out"
TARGET_KEY="fl"
TARGET_MASK_KEY="mask_fl"
LONGITUDINAL_CHECKPOINTS="/home/juhyung/data/longitudinal/log/log_recon_2025_07_13/00004_train/checkpoints/checkpoint_best.ckpt"
nohup $PYTHON_PATH test_wholebrain.py \
  --gpu $GPU \
  --longitudinal_checkpoints $LONGITUDINAL_CHECKPOINTS \
  --prior_key $PRIOR_KEY \
  --target_key $TARGET_KEY \
  --target_mask_key $TARGET_MASK_KEY \
  > /dev/null 2>&1 &
sleep 5

find $LOG_DATE -type f -name "*.log" -exec tail -n 100 -f {} +


