#!/bin/bash

MODE=longitudinal  # no_prior | longitudinal | t2prior | t2pred

if [ -z "$MODE" ]; then
  echo "[ERROR] Please provide a mode: no_prior | longitudinal | t2prior | t2pred"
  exit 1
fi

LOG_DATE="/home/intern2/recon_3d/log/log_recon_250812"
PYTHON_PATH=~/.conda/envs/torchenv/bin/python

# echo "[INFO] Removing previous log directory: $LOG_DATE"
# rm -rf $LOG_DATE

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

export DATA_ROOT="/fast_storage/intern2/slabs"
export RUN_DIR=$LOG_DATE

GPU="1"
ACS_NUM=24
PARALLEL_FACTOR=8
TRAIN_BATCH=2
USING_CONSISTENCY=True
PREV_CHECKPOINT="/home/intern2/code_recon/log_recon_250805/checkpoints/checkpoint_best.ckpt"

COMMON_ARGS="\
  --gpu $GPU \
  --acs_num $ACS_NUM \
  --train_batch $TRAIN_BATCH \
  --using_consistency $USING_CONSISTENCY"


case $MODE in
  no_prior)
    export DATASET_MODE="base"
    export TRAIN_ITER=1
    echo "[MODE] Running NO PRIOR"
    nohup $PYTHON_PATH train.py \
      $COMMON_ARGS \
      --prior_key none \
      --target_key fl \
      --target_mask_key fl_mask \
      --using_prior False \
      --using_registration False \
      > /dev/null 2>&1 &
    ;;

  longitudinal)
    export DATASET_MODE="longitudinal"
    export TRAIN_ITER=2
    echo "[MODE] Running LONGITUDINAL"
    nohup $PYTHON_PATH train.py \
      $COMMON_ARGS \
      --prior_key img1_reg \
      --target_key img2 \
      --target_mask_key mask2 \
      # > /dev/null 2>&1 &
    ;;

  t2prior)
    export DATASET_MODE="base"
    export TRAIN_ITER=1
    echo "[MODE] Running T2 PRIOR"
    nohup $PYTHON_PATH train.py \
      $COMMON_ARGS \
      --prior_key t2_reg \
      --target_key fl \
      --target_mask_key fl_mask \
      > /dev/null 2>&1 &
    ;;

  t2pred)
    export DATASET_MODE="t2pred"
    export TRAIN_ITER=1
    echo "[MODE] Running T2 PREDICTION"
    nohup $PYTHON_PATH train.py \
      $COMMON_ARGS \
      --prior_key out \
      --target_key fl \
      --target_mask_key fl_mask \
      --use_meta True \
      > /dev/null 2>&1 &
    ;;


  *)
    echo "[ERROR] Invalid mode: $MODE"
    exit 1
    ;;
esac

# Clean up
unset DATA_ROOT
unset TRAIN_ITER
unset RUN_DIR
unset DATASET_MODE

# Log follow-up
# sleep 5
# echo "[INFO] Tail logs in: $LOG_DATE"
# find "$LOG_DATE" -type f -name "*.log" -exec tail -n 100 -f {} +
