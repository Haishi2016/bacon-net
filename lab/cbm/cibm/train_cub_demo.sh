#!/bin/bash
# Train a baseline CBM and IB-regularized variants on CUB-200
# Backbone training is disabled for a demo run; images are embedded with a frozen Inception V3
# Checkpoints are saved to outputs/<timestamp>/model.pth

set -e

EPOCHS=100
NUM_RUNS=1
LR=0.0003
WD=0.001
BATCH_SIZE=64
SAMPLES_MI=64
OPTIMIZER=adam
LOG_BASE=outputs

echo "=== [1/3] Training Basic CBM ==="
python ./src/train.py \
    --model_arch '[2048]' \
    --dataset_name=CUB \
    --is_blackbox=False \
    --is_stochastic=False \
    --verbose=2 \
    --num_runs=$NUM_RUNS \
    --epochs=$EPOCHS \
    --wd=$WD \
    --samples_mi=$SAMPLES_MI \
    --train_backbone=False \
    --lr=$LR \
    --collect_MIs=False \
    --merge_train_val=False \
    --beta=0.5 \
    --beta_lr=0.0 \
    --optimizer=$OPTIMIZER \
    --batch_size=$BATCH_SIZE \
    --use_HC=False \
    --log_base=$LOG_BASE

echo "=== [2/3] Training IB-CBM (variational MI objective) ==="
python ./src/train.py \
    --model_arch '[2048]' \
    --dataset_name=CUB \
    --is_blackbox=False \
    --is_stochastic=True \
    --verbose=2 \
    --num_runs=$NUM_RUNS \
    --epochs=$EPOCHS \
    --wd=$WD \
    --samples_mi=$SAMPLES_MI \
    --train_backbone=False \
    --lr=$LR \
    --collect_MIs=False \
    --merge_train_val=False \
    --beta=0.25 \
    --beta_lr=-0.01 \
    --optimizer=$OPTIMIZER \
    --batch_size=$BATCH_SIZE \
    --use_HC=False \
    --log_base=$LOG_BASE

echo "=== [3/3] Training IB-CBM (entropy surrogate H(C)) ==="
python ./src/train.py \
    --model_arch '[2048]' \
    --dataset_name=CUB \
    --is_blackbox=False \
    --is_stochastic=True \
    --verbose=2 \
    --num_runs=$NUM_RUNS \
    --epochs=$EPOCHS \
    --wd=$WD \
    --samples_mi=$SAMPLES_MI \
    --train_backbone=False \
    --lr=$LR \
    --collect_MIs=False \
    --merge_train_val=False \
    --beta=0.25 \
    --beta_lr=0.0 \
    --optimizer=$OPTIMIZER \
    --batch_size=$BATCH_SIZE \
    --use_HC=True \
    --log_base=$LOG_BASE