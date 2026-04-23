#!/usr/bin/env bash

source .env && python train_batch.py \
  --env_name=square \
  --pretrain_r=False \
  --pretrain_q=False \
  --pretrain_steps=200000 \
  --start_training=0 \
  --trajs_per_update=200 \
  --max_iter=15 \
  --eval_episodes=50 \
  --grad_updates_per_iter=6000 \
  --config.T=100 \
  --config.r_action_scale=0.1 \
  --config.filter_temperature_train=0.0 \
  --config.filter_temperature_eval=0.0 $@
