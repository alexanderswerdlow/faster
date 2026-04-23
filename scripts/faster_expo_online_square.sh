#!/usr/bin/env bash

source .env && python train_robo.py \
  --dataset_dir=ph \
  --config.model_cls=BetterDiffusionSACLearner \
  --env_name=square $@
