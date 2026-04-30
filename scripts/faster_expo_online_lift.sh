#!/usr/bin/env bash

source .env && python train_robo.py \
  --dataset_dir=ph \
  --config.model_cls=FasterEXPOLearner \
  --env_name=lift \
  --num_data=10 $@
