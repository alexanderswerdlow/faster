#!/usr/bin/env bash

source .env && python train_robo.py \
  --dataset_dir=ph \
  --config.model_cls=FasterEXPOLearner \
  --env_name=tool_hang \
  --max_steps=1500000 \
  --config.filter_temperature_train=0.1 $@
