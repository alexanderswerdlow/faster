#!/usr/bin/env bash

source .env && python train_robo.py \
  --env_name=can \
  --config=faster/agents/faster_idql_learner.py \
  --config.model_cls=FasterIDQLLearner \
  --config.T=100 \
  --start_training=10000 \
  --config.filter_temperature_eval=10.0 \
  --config.expectile=0.8 $@
