#!/usr/bin/env bash
  
source .env && python train_robo.py \
  --env_name=lift \
  --config=rlpd/agents/sac/idql_learner_fast.py \
  --config.model_cls=IDQLLearnerFast \
  --config.T=100 \
  --start_training=10000 \
  --config.expectile=0.8 \
  --config.filter_temperature_eval=10.0 \
  --num_data=10 $@
