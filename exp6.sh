#!/bin/sh
python latent_active_learning/scripts/train_hbc.py with discrete_env boxworld_8targets simple_repr query_percent=1 num_demos=200
python latent_active_learning/scripts/train_hbc.py with discrete_env boxworld_8targets simple_repr query_percent=1 num_demos=300
python latent_active_learning/scripts/train_hbc.py with discrete_env boxworld_8targets simple_repr query_percent=1 num_demos=400
python latent_active_learning/scripts/train_hbc.py with discrete_env boxworld_8targets simple_repr query_percent=1 num_demos=500
