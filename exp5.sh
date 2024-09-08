#!/bin/sh
python latent_active_learning/scripts/train_hbc.py with discrete_env boxworld_8targets simple_repr query_percent=1 num_demos=50
python latent_active_learning/scripts/train_hbc.py with discrete_env boxworld_8targets simple_repr query_percent=1 num_demos=100
python latent_active_learning/scripts/train_hbc.py with discrete_env boxworld_8targets simple_repr query_percent=1 num_demos=150
