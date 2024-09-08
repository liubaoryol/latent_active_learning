#!/bin/sh
python latent_active_learning/scripts/train_hbc.py with movers movers_optimal=False student_type=iterative_random
python latent_active_learning/scripts/train_hbc.py with movers movers_optimal=False student_type=action_entropy
python latent_active_learning/scripts/train_hbc.py with movers movers_optimal=False student_type=action_intent_entropy 
