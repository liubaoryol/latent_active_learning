from typing import List
import numpy as np
import torch

from stable_baselines3.common.evaluation import evaluate_policy
from latent_active_learning.data.types import TrajectoryWithLatent


class Evaluator():
    def __init__(self, logger, env):
        self._logger = logger
        self._env = env
        
    def evaluate_and_log(
        self,
        model, 
        student,
        oracle,
        epoch
        ):

        hamming_train = self.hamming_distance(student.demos, oracle.true_options)
        hamming_test = self.hamming_distance(student.demos_test, oracle.true_options_test)
        mean_return, std_return = self.env_interaction(model)

        self._logger.log_batch(
            epoch_num=epoch,
            hamming_loss=hamming_train,
            hamming_loss_test=hamming_test,
            rollout_mean=mean_return,
            rollout_std=std_return
        )

        self._logger.last_mean = mean_return
        self._logger.last_std = std_return
        self._logger.last_01_distance = hamming_train
    
    def hamming_distance(self, demos, true_options):
        with torch.no_grad():
            options = [demo.latent for demo in demos]
            f = lambda x: np.linalg.norm(
                (options[x].squeeze()[1:] - true_options[x]), 0)/len(options[x])
            distances = list(map(f, range(len(options))))
        return np.mean(distances)
    
    def env_interaction(self, model):
        # mean_return, std_return = evaluate_policy(
        #     model,
        #     TransformBoxWorldReward(self._env),
        #     10
        #     )
        mean_return, std_return = evaluate_policy(model, self._env, 10)
        return mean_return, std_return