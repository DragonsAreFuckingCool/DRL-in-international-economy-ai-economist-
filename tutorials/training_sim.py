import sys
import ray
import numpy as np
import tensorflow as tf
import gym


import os
import json
import pickle
import time
from copy import deepcopy

import numpy as np
import pandas as pd

import random

sys.path.append(r'C:\Users\adria\coding\katja\DRL-in-international-economy-ai-economist-')

from ai_economist import foundation

from utils import plotting

import ray
from ray.rllib.agents.ppo import PPOTrainer

from rllib.env_wrapper import RLlibEnvWrapper


import os
import warnings

# TensorFlow / C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Python warnings
warnings.filterwarnings("ignore")

# Ray logging
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"

import simulation as exp

import ray
ray.shutdown()
ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR")

settings = exp.ExperimentSettings(
    phase1_iters = 1,
    phase2_iters = 1,
    phase3a_iters = 1,
    phase3b_iters = 1,
    save_results=False,
    travel_enabled_phase3a=False,
    travel_enabled_phase3b=False,
    restrict_trade_to_region = True,
    experiment_extra_tag = "Original_with_travel",

    num_workers=0,
    num_envs_per_worker=1,
    num_gpus=1,
    rollout_fragment_length=100,
    train_batch_size=500,
    sgd_minibatch_size=100,
    num_sgd_iter=2,

    # train_batch_size=6000,      # 15 workers * 2 envs * 200 steps
    # sgd_minibatch_size=1500,
    # num_sgd_iter=4,    
    


)

#results = exp.run_experiment(settings)
def results(): 
    return exp.run_experiment(settings)