from stable_baselines3 import DQN, PPO

import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(base_path)

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure

from kube_mm_scheduler.model.promes import Net5_
from kube_mm_scheduler.model.net3 import Model as Net3

from typing import Dict, List, Tuple, Type, Union

device = th.device("cuda" if th.cuda.is_available() else "cpu")

import numpy as np


class SimRlScheduler:
    def __init__(self, env, model_fname='DQN_mm_pr_dynamic.zip'):
        self.env = env
        self.model_name = model_fname.split('.')[0]
        # print(f"Model file name: {self.model_name}")
        if self.model_name.startswith('_'):
            self.model_fpath = os.path.join(base_path, 'notebook', 'training', 'model', self.model_name[1:])
        else:
            self.model_fpath = os.path.join(base_path, 'kube_rl_scheduler', 'strategies', 'model', self.model_name)
        self.model_type = model_fname.split('_')[0]
        if self.model_type == '':
            self.model_type = model_fname.split('_')[1]
        self.model = None

        print(f"Model type: {self.model_type}")

        # Capitalize model type
        self.model_type = self.model_type.upper()
        if self.model_type.startswith('DQN'):
            self.model = DQN.load(self.model_fpath)
        elif self.model_type.startswith('PPO'):
            self.model = PPO.load(self.model_fpath)

    def decision(self, env):

        state = env.get_state()

        # Temporary add function : If there's no available nodes or no pending pod, return action 0
        if not env.cluster.pending_pods:
            return 0
        if self.get_available_nodes(env) == [0]:
            return 0
        
        action, _ = self.model.predict(state)

        action = action.item()

        return action
    
    def get_available_nodes(self, env):
        if not env.cluster.pending_pods:
            return [0]

        pod = env.cluster.pending_pods[0]
        ret = []
        
        for idx, node in enumerate(env.cluster.nodes):
            cpu_avail = node.spec["cpu_pool"] - node.status["cpu_util"]
            mem_avail = node.spec["mem_pool"] - node.status["mem_util"]
            if pod.spec["cpu_req"] <= cpu_avail and pod.spec["mem_req"] <= mem_avail:
                ret.append(idx+1)
        if not ret:
            ret.append(0)
        return ret