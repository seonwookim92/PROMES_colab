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

class FE_PROMES(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 80):
        super(FE_PROMES, self).__init__(observation_space, features_dim)
        self.net = Net5_().to(device)
        self.net.load_state_dict(th.load(os.path.join(base_path,'kube_mm_scheduler/weight/net5.pt')))
        self.net.eval()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        input1 = observations[:, :10].to(device)
        input2 = observations[:, 10:].to(device)

        return self.net(input1, input2)

class FE_NAIVE(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 80):
        super(FE_NAIVE, self).__init__(observation_space, features_dim)
        self.net = nn.Linear(observation_space.shape[0], features_dim).to(device)
        self.net.eval()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

policy_kwargs_promes = dict(
    features_extractor_class=FE_PROMES,
    features_extractor_kwargs=dict(features_dim=80),
)

policy_kwargs_naive = dict(
    features_extractor_class=FE_NAIVE,
    features_extractor_kwargs=dict(features_dim=80),
)


class SimRlScheduler:
    def __init__(self, env, model_fname='ppo_1st.zip', policy_kwargs=policy_kwargs_promes):
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

        if self.model_type.startswith('DQN'):
            self.model = DQN.load(self.model_fpath)
        elif self.model_type.startswith('PPO'):
            self.model = PPO.load(self.model_fpath)

        self.model_policy = self.model.policy


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


    def decision(self, env):

        # Get available nodes
        available_nodes = self.get_available_nodes(env)
        # print(f"Available nodes: {available_nodes}")

        if available_nodes == [0]:
            return 0
        else:

            state = env.get_state()
            state = th.tensor(state, dtype=th.float32).unsqueeze(0).to(device)

            features = self.model_policy.extract_features(state)

            latent_pi, latent_vf = self.model_policy.mlp_extractor(features)

            scores = self.model_policy.action_net(latent_pi).tolist()[0]

            # Exclude unavailable nodes
            for idx in range(len(scores)):
                if idx not in available_nodes:
                    # print(f"Excluding node {idx}")
                    scores[idx] = -1e9

            # print(f"Scores: {scores}")

            action = np.argmax(scores)

            return action