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


class Net3_(Net3):
    def __init__(self):
        super(Net3_, self).__init__()
        self.fc4 = None

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1_1(x1))  
        x2 = F.relu(self.fc1_2(x2))
        x = torch.cat((x1, x2), dim=1) 
        x = F.relu(self.fc2(x))  
        x = F.relu(self.fc3(x))
        return x


class PromesPPO(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 80):
        super(PromesPPO, self).__init__(observation_space, features_dim)
        self.net = Net5_().to(device)
        self.net.load_state_dict(th.load(os.path.join(base_path,'kube_mm_scheduler/weight/net5.pt')))
        self.net.eval()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        input1 = observations[:, :10].to(device)
        input2 = observations[:, 10:].to(device)

        return self.net(input1, input2)

class Naive(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 80):
        super(Naive, self).__init__(observation_space, features_dim)
        self.net = nn.Linear(observation_space.shape[0], features_dim).to(device)
        self.net.eval()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

policy_kwargs_promes = dict(
    features_extractor_class=PromesPPO,
    features_extractor_kwargs=dict(features_dim=80),
)

policy_kwargs_naive = dict(
    features_extractor_class=Naive,
    features_extractor_kwargs=dict(features_dim=80),
)

class MlpExtractor(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
    
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ActorCriticPolicy(nn.Module):

    def __init__(self, env):
        super().__init__()

        # Features extractor
        self.features_extractor = PromesPPO(observation_space=env.observation_space ,features_dim=80)

        # MLP extractor
        self.mlp_extractor = MlpExtractor(feature_dim=80, net_arch=[64, 64], activation_fn=nn.Tanh, device=device)

        # Action net
        self.action_net = nn.Linear(in_features=64, out_features=6, bias=True)

        # Value net
        self.value_net = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, state):
        # Extract features
        features = self.features_extractor(state)

        # Extract policy and value
        policy = self.mlp_extractor.policy_net(features)
        value = self.mlp_extractor.value_net(features)

        # Get action
        action = self.action_net(policy)

        # Get value
        value = self.value_net(value)

        return action, value

# class ActorCriticPolicy(nn.Module):

#     def __init__(self, env, feature_dim):
#         super().__init__()

#         # Features extractor
#         self.features_extactor = PromesPPO(observation_space=env.observation_space ,features_dim=80)
#         # self.features_extractor = Naive(observation_space=env.observation_space ,features_dim=feature_dim)

#         # MLP extractor
#         self.mlp_extractor = MlpExtractor(feature_dim=80, net_arch=[64, 64], activation_fn=nn.Tanh, device=device)

#         # Action net
#         self.action_net = nn.Linear(in_features=64, out_features=6, bias=True)

#         # Value net
#         self.value_net = nn.Linear(in_features=64, out_features=1, bias=True)

#     def forward(self, state):
#         # Extract features
#         features = self.features_extractor(state)

#         # Extract policy and value
#         policy = self.mlp_extractor.policy_net(features)
#         value = self.mlp_extractor.value_net(features)

#         # Get action
#         action = self.action_net(policy)

#         # Get value
#         value = self.value_net(value)

#         return action, value
    


class SimRlScheduler:
    def __init__(self, env, model_fname='ppo_1st.zip'):
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

        if self.model_type == 'DQN':
            self.model = DQN.load(self.model_fpath, env=self.env)
        elif self.model_type == 'PPO':
            self.model = PPO.load(self.model_fpath, env=self.env)

        model_policy = self.model.policy

        # print(f"model_policy: {model_policy}")

        # Get the feature_dim from the model
        feature_dim = model_policy.mlp_extractor.policy_net[0].in_features
        # print("Feature dim: ", feature_dim)

        self.model = ActorCriticPolicy(env)

        # print(f"model: {self.model}")

        self.model.load_state_dict(model_policy.state_dict())
        self.model.eval()

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
            state = th.tensor(state, dtype=th.float32).unsqueeze(0)
            scores = self.model(state)[0]
            scores = scores.tolist()[0]
            # print(f"Scores: {scores}")

            # Exclude unavailable nodes
            for idx in range(len(scores)):
                if idx not in available_nodes:
                    # print(f"Excluding node {idx}")
                    scores[idx] = -1e9

            # print(f"Scores: {scores}")

            action = np.argmax(scores)

            return action