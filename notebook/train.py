import os, sys

base_path = os.path.join(os.getcwd(), "..")
print(f"Base Path: {base_path}")
sys.path.append(base_path)

# Stable baselines3
import stable_baselines3 as sb3

# env
import gym
from kube_sim_gym.envs.sim_kube_env import SimKubeEnv

import torch as th
import torch.nn as nn
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure

from kube_mm_scheduler.model.promes import Net5_

device = th.device("cuda" if th.cuda.is_available() else "cpu")

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

def test_model(scenario_file, model_fname, log_name, scenario_idx, policy_kwargs):
    # ============================== Performance Test ===============================

    # Previous model performance test (vs. defautl scheduler)
    # Test scenario : scenario-5l-5m-1000p-10m.csv
    test_env1 = gym.make('SimKubeEnv-v0', reward_file='train_dynamic.py', scenario_file=scenario_file)
    test_env2 = gym.make('SimKubeEnv-v0', reward_file='train_dynamic.py', scenario_file=scenario_file)

    # Default Scheduler
    from kube_hr_scheduler.scheduler.sim_hr_scheduler import SimHrScheduler
    default_scheduler = SimHrScheduler(test_env2, 'default.py')

    # RL Scheduler
    from kube_rl_scheduler.scheduler.sim_rl_scheduler import SimRlScheduler
    rl_scheduler = SimRlScheduler(test_env1, f'_{model_fname}.zip', policy_kwargs=policy_kwargs)


    # Test the model
    obs1 = test_env1.reset()
    obs2 = test_env2.reset()
    done1 = False
    done2 = False
    step1 = 0
    step2 = 0
    acc_rew1 = 0
    acc_rew2 = 0

    print(f"Testing with {scenario_file} (my model vs. default)")
    while not done1 or not done2:
        if not done1:
            # action1, _ = model.predict(obs1)
            action1 = rl_scheduler.decision(test_env1)
            obs1, reward1, done1, _ = test_env1.step(action1)
            step1 += 1
            acc_rew1 += reward1
        if not done2:
            action2 = default_scheduler.decision(test_env2)
            obs2, reward2, done2, _ = test_env2.step(action2)
            step2 += 1
            acc_rew2 += reward2

    acc_rew1 = round(acc_rew1, 2)
    acc_rew2 = round(acc_rew2, 2)

    print(f"Test result(reward): {acc_rew1} vs. {acc_rew2}")
    print(f"Test result(step): {step1} vs. {step2}")

    return acc_rew1, acc_rew2, step1, step2

    # ============================== ==================== ===============================
    
from IPython.display import clear_output

def training(json_tracker_fname):

    log_name = json_tracker_fname.split('.')[0]
    log_path = 'training/log/' + log_name
    if not os.path.exists(log_path):
        os.makedirs(f'training/log/{log_name}')

    new_logger = configure(log_path, ['stdout', 'csv', 'tensorboard'])

    # Load the json tracker
    import json
    with open(f'training/{json_tracker_fname}', 'r') as f:
        json_tracker = json.load(f)

    reward_file = json_tracker['reward_file']
    reward_key = os.path.splitext(reward_file)[0].split('_')[1]

    # Environment
    envs = []
    for i in range(1, 50):
        env = gym.make('SimKubeEnv-v0', reward_file=reward_file, scenario_file=f'trace2017_100_{i}.csv')
        envs.append(env)

    # Check if the last scenario is None
    if json_tracker['last_scenario'] == 0:
        # If it is None, then start from the first scenario
        scenario_idx = 1
    else:
        # If it is not None, then continue from the last scenario
        scenario_idx = int(json_tracker['last_scenario']) + 1

    n_scenario = len(os.listdir(os.path.join(base_path, 'scenarios', 'trace2017')))

    model_name = json_tracker['model_name']

    feature_net = json_tracker['feature_net']
    if feature_net == 'Promes':
        policy_kwargs = policy_kwargs_promes
    elif feature_net == 'Naive':
        policy_kwargs = policy_kwargs_naive
    else:
        print("Wrong feature net name")
        return

    if model_name == 'DQN':
        model = sb3.DQN
    elif model_name == 'PPO':
        model = sb3.PPO

    env = envs[scenario_idx-1]
    timesteps = json_tracker['total_steps']

    model_fname = f'{model_name}_{feature_net}_{reward_key}.zip'
    model_fpath = f'training/model/{model_fname}'

    # Check if the model is already trained
    if os.path.exists(model_fpath):
        print(f"Model {model_name} is already trained.")
        model = model.load(model_fpath)
    else:
        print(f"Training with scenario {env.scenario_file}")
        model = model('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
        model.save(model_fpath)

    # Logger
    model.set_logger(new_logger)

    while scenario_idx < n_scenario:

        a1, a2, a3, a4 = test_model('scenario-5l-5m-1000p-10m.csv', model_fname, log_name, scenario_idx, policy_kwargs=policy_kwargs)
        b1, b2, b3, b4 = test_model('scenario-3l-10m-1000p-10m.csv', model_fname, log_name, scenario_idx, policy_kwargs=policy_kwargs)
        c1, c2, c3, c4 = test_model('scenario-10l-3m-1000p-10m.csv', model_fname, log_name, scenario_idx, policy_kwargs=policy_kwargs)

        with open(f'training/log/{log_name}/test_result.txt', 'a') as f:
            f.write(f"{scenario_idx - 1}, {a1}, {a2}, {a3}, {a4}, {b1}, {b2}, {b3}, {b4}, {c1}, {c2}, {c3}, {c4}\n")

        env = envs[scenario_idx-1]
        model.set_env(env)
        model.learn(total_timesteps=timesteps)

        # Save the model
        model.save(f'training/model/{model_name}.zip')

        # Update the json tracker
        json_tracker['last_scenario'] = scenario_idx

        # Save the json tracker
        with open(f'training/{json_tracker_fname}', 'w') as f:
            json.dump(json_tracker, f)

        scenario_idx += 1
        
if __name__ == "__main__":
    json_fname = sys.argv[1]
    
    training(json_fname)