import os, sys

base_path = os.path.join(os.getcwd(), "..")
print(f"Base Path: {base_path}")
sys.path.append(base_path)

import glob

import stable_baselines3 as sb3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure

from datetime import datetime

import gym
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

from kube_sim_gym.envs import *

def test_rl_model(scenario_file, rl_model, reward_file):

    test_env1 = gym.make('SimKubeEnv-v0', reward_file=reward_file, scenario_file=scenario_file)
    test_env2 = gym.make('SimKubeEnv-v0', reward_file=reward_file, scenario_file=scenario_file)

    # RL Scheduler
    rl_model.set_env(test_env1)

    # Default Scheduler
    from kube_hr_scheduler.scheduler.sim_hr_scheduler import SimHrScheduler
    default_scheduler = SimHrScheduler(test_env2, 'default.py')


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
            action1, _ = rl_model.predict(obs1)
            # action1 = rl_scheduler.decision(test_env1)
            obs1, reward1, done1, _ = test_env1.step(action1)
            step1 += 1
            acc_rew1 += reward1
        if not done2:
            action2 = default_scheduler.decision(test_env2)
            obs2, reward2, done2, _ = test_env2.step(action2)
            step2 += 1
            acc_rew2 += reward2

        # If any of the env is done and 2000 steps have passed, stop the other
        if step1 - step2 >= 2000 and not done1 and done2:
            done1 = True
        if step2 - step1 >= 2000 and not done2 and done1:
            done2 = True

    acc_rew1 = round(acc_rew1, 2)
    acc_rew2 = round(acc_rew2, 2)

    print(f"Test result(reward): {acc_rew1} vs. {acc_rew2}")
    print(f"Test result(step): {step1} vs. {step2}")

    return acc_rew1, acc_rew2, step1, step2


from IPython.display import clear_output
from notebook.net_arch import *

def train_rl_model(json_tracker_fname):

    # Current date time
    date = datetime.now()
    date = date.strftime("%m%d%Y%H%M")

    log_name = json_tracker_fname.split('.')[0]
    log_path = f'training/log/{log_name}'

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = configure(log_path, ['stdout', 'csv', 'tensorboard'])
    
    # Load the json tracker
    import json
    with open(f'training/{json_tracker_fname}', 'r') as f:
        json_tracker = json.load(f)

    last_idx = json_tracker['last_idx']
    learning_steps = json_tracker['learning_steps']
    model_type = json_tracker['model_type']
    reward_file = json_tracker['reward_file']
    model_fname = json_tracker['model_fname']

    # Environment
    envs = []
    for i in range(1, 50):
        env = gym.make('SimKubeEnv-v0', reward_file=reward_file, scenario_file=f'trace2017_100_{i}.csv')
        envs.append(env)

    current_idx = last_idx + 1 # -1 as default

    # Model type : DQN or PPO
    if model_type == 'DQN':
        model = sb3.DQN
    elif model_type == 'PPO':
        model = sb3.PPO
    else:
        print(f"Unknown model type: {model_type}")
        return
    
    model_fpath = f'net_arch/{model_fname}.zip'

    # Check if the model exists
    # Load Model
    if os.path.exists(model_fpath):
        print(f"Loading the model from {model_fname}")
        model = model.load(model_fpath)
    else: # Error
        print(f"Model file does not exist: {model_fname}")
        return
    
    # If last_idx is not -1 and there's a model trained in training/model, then load the model
    if last_idx != -1 and glob.glob(f'training/model/{model_fname}_*'):
        # Load the model with the latest date
        model_fpaths = glob.glob(f'training/model/{model_fname}_*')
        model_fpaths.sort()
        model_fpath = model_fpaths[-1]
        print(f"Loading the model from {model_fpath}")

        model = model.load(model_fpath)        
    
    # Save the model, append _{date} to the model name
    trained_model_fname = f'{model_fname}_{date}'
    trained_model_fpath = f'training/model/{trained_model_fname}'

    # Set logger
    model.set_logger(logger)

    # Train the model
    while current_idx < 20: # Target training steps (Can be changed!)
        print(f"Training with {current_idx}th trace")

        # Test the model first
        a1, a2, a3, a4 = test_rl_model('scenario-5l-5m-1000p-10m_unbalanced.csv', model)
        b1, b2, b3, b4 = test_rl_model('scenario-10l-3m-1000p-10m_unbalanced.csv', model)
        c1, c2, c3, c4 = test_rl_model('scenario-3l-10m-1000p-10m_unbalanced.csv', model)

        with open(f'training/log/{log_name}/test_result.txt', 'a') as f:
            f.write(f"{current_idx},{a1},{a2},{a3},{a4},{b1},{b2},{b3},{b4},{c1},{c2},{c3},{c4}\n")

        env = envs[current_idx]
        model.set_env(env)
        model.learn(total_timesteps=learning_steps)

        # Save the model
        model.save(trained_model_fpath)

        # Update the json tracker
        json_tracker['last_idx'] = current_idx
        with open(f'training/{json_tracker_fname}', 'w') as f:
            json.dump(json_tracker, f)

        current_idx += 1

        clear_output()

        

if __name__ == '__main__':
    import sys
    json_tracker_fname = sys.argv[1]
    train_rl_model(json_tracker_fname)