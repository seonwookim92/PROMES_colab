import os, sys

base_path = os.path.join(os.getcwd(), "..")
print(f"Base Path: {base_path}")
sys.path.append(base_path)

# Load gym environment
import gym
from kube_sim_gym import *
from kube_sim_gym.envs.sim_kube_env import SimKubeEnv

from kube_hr_scheduler.scheduler.sim_hr_scheduler import SimHrScheduler
from kube_hr_scheduler.strategies.model.default import Model

import gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import Transitions

def init_env():
    env = gym.make('SimKubeEnv-v0', reward_file='train_step_3.py', scenario_file='random')
    return env

def init_model(env):
    # Create model and load policy
    model = PPO('MlpPolicy', env, verbose=1)

    return model

def init_eval_env(data_rate):
    # Prepare Eval ENV & Callback
    eval_env0 = gym.make("SimKubeEnv-v0", reward_file='train_step_3.py', scenario_file='scenario-5l-5m-1000p-10m_unbalanced.csv')
    eval_env1 = gym.make("SimKubeEnv-v0", reward_file='eval_rur.py', scenario_file='scenario-5l-5m-1000p-10m_unbalanced.csv')
    eval_env2 = gym.make("SimKubeEnv-v0", reward_file='eval_rbd1.py', scenario_file='scenario-5l-5m-10000p-10m_unbalanced.csv')
    eval_env3 = gym.make("SimKubeEnv-v0", reward_file='eval_rbd2.py', scenario_file='scenario-5l-5m-1000p-10m_unbalanced.csv')
    eval_env4 = gym.make("SimKubeEnv-v0", reward_file='eval_ct.py', scenario_file='scenario-5l-5m-1000p-10m_unbalanced.csv')

    return [eval_env0, eval_env1, eval_env2, eval_env3, eval_env4]

def eval_model(model, eval_envs):
    ret = []
    print('Evaluation : train_step_3')
    mean_reward, std_reward = evaluate_policy(model, eval_envs[0], n_eval_episodes=1, deterministic=True)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    ret += [mean_reward, std_reward]

    print('Evaluation : eval_rur')
    mean_reward, std_reward = evaluate_policy(model, eval_envs[1], n_eval_episodes=1, deterministic=True)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    ret += [mean_reward, std_reward]

    print('Evaluation : eval_rbd1')
    mean_reward, std_reward = evaluate_policy(model, eval_envs[2], n_eval_episodes=1, deterministic=True)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    ret += [mean_reward, std_reward]

    print('Evaluation : eval_rbd2')
    mean_reward, std_reward = evaluate_policy(model, eval_envs[3], n_eval_episodes=1, deterministic=True)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    ret += [mean_reward, std_reward]

    print('Episode length :')
    mean_reward, std_reward = evaluate_policy(model, eval_envs[4], n_eval_episodes=1, deterministic=True)
    print(f"Episode length:{mean_reward:.2f} +/- {std_reward:.2f}")
    ret += [mean_reward, std_reward]

    return ret

def pretrain(expert_demo, num_epochs=5):

    env = gym.make("SimKubeEnv-v0", reward_file='train_step_3.py')
    rng = np.random.default_rng(0)

    # Pretrain the model
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_demo,
        rng=rng,
    )

    bc_trainer.train(n_epochs=num_epochs, log_interval=100000)

    return bc_trainer

def load_expert_demo(path, rate):
    # Get the full length
    with open(path, 'r') as f:
        full_length = len(f.readlines())

    load_length = 10 ** rate
    print(f"Num of expert demo loading: {load_length}")

    # Load the expert demonstration
    data = np.genfromtxt(path, delimiter=',', max_rows=load_length)

    obs = data[:, :12]
    acts = data[:, 12]
    infos = np.empty(len(data), dtype=dict)
    next_obs = data[:, 13:25]
    dones = data[:, 25].astype(bool)

    transitions = Transitions(obs, acts, infos, next_obs, dones)

    return transitions

def log_eval(eval_log, log_path):
    # Create the log file if it doesn't exist
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('data_rate,train_step_3,eval_rur,eval_rbd1,eval_rbd2,episode_length\n')

    # Append the log
    with open(log_path, 'a') as f:
        f.write(','.join(map(str, eval_log)) + '\n')


# env = init_env()
# model = init_model(env)
# data_rate = int(input("Data Rate: "))
# eval_envs = init_eval_env(data_rate)
# eval_model(model, eval_envs)

# if data_rate != 0:
#     print("Pretraining...")
#     expert_demo = load_expert_demo(os.path.join(base_path, 'dataset', 'data_expert.csv'), data_rate)
#     policy = pretrain(expert_demo)
#     print("Pretraining Done")

# print("Now it's ready to proceed!")



if __name__ == "__main__":
    env = init_env()
    model = init_model(env)
    data_rate = int(input("Data Rate: "))
    log_path = f"results/{data_rate}.csv"
    eval_envs = init_eval_env(data_rate)
    eval_log = eval_model(model, eval_envs)

    # Delete the log file if it exists
    if os.path.exists(log_path):
        os.remove(log_path)

    # Write eval_log as csv
    with open(log_path, 'a') as f:
        f.write(f"{0}, {eval_log[0]}, {eval_log[2]}, {eval_log[4]}, {eval_log[6]}, {eval_log[8]}\n")

    if data_rate != 0:
        print("Pretraining...")
        expert_demo = load_expert_demo(os.path.join(base_path, 'dataset', 'data_expert.csv'), data_rate)
        bc_trainer = pretrain(expert_demo)

        model.policy = bc_trainer.policy
        model.policy.train(True)
        print("Pretraining Done")

    print("Now it's ready to proceed!")
    # input("Press Enter to continue...")

    # Train the model and evaluate it every 50000 steps
    for i in range(60):
        print(f"Training {i}th iteration...")

        model.learn(total_timesteps=50000, log_interval=10000)
        print(f"{i}th training done")
        eval_log = eval_model(model, eval_envs)

        # Write eval_log as csv
        with open(log_path, 'a') as f:
            f.write(f"{i+1}, {eval_log[0]}, {eval_log[2]}, {eval_log[4]}, {eval_log[6]}, {eval_log[8]}\n")

    # Save the model
    model.save(f"model/static_{data_rate}")

