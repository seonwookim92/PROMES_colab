import os, sys
base_path = os.path.join(os.path.dirname(__file__))
sys.path.append(base_path)

import gym
import glob
import importlib
from tabulate import tabulate

from kube_rl_scheduler.scheduler.sim_rl_scheduler import SimRlScheduler
from kube_sim_gym.envs.sim_kube_env import SimKubeEnv
from stable_baselines3.common.logger import configure

if __name__ == "__main__":

    while True:

        # Model selection
        print('=' * 30)
        print()
        print("Trainable models: ")

        models = {
            "DQN" : "Deep Q Network (DQN) ... uses a replay buffer, a target network and gradient clipping.",
            "PPO" : "Proximal Policy Optimization (PPO) ... policy gradient method ... few tricks to improve stability and sample efficiency",
            "A2C" : "Advantage Actor Critic (A2C) ... policy gradient method ... few tricks to improve stability and sample efficiency",
            "DDPG" : "Deep Deterministic Policy Gradient (DDPG) is an actor-critic method that uses off-policy ... Bellman equation ... DQN for continuous action spaces",
            "HER" : "Hindsight Experience Replay (HER) ... learn from sparse and indirect rewards ... replaying trajectories with substituted goals",
            "SAC" : "Soft Actor Critic (SAC) ... actor-critic method ... stochastic policy in an off-policy way ... extension of TD3 ... uses a differentiable soft actor-critic loss",
            "TD3" : "Twin Delayed DDPG (TD3) ... off-policy actor-critic method ... uses two critics to limit overestimation of the Q-values ... uses delayed policy updates",
        }
        models = [ [idx, model, description] for idx, (model, description) in enumerate(models.items()) ]
        
        print(tabulate(models, headers=["Index", "Model", "Description"]))
        model_idx = int(input("Select model index: "))

        # Get the name of the model
        model_name = models[model_idx][1]
        print('-'*30)
        print(f"Selected model: {model_name}")

        sb3 = importlib.import_module("stable_baselines3")

        if model_idx == 0: model = sb3.DQN
        elif model_idx == 1: model = sb3.PPO
        elif model_idx == 2: model = sb3.A2C
        elif model_idx == 3: model = sb3.DDPG
        elif model_idx == 4: model = sb3.HER
        elif model_idx == 5: model = sb3.SAC
        elif model_idx == 6: model = sb3.TD3
        else:
            print("Wrong idx")
            exit()

        # Reward function selection
        print('='*30)
        print()
        print("Available reward functions: ")
        rewards_path = glob.glob(os.path.join(base_path, 'kube_rl_scheduler', 'strategies', 'reward', '*.py'))
        rewards_fname = [os.path.basename(reward_path) for reward_path in rewards_path]
        rewards = [ [idx, reward_fname, ""] for idx, reward_fname in enumerate(rewards_fname) ]
        print(tabulate(rewards, headers=["Index", "Reward", "Description"]))
        print()
        reward_idx = int(input("Select reward index: "))
        reward_fname = rewards[reward_idx][1]
        print('-'*30)
        print(f"Selected reward: {reward_fname}")

        # Scenario selection
        print('='*30)
        print()
        print("Trainable scenarios: ")
        # Fetch Scenarios from {base_path}/scenarios/
        scenarios_path = glob.glob(os.path.join(base_path, 'scenarios', '*.csv'))
        scenarios_fname = [os.path.basename(scenario_path) for scenario_path in scenarios_path]

        def describe_scenario(scenario_fname):
            try:
                _, level, duration, n_pod, time = scenario_fname.split('-')
                level = int(level[:-1])
                duration = int(duration[:-1])
                n_pod = int(n_pod[:-1])
                time = int(time[:-5])
                return f"Level: {level}, Duration: {duration}, n_pod: {n_pod}, time: {time}"
            except:
                return ""
            
        scenarios = [[0, 'random', 'Random scenario (length : 1000)']]
        scenarios += [ [idx+1, scenario_fname, describe_scenario(scenario_fname)] for idx, scenario_fname in enumerate(scenarios_fname) ]

        print(tabulate(scenarios, headers=["Index", "Scenario", "Description"]))
        print()
        
        scenario_idx = int(input("Select scenario index: "))

        # Load the environment
        scenario_fname = scenarios[scenario_idx][1]
        print('-'*30)
        print(f"Selected scenario: {scenario_fname}")
        env = SimKubeEnv(reward_fname, scenario_fname)


        # Continue training?
        print('='*30)
        print()
        continue_training = input("Continue training? (y/n): ")
        if continue_training == "y":
            continue_training = True
        else:
            continue_training = False

        # Trained Model selection
        if continue_training:
            print(f"Trained models({model_name}):")
            trained_models_path = glob.glob(os.path.join(base_path, 'kube_rl_scheduler', 'strategies', 'model', f'{model_name}_*.zip'))
            trained_models_fname = [os.path.basename(trained_model_path) for trained_model_path in trained_models_path]
            trained_models = [ [idx, trained_model_fname, ""] for idx, trained_model_fname in enumerate(trained_models_fname) ]
            print(tabulate(trained_models, headers=["Index", "Trained Model", "Description"]))
            print()
            trained_model_idx = int(input("Select trained model index: "))
            trained_model_fname = trained_models[trained_model_idx][1]
            print('-'*30)
            print(f"Selected trained model: {trained_model_fname}")
            model = model.load(os.path.join(base_path, 'kube_rl_scheduler', 'strategies', 'model', trained_model_fname), env=env, verbose=1)
        else:
            model = model('MlpPolicy', env=env, verbose=1)

        # Train steps
        train_steps = int(input("Train steps(ex. 100000): "))

        # Current setup
        print("Current setup:")
        print(f"Model: {model_name}")
        print(f"Reward: {reward_fname}")
        print(f"Scenario: {scenario_fname}")
        print(f"Train steps: {train_steps}")
        print()

        save_fname = input(f"Save model name (default: {model_name}_{reward_fname.split('.')[0]}_{train_steps}): ")
        log_name = f"log_{save_fname}"
        log_path = f'notebook/training/log/{log_name}'
        logger = configure(log_path, ['stdout', 'csv', 'tensorboard'])
        model.set_logger(logger)

        from stable_baselines3.common.callbacks import EvalCallback
        eval_env = gym.make("SimKubeEnv-v0", reward_file='eval_rbd1.py', scenario_file='scenario-5l-5m-1000p-10m_unbalanced.csv')
        eval_callback = EvalCallback(eval_env, eval_freq=20000, n_eval_episodes=2, deterministic=True, render=False, log_path=log_path)

        start_training = input(f"Start training? (y/n): ")
        if start_training == "y":
            model.learn(total_timesteps=train_steps)
            save_model = input(f"Save model? (y/n): ")
            if save_model == "y":
                save_fname = input(f"Save model name (default: {model_name}_{reward_fname.split('.')[0]}_{train_steps}): ")
                if save_fname == "":
                    save_fname = f"{model_name}_{reward_fname}_{train_steps}"
                    model.save(os.path.join(base_path, 'kube_rl_scheduler', 'strategies', 'model', f'{save_fname}.zip'))
                else:
                    model.save(os.path.join(base_path, 'kube_rl_scheduler', 'strategies', 'model', f'{model_name}_{save_fname}.zip'))
            else:
                print("Back to the beginning")
                continue
        else:
            print("Training canceled")
            exit()