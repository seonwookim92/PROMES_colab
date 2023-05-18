import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(base_path)

import gym
import numpy as np
import matplotlib.pyplot as plt

from kube_rl_scheduler.scheduler.sim_rl_scheduler import SimRlScheduler
from kube_hr_scheduler.scheduler.sim_hr_scheduler import SimHrScheduler
from kube_sim_gym.envs.sim_kube_env import SimKubeEnv

env = gym.make('SimKubeEnv-v0', reward_file="try.py", scenario_file="scenario-10l-3m-1000p-10m.csv")

hr_scheduler_default = SimHrScheduler(env, model_fname='default.py')
hr_scheduler_random = SimHrScheduler(env, model_fname='random.py')
hr_scheduler_roundRobin = SimHrScheduler(env, model_fname='roundRobin.py')
rl_scheduler_ppo = SimRlScheduler(env, model_fname='PPO_1M.zip')
scheduler = {
    'hr_default': hr_scheduler_default,
    'hr_random': hr_scheduler_random,
    'hr_roundRobin': hr_scheduler_roundRobin,
    'rl_ppo': rl_scheduler_ppo
}

# Collect data for each scheduler regarding the number of pods scheduled at each time step
cdf = {
    'hr_default': [],
    'hr_random': [],
    'hr_roundRobin': [],
    'rl_ppo': []
}

def get_num_pods_scheduled(env):
    scheduled_pods = env.cluster.running_pods + env.cluster.terminated_pods
    return len(scheduled_pods)

for key in cdf.keys():

    env.reset()
    while not env.get_done():
        env.step(scheduler[key].decision(env))
        cdf[key].append(get_num_pods_scheduled(env))

    env.reset()

# Draw cdf graph for each scheduler regarding the number of pods scheduled at each time step
plt.figure(figsize=(15, 15))
for key in cdf.keys():
    plt.plot(cdf[key], label=key)

plt.legend()
plt.show()
