import gym
import time
from time import sleep
import subprocess

import warnings
warnings.filterwarnings("ignore")

import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(base_path)

import kube_real_gym, kube_hr_scheduler, kube_stress_generator
from kube_real_gym.envs.real_kube_env import RealKubeEnv

env = gym.make('RealKubeEnv-v0')
envv = RealKubeEnv()
done = False

action_map = {
    0: "standby",
    1: "node-1",
    2: "node-2",
    3: "node-3",
    4: "node-4",
    5: "node-5",
}


state = env.reset()
sleep(1)

print("Start stress generator")
env.start_stress_gen()
sleep(1)

turn = 0
while done == False:
    turn += 1
    print("Turn: ", turn)
    action = env.action_space.sample()
    # print("Action: ", action)
    state, reward, done, _ = env.step(action)
    # print("State: ", state)
    # print("Reward: ", reward)
    # print("Done: ", done)
    print("================================")
    sleep(1)

print("End of simulation")