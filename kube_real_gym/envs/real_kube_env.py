import gym
from gym import spaces
from time import sleep
import numpy as np
import subprocess
import signal

import warnings
warnings.filterwarnings("ignore")

import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(base_path)

# from kube_real_gym.utils.monitor import Monitor
# from kube_real_gym.utils.unit_matcher import *
from utils.real_utils.monitor import Monitor
from utils.real_utils.unit_converter import *
from utils.real_utils.deployer import Deployer

from kube_hr_scheduler.scheduler.real_hr_scheduler import RealHrScheduler
# from kube_rl_scheduler.scheduler.real_rl_scheduler import RealRlScheduler

from kube_stress_generator.stress_gen import StressGen

class RealKubeEnv(gym.Env):
    def __init__(self):
        super(RealKubeEnv, self).__init__()
        self.monitor = Monitor()
        self.deployer = Deployer()

        # Nodes
        self.node_list = self.monitor.get_nodes()[0]
        # If there are controlplane and node-0, remove them
        if "controlplane" in self.node_list:
            self.node_list.remove("controlplane")
        if "node-0" in self.node_list:
            self.node_list.remove("node-0")
        self.num_nodes = len(self.node_list)

        # Initialize node observation space Should be the number of nodes x 3 matrix
        self.node_observation_space = spaces.Box(low=0, high=100, shape=(self.num_nodes * 2,), dtype=int)

        # Initialize the most recent pending pod observation space
        self.pod_observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=int)

        # Initialize the observation space
        self.observation_space = spaces.Tuple((self.node_observation_space, self.pod_observation_space))

        # Map node name to index
        self.idx_to_node = {}
        for i, node_name in enumerate(self.node_list):
            self.idx_to_node[i] = node_name

        # Initialize the action space
        self.action_space = spaces.Discrete(self.num_nodes + 1)

        # Number of scheduling pods in scenario
        self.num_pods_in_scenario = 0

        # Signal handler
        signal.signal(signal.SIGINT, self.ctrl_c_handler)
        self.stress_gen_pid = None

    def ctrl_c_handler(self, signum, frame):
        print("Terminate all processes...")
        os.kill(self.stress_gen_pid, signum)
        exit(0)

    def start_stress_gen(self, silent="False", scenario_file="scenario-2023-02-27.csv"):
        # Start stress generator in a separate process
        # TODO Need to elaborate arguments part
        p_stress_gen = subprocess.Popen(["python", os.path.join(base_path,"kube_stress_generator/main.py"), silent, scenario_file])
        print("Stress generator PID: " + str(p_stress_gen.pid))
        self.stress_gen_pid = p_stress_gen.pid

        # Get number of scheduling pods in scenario
        with open(os.path.join(base_path, "kube_stress_generator/scenarios", scenario_file), "r") as f:
            self.num_pods_in_scenario = len(f.readlines())

        print("Number of scheduling pods in scenario: " + str(self.num_pods_in_scenario))
        


    def get_pending_pod(self, debug=False):
        # Get the most recent pending pod
        pending_pods_names = self.monitor.get_pending_pods()[0]
        if debug:
            print("Pending Pod: " + str(pending_pods_names))
        if len(pending_pods_names) == 0:
            pending_pod_name = ""
            pending_pod_obs = np.array([0, 0])
            return pending_pod_name, pending_pod_obs
        else:
            pending_pod_name = pending_pods_names[0]
            if debug:
                print("Pending Pod Name: " + str(pending_pod_name))
            pending_pod_rqsts = self.monitor.get_pod_rqsts(pending_pod_name)

            if debug:
                print("Pending Pod Requests: " + str(pending_pod_rqsts))

            node_cpu_cap = self.monitor.get_node_rsrc(self.node_list[0])["cpu"][1]
            node_memory_cap = self.monitor.get_node_rsrc(self.node_list[0])["memory"][1]
            if debug:
                print("Node CPU Capacity: " + str(node_cpu_cap))
                print("Node Memory Capacity: " + str(node_memory_cap))

            pending_pod_cpu_rqst = max(int(pending_pod_rqsts["cpu"] / node_cpu_cap * 100), 1)
            pending_pod_memory_rqst = max(int(pending_pod_rqsts["memory"] / node_memory_cap * 100), 1)

            pending_pod_obs = np.array([pending_pod_cpu_rqst, pending_pod_memory_rqst])
            if debug:
                print("Pending Pod Observation: " + str(pending_pod_obs))

            return pending_pod_name, pending_pod_obs

    def get_reward(self, debug=False):
        # Utilization of resources on each node
        util = {}
        for node in self.node_list:
            _util = self.monitor.get_node_rsrc(node)
            if debug:
                print(_util)
            util[node] = {
                "cpu": 100 - _util["cpu"][0] / _util["cpu"][1] * 100,
                "memory": 100 - _util["memory"][0] / _util["memory"][1] * 100
            }

        # AvgUtil = mean of cpu and mem utilization of all node
        avg_cpu = round(np.mean([util[node]["cpu"] for node in self.node_list]),3)
        avg_mem = round(np.mean([util[node]["memory"] for node in self.node_list]),3)
        avg_util = round((avg_cpu + avg_mem) / 2 , 3)
        if debug:
            print("AvgCPU: " + str(avg_cpu))
            print("AvgMemory: " + str(avg_mem))
            print("AvgUtil: " + str(avg_util))

        # ImBalance = summation of standard deviation of each resource in all nodes
        std_cpu = round(np.std([util[node]["cpu"] for node in self.node_list]),3)
        std_mem = round(np.std([util[node]["memory"] for node in self.node_list]),3)
        imbalance = std_cpu + std_mem
        if debug:
            print("StdCPU: " + str(std_cpu))
            print("StdMem: " + str(std_mem))
            print("Imbalance: " + str(imbalance))

        # Reward = a*AvgUtil - b*ImBalance
        a = 1
        b = 1
        reward = round(a * avg_util - b * imbalance, 3)

        print("Reward: " + str(reward))

        return reward
    
    def get_state(self, debug=False):
        
        pending_pod_name, pending_pod_obs = self.get_pending_pod(debug=debug)

        # Get the resource utilization of each node
        node_obs = []
        for node in self.node_list:
            node_rsrc = self.monitor.get_node_rsrc(node)

            node_cpu_util = node_rsrc["cpu"][3]
            node_memory_util = node_rsrc["memory"][3]

            # node_obs.append([node_cpu_util, node_memory_util])
            node_obs += [node_cpu_util, node_memory_util]

        if debug:
            print("Node Observation: " + str(node_obs))

        state = (node_obs, pending_pod_obs)

        if debug:
            print("State: " + str(state))

        return state
    
    def get_done(self, debug=False):
        done = True

        # Check number of jobs
        if len(self.monitor.get_jobs()[0]) != self.num_pods_in_scenario:
            done = False

        # If all pods are scheduled, then done
        _, pods = self.monitor.get_pending_pods()
        if len(pods) != 0:
            done = False

        if debug:
            print("Done: " + str(done))

        return done


    def step(self, action):
        # Get first pending pod
        # pod_name = self.monitor.get_pending_pods()[0][0]
        pod_name, _ = self.get_pending_pod()

        # Action map
        action_map = {
            0: "standby",
            1: "node-1",
            2: "node-2",
            3: "node-3",
            4: "node-4",
            5: "node-5",
        }

        if action != 0 and pod_name != "":
            node_name = action_map[action]

            # Take an action in the environment based on the provided action
            print("Action: " + str(action))
            print("Pod Name: " + str(pod_name))
            print("Node Name: " + str(node_name))

            # self.scheduler.scheduling(pod_name, node_name)
            self.deployer.deploy(pod_name, node_name)

            while pod_name not in self.monitor.get_pods_in_node(node_name)[0]:
                print("Waiting for pod to be scheduled...")
                sleep(1)
            sleep(3)
        
        else:
            print("Action: " + str(action))
            print("Pod Name: " + str(pod_name))
            print("Node Name: " + str(""))
            print("Nothing to do...")

        # Observe the state of the environment
        state = self.get_state()
        print("State: " + str(state))

        # Calculate the reward
        reward = self.get_reward()
        print("Reward: " + str(reward))

        # Check if the episode is done ===> Set to False for now
        done = self.get_done()

        # Return the state, reward, and done
        return state, reward, done, {}

    def reset(self):
        # kubectl delete jobs --all command
        os.system("kubectl delete jobs --all")
        
        # Check every 10 seconds if all jobs are deleted
        while True:
            sleep(10)
            if len(self.monitor.get_jobs()[0]) == 0:
                break
            else:
                print("Waiting for jobs to be deleted...")

        sleep(3)
        state = self.get_state()


if __name__ == "__main__":
    env = RealKubeEnv()

    print("real_kube_env.py is running...")
    print(f"Node Observation Space: {env.node_observation_space}")
    print(f"Pod Observation Space: {env.pod_observation_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    print("Observing state...")
    state = env.get_state(debug=True)
    print("Calculating reward...")
    reward = env.get_reward(debug=True)