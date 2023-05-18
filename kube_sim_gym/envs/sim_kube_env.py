import gym
import numpy as np
import importlib
import matplotlib.pyplot as plt

import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(base_path)

from kube_sim_gym.components.cluster import Cluster
from kube_sim_gym.components.pod import Pod
from kube_sim_gym.utils.sim_stress_gen import SimStressGen

# Simulate kubernetes node and pods with cpu, memory resources
class SimKubeEnv(gym.Env):
    def __init__(self, reward_file="try.py", scenario_file="trace2017.csv", n_node=5, cpu_pool=50000, mem_pool=50000, debug=None):
        # self.debug = True if debug == None else debug
        self.debug = False

        # reward
        self.reward_file = reward_file
        self.reward_fn_name = os.path.splitext(self.reward_file)[0]

        self.stress_gen = SimStressGen(scenario_file, self.debug)
        
        self.n_node = n_node
        self.cpu_pool = cpu_pool
        self.mem_pool = mem_pool

        self.cluster = Cluster(n_node, cpu_pool, mem_pool, self.debug)
        self.scheduler_type = 'rl'
        
        self.time = 0
        
        self.reward = 0
        self.done = False
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(n_node * 2 + 2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n_node + 1)

        self.action_map = {'0': 'standby'}
        for i in range(n_node):
            self.action_map[str(i + 1)] = 'node-{}'.format(i+1)

        self.info = {
            'last_pod' : None,
            'is_scheduled' : None
        }

        reward_module_path = os.path.join(f"kube_{self.scheduler_type}_scheduler", "strategies", "reward", self.reward_fn_name).replace('/', '.')
        self.reward_fn = importlib.import_module(reward_module_path)

    def random_state_gen(self):
        """
        This is a extra function to generate random state for data generation.\n
        Do not use it in normal situation, it will screw up the cluster.
        """
        no_pod_thres = 0.8
        no_pod_prob = np.random.random()
        # Randomly generate state
        # Node state can be in range 0 to 1 (rounded to 2 decimal places)
        cpu_pool = self.cluster.nodes[0].spec['cpu_pool']
        mem_pool = self.cluster.nodes[0].spec['mem_pool']

        state = []
        for node in self.cluster.nodes:
            node_cpu_ratio = round(np.random.uniform(0, 1), 2)
            node_mem_ratio = round(np.random.uniform(0, 1), 2)
            state += [node_cpu_ratio, node_mem_ratio]

        # Pending pod state can be in range 0 to 1 (rounded to 2 decimal places)
        if no_pod_prob < no_pod_thres:
            pending_pod_cpu_ratio = round(np.random.uniform(0, 0.5), 2)
            pending_pod_mem_ratio = round(np.random.uniform(0, 0.5), 2)
            state += [pending_pod_cpu_ratio, pending_pod_mem_ratio]
        else:
            state += [0, 0]
        
        # Update cluster
        self.reset()

        for i in range(self.n_node):
            self.cluster.nodes[i].status['cpu_ratio'] = state[i * 2]
            self.cluster.nodes[i].status['mem_ratio'] = state[i * 2 + 1]
            self.cluster.nodes[i].status['cpu_util'] = state[i * 2] * cpu_pool
            self.cluster.nodes[i].status['mem_util'] = state[i * 2 + 1] * mem_pool

        self.cluster.pending_pods = []

        if no_pod_prob < no_pod_thres:
            cpu_quota = state[self.n_node * 2] # * cpu_pool
            mem_quota = state[self.n_node * 2 + 1] # * mem_pool

            pod = Pod([0, 0, 0, cpu_quota, mem_quota], {"cpu_pool":cpu_pool, "mem_pool":mem_pool}, self.debug)

            self.cluster.pending_pods.append(pod)

        return np.array(state, dtype=np.float32)



    def get_reward(self, cluster, action, is_scheduled, time):

        reward = self.reward_fn.get_reward(cluster, action, is_scheduled, time, self.debug)

        return reward

    def get_state(self):
        node_state = []
        for node in self.cluster.nodes:
            node_cpu_ratio = node.get_node_rsrc_ratio()[0]
            node_mem_ratio = node.get_node_rsrc_ratio()[1]
            node_state += [node_cpu_ratio, node_mem_ratio]

        if  self.cluster.pending_pods:
            pending_pod = self.cluster.pending_pods[0]
            pending_pod_state = [pending_pod.spec["cpu_ratio"], pending_pod.spec["mem_ratio"]]
        else:
            pending_pod_state = [0, 0]

        if self.debug:
            print(f"(SimKubeEnv) Pending Pod State: {pending_pod_state}")
            print(f"(SimKubeEnv) Node state: {node_state}")

        state = node_state + pending_pod_state

        return np.array(state, dtype=np.float32)
    
    def get_done(self):
        len_scenario = len(self.stress_gen.scenario)
        len_scheduled = len(self.cluster.terminated_pods + self.cluster.running_pods)
        if len_scenario == len_scheduled:
            self.done = True
        else:
            self.done = False
        return self.done

    def step(self, action):
        self.time += 1
        is_scheduled = None
        
        new_pod_spec = self.stress_gen.create_pod(self.time)
        node_spec = self.cluster.nodes[0].spec
        if new_pod_spec:
            self.cluster.queue_pod(new_pod_spec, node_spec)

        # Update cluster
        self.cluster.update(self.time)

        # Initialize info
        self.info = {
            'last_pod' : None,
            'is_scheduled' : None
        }


        # Do action
        pending_pods = self.cluster.pending_pods

        if pending_pods:
            try:
                deploy_node = self.cluster.get_node(self.action_map[str(action)])
            except:
                deploy_node = None
            if deploy_node:
                if self.debug:
                    print(f"(SimKubeEnv) Deploying pod to node {deploy_node.node_name}")
                pending_pod = pending_pods[0]
                is_scheduled = self.cluster.deploy_pod(pending_pod, deploy_node, self.time)
                if is_scheduled:
                    if self.debug:
                        print(f"(SimKubeEnv) Pod deployed to node {deploy_node.node_name}")
                else:
                    if self.debug:
                        print(f"(SimKubeEnv) Failed to deploy pod to node {deploy_node.node_name}")
                # for pod in self.cluster.pending_pods:
                #     if self.cluster.deploy_pod(pod, deploy_node, self.time):
                #         break
                self.info = {
                    'last_pod' : pending_pod,
                    'is_scheduled' : is_scheduled
                }
            else:
                if self.debug:
                    print(f"(SimKubeEnv) Standby")
        else:
            if self.debug:
                print(f"(SimKubeEnv) No pending pods")

        # Get reward
        self.reward = self.get_reward(self.cluster, action, is_scheduled, self.time)

        # Get state
        state = self.get_state()

        # Get done
        self.done = self.get_done()

        return state, self.reward, self.done, self.info

    def reset(self):
        self.time = 0
        self.cluster.reset()
        self.stress_gen.reset()

        return self.get_state()
