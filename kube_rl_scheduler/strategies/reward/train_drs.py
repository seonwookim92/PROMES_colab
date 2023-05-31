import os, sys

base_path = os.getcwd()
if base_path.split('/')[-1] != 'PROMES_colab':
    base_path = os.path.join(base_path, '..')
print(f"Base Path: {base_path}")
sys.path.append(base_path)

import numpy as np
from kube_hr_scheduler.scheduler.sim_hr_scheduler import SimHrScheduler
from kube_hr_scheduler.strategies.model.default import Model

def get_available_nodes(cluster):
    if not cluster.pending_pods:
        return [0]

    pod = cluster.pending_pods[0]
    ret = []
    
    for idx, node in enumerate(cluster.nodes):
        cpu_avail = node.spec["cpu_pool"] - node.status["cpu_util"]
        mem_avail = node.spec["mem_pool"] - node.status["mem_util"]
        if pod.spec["cpu_req"] <= cpu_avail and pod.spec["mem_req"] <= mem_avail:
            ret.append(idx+1)
    if not ret:
        ret.append(0)
    return ret

def get_reward(env_prev, cluster, action, info, time, debug=False):

    # Weights for each factor
    w1 = 1
    w2 = 3
    w3 = 3
    # w4 = 1

    util = {}
    for node in cluster.nodes:
        cpu_ratio, mem_ratio = node.get_node_rsrc_ratio()
        util[node.node_name] = {
            "cpu": cpu_ratio,
            "mem": mem_ratio
        }

    util_array = np.array([util[node]["cpu"] for node in util] + [util[node]["mem"] for node in util])
    
    # AvgUtil : Mean of the overall utilization of all nodes
    avgUtil = round(np.mean(util_array), 2)

    # Imbalance : Standard deviation of the utilization of all nodes
    imbalance = round(np.std(util_array), 2)

    # Reward : alpha * avgUtil - beta * imbalance
    alpha = 1
    beta = 1

    reward = alpha * avgUtil - beta * imbalance
    reward = round(reward, 2)

    print(f"Reward: {alpha} * {avgUtil} - {beta} * {imbalance} = {reward}")

    return reward