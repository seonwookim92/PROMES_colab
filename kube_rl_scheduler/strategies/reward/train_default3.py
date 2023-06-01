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
    w1 = 0.5
    w2 = 0.5

    util = {}
    for node in cluster.nodes:
        cpu_ratio, mem_ratio = node.get_node_rsrc_ratio()
        util[node.node_name] = {
            "cpu": cpu_ratio,
            "mem": mem_ratio
        }

    util_array = np.array([util[node]["cpu"] for node in util] + [util[node]["mem"] for node in util])

    # rur : Mean of the overall utilization of all nodes
    rur = round(np.mean(util_array), 2)

    # rbd1 : Resource std across nodes
    std_cpu = round(np.std([util[node]["cpu"] for node in util]), 2)
    std_mem = round(np.std([util[node]["mem"] for node in util]), 2)
    rbd1 = (std_cpu ** 2 + std_mem ** 2) ** 0.5
    rbd1 = round(rbd1, 2)
    rbd1 = 1 - rbd1

    # rbd2 : Resource balance in each node between cpu and mem (1: best, 0: worst)
    # The worst case should be 1 (e.g. All cpu: 1 and All mem: 0)
    rbd2_ = []
    for node in util:
        rbd2_.append(abs(util[node]["cpu"] - util[node]["mem"]))
    rbd2 = sum([d ** 2 for d in rbd2_]) ** 0.5


    # rbd3 : Resource balance in each node between cpu and mem (1: best, 0: worst)


    # Calculate reward
    reward = w1 * cpu_score + w2 * mem_score

    return reward