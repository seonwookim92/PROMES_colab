import os, sys

base_path = os.getcwd()
if base_path.split('/')[-1] != 'PROMES_colab':
    base_path = os.path.join(base_path, '..')
print(f"Base Path: {base_path}")
sys.path.append(base_path)

import numpy as np
from kube_hr_scheduler.scheduler.sim_hr_scheduler import SimHrScheduler
from kube_hr_scheduler.strategies.model.default import Model

def get_available_nodes(cluster_state):
    if cluster_state['pods'][1] == [0, 0]:
        return [0]
    
    last_pod_state = cluster_state['pods'][1]

    ret = []
    
    if last_pod_state == [0, 0]:
        ret = [i + 1 for i in range(len(cluster_state['nodes']))]
        return ret
    
    for idx, node_idx in enumerate(cluster_state['nodes']):
        node_state = cluster_state['nodes'][node_idx]
        if (node_state[0] - last_pod_state[0] > 0) and (node_state[1] - last_pod_state[1] > 0):
            ret.append(idx+1)
    if not ret:
        ret.append(0)
    return ret

def get_reward(env_prev, cluster, action, info, time, debug=False):

    is_scheduled = info['is_scheduled']
    last_cluster_state = env_prev.last_cluster_state
    last_pod_state = last_cluster_state['pods'][1]
    # print(last_cluster_state)

    if is_scheduled == False:
        pwd =  -1
    elif is_scheduled == None and last_pod_state != [0, 0]:
        if get_available_nodes(last_cluster_state) == [0]:
            pwd = 0
        else:
            pwd =  -1
    else:
        pwd =  0

    # Resource Utilization
    util = {}
    for node in cluster.nodes:
        cpu_ratio, mem_ratio = node.get_node_rsrc_ratio()
        util[node.node_name] = {
            "cpu": cpu_ratio,
            "mem": mem_ratio
        }

    # rbd1 : Resource Balance Degree across nodes
    std_cpu = round(np.std([util[node]["cpu"] for node in util]), 2)
    std_mem = round(np.std([util[node]["mem"] for node in util]), 2)
    rbd1 = (std_cpu ** 2 + std_mem ** 2) ** 0.5
    rbd1 = - round(rbd1, 2)

    # rbd2 : Resource Difference of the scheduled(tried) node
    if action == 0:
        rbd2 = 0
    else:
        _cpu = 1 - util[cluster.nodes[action-1].node_name]["cpu"] - last_pod_state[0]
        _mem = 1 - util[cluster.nodes[action-1].node_name]["mem"] - last_pod_state[1]
        rbd2 = abs(_cpu - _mem)
        rbd2 = - round(rbd2, 2)

    # np.array([1,2]) -> Difference of this elements -> np.diff(np.array([1,2])) -> [1]


    reward = pwd / 3 + rbd1 / 2 + rbd2

    reward = round(reward, 2)

    # print(f"Reward: pwd({pwd}) / 3 + rbd1({rbd1}) / 2 + rbd2({rbd2}) = {reward}")

    return reward
