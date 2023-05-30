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

    # print(f"Action: {action}, Availabel nodes: {get_available_nodes(env_prev.cluster)}")

    # # If there's no available node, and takes action 0, then give reward 1 anyway
    # if get_available_nodes(env_prev.cluster) == [0]:
    #     if action == 0:
    #         # print("No available node, but action is 0.. Good!!")
    #         return 1.0
    #     else:
    #         # print("No available node, but action is not 0.. Bad!!")
    #         return -1.0
    
    # Weights for each factor
    w1 = 1
    w2 = 1
    w3 = 1

    default_scheduler = SimHrScheduler(env_prev, 'default.py')
    default_action = default_scheduler.decision(env_prev)
    _, _, _, default_info = env_prev.step(default_action)
    default_cluster = env_prev.cluster
    default_time = time

    reward, rur, rbd1, rbd2 = reward_helper(cluster, action, info, time, debug)
    d_reward, d_rur, d_rbd1, d_rbd2 = reward_helper(default_cluster, default_action, default_info, default_time, debug)

    f_rur = round(rur - d_rur, 2) # The greater the avgUtil, the better the performance
    f_rbd1 = round(rbd1 - d_rbd1, 2) # The greater the imbalance, the worse the performance
    f_rbd2 = round(rbd2 - d_rbd2, 2) # The greater the imbalance, the worse the performance

    if info['is_scheduled'] == False:
        pwd = -0.5
    else:
        pwd = 0

    reward = w1 * f_rur - w2 * f_rbd1 - w3 * f_rbd2 + pwd
    reward = round(reward, 2)

    # print(f"f_rur:{f_rur} / rur:{rur} / d_rur:{d_rur}")
    # print(f"f_rbd1:{f_rbd1} / rbd1:{rbd1} / d_rbd1:{d_rbd1}")
    # print(f"f_rbd2:{f_rbd2} / rbd2:{rbd2} / d_rbd2:{d_rbd2}")
    # print(f"{w1} * {f_rur} - {w2} * {f_rbd1} - {w3} * {f_rbd2} = {reward}")

    return reward


def reward_helper(cluster, action, info, time, debug=False):
    
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
    rbd1 = round(std_cpu + std_mem, 4)

    # rbd2 : Resource balance in each node between cpu and mem (1: best, 0: worst)
    # The worst case should be 1 (e.g. All cpu: 1 and All mem: 0)
    rbd2 = 0
    for node in util:
        rbd2 += abs(util[node]["cpu"] - util[node]["mem"])
    rbd2 = round(rbd2 / len(util), 2)

    w1 = 1
    w2 = 1
    w3 = 1

    reward = w1 * rur - w2 * rbd1 - w3 * rbd2

    return reward, rur, rbd1, rbd2