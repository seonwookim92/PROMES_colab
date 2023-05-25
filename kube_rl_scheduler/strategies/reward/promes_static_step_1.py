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
    # w1 = 1
    # w2 = 5
    # w3 = 5
    # w4 = 1

    # util = {}
    # for node in cluster.nodes:
    #     cpu_ratio, mem_ratio = node.get_node_rsrc_ratio()
    #     util[node.node_name] = {
    #         "cpu": cpu_ratio,
    #         "mem": mem_ratio
    #     }

    # # rur = mean of cpu and mem utilization of all node
    # rur_cpu = round(np.mean([util[node]["cpu"] for node in util]), 2)
    # rur_mem = round(np.mean([util[node]["mem"] for node in util]), 2)
    # rur = round((rur_cpu + rur_mem) / 2, 2)
    # if debug:
    #     print(f"(Stragegy_Default) Avg CPU util: {rur_cpu}")
    #     print(f"(Stragegy_Default) Avg Mem util: {rur_mem}")
    #     print(f"(Stragegy_Default) Avg Util: {rur}")

    # # rbd1 = summation of standard deviation of each resource in all nodes
    # std_cpu = round(np.std([util[node]["cpu"] for node in util]), 2)
    # std_mem = round(np.std([util[node]["mem"] for node in util]), 2)
    # rbd1 = round(std_cpu + std_mem, 2)
    # if debug:
    #     print(f"(Stragegy_Default) Std CPU util: {std_cpu}")
    #     print(f"(Stragegy_Default) Std Mem util: {std_mem}")
    #     print(f"(Stragegy_Default) Imbalance: {rbd1}")
    # # It's penalty
    # rbd1 = - rbd1

    # # rbd2 = Resource balance in each node between cpu and mem (1: best, 0: worst)
    # # The worst case should be 1 (e.g. All cpu: 1 and All mem: 0)
    # rbd2 = 0
    # for node in util:
    #     rbd2 += abs(util[node]["cpu"] - util[node]["mem"])
    # rbd2 = round(rbd2 / len(util), 2)
    # # It's penalty
    # rbd2 = - rbd2
 
    # pwd = Penalty for the wrong decision (-5 for wrong decision, 0 for correct decision)
    avail_nodes = get_available_nodes(cluster)
 
    if info is not None:
        if action != 0 and info['is_scheduled'] == False and info['last_pod'] != None: # 만석입니다.
            pwd = -5
        elif action != 0 and info['is_scheduled'] == False and info['last_pod'] == None: # 헛발질
            pwd = -5
        elif action == 0 and info['is_scheduled'] is None and info['last_pod'] == None: # 기다릴 때를 알아야
            pwd = 5
        elif action == 0 and avail_nodes == [0]: # 기다릴 때를 알아야
            pwd = 5
        elif action == 0 and info['is_scheduled'] is None and info['last_pod'] != None: # 왜 가만히 있니
            pwd = -5
        else: # 정상
            pwd = 0
    else:
        pwd = 0

    # Calculate reward
    # reward = w1 * rur + w2 * rbd1 + w3 * rbd2 + w4 * pwd
    reward = pwd

    # print(f"Reward details...\n\tRUR: {rur}\n\tRBD1: {rbd1}\n\tRBD2: {rbd2}\n\tPWD: {pwd}\n==========\nSUM> {reward}")

    return reward

