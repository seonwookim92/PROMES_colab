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
    w1 = 3
    w2 = 3
    w3 = 1

    default_scheduler = SimHrScheduler(env_prev, 'default.py')
    default_action = default_scheduler.decision(env_prev)
    _, _, _, default_info = env_prev.step(default_action)
    default_cluster = env_prev.cluster
    default_time = time


    reward, rbd1, rbd2, prg = reward_helper(cluster, action, info, time, debug)
    d_reward, d_rbd1, d_rbd2, d_prg = reward_helper(default_cluster, default_action, default_info, default_time, debug)

    # f_rbd1 = round(d_rbd1 / rbd1, 4) - 1 if rbd1 != 0 else round((d_rbd1 + 0.01) / (rbd1 + 0.01), 4) - 1
    # f_rbd2 = round(d_rbd2 / rbd2, 4) - 1 if rbd2 != 0 else round((d_rbd2 + 0.01) / (rbd2 + 0.01), 4) - 1
    f_rbd1 = d_rbd1 - rbd1
    f_rbd2 = d_rbd2 - rbd2
    f_prg = prg - d_prg

    f_rbd1 = round(f_rbd1, 4)
    f_rbd2 = round(f_rbd2, 4)
    f_prg = round(f_prg, 4)

    # reward = reward - d_reward + f_prg
    reward = w1 * f_rbd1 + w2 * f_rbd2 + w3 * f_prg

    # If it does as good as default scheduler, give 0.1 for each factor
    if info['is_scheduled'] == False:
        # print(0)
        reward = -1.0
    elif f_rbd1 == 0 and f_rbd2 == 0 and f_prg == 0:
        # print(1)
        reward = 0.5
    # Or if it outperforms default scheduler, give 0.5 for each factor
    elif f_rbd1 > 0 and f_rbd2 > 0 and f_prg >= 0:
        # print(2)
        reward = 1.0
    # If reward is too low (less than 0.01), then multiply it by 10
    elif (reward < 0.01 and reward > 0) or (reward > -0.01 and reward < 0) :
        # print(3)
        reward *= 10
    elif f_prg < 0: # If it does worse than default scheduler, give 0.1 for each factor
        # print(4)
        reward = -0.5
    elif reward > 1:
        # print(5)
        reward = 1.0
    elif reward < -1:
        # print(6)
        reward = -1.0
    else:
        # print(7)
        reward = round(reward, 4)

    # print(f"Reward details: rbd1({f_rbd1}: {d_rbd1} - {rbd1}), rbd2({f_rbd2}: {d_rbd2} - {rbd2}), prg({f_prg}: {prg} - {d_prg})")
    # print(f"Reward is {reward}")
    
    return reward


def reward_helper(cluster, action, info, time, debug=False):
    
    util = {}
    for node in cluster.nodes:
        cpu_ratio, mem_ratio = node.get_node_rsrc_ratio()
        util[node.node_name] = {
            "cpu": cpu_ratio,
            "mem": mem_ratio
        }

    # rbd1 = summation of standard deviation of each resource in all nodes
    std_cpu = round(np.std([util[node]["cpu"] for node in util]), 4)
    std_mem = round(np.std([util[node]["mem"] for node in util]), 4)
    rbd1 = round(std_cpu + std_mem, 4)

    # rbd2 = Resource balance in each node between cpu and mem (1: best, 0: worst)
    # The worst case should be 1 (e.g. All cpu: 1 and All mem: 0)
    rbd2 = 0
    for node in util:
        rbd2 += abs(util[node]["cpu"] - util[node]["mem"])
    rbd2 = round(rbd2 / len(util), 4)

    # prg = The number of scheduled pods over time (1: best, 0: worst) - Competitive reward only
    prg = len(cluster.running_pods) + len(cluster.terminated_pods)

    # app = Average pending pods over time (1: best, 0: worst) - Competitive reward only
    
    rbd1 = round(rbd1, 4)
    rbd2 = round(rbd2, 4)

    # reward = w1 * rur + w2 * rbd1 + w3 * rbd2 + w4 * pwd # + w5 * prg
    _reward = rbd1 + rbd2 + prg # + prg

    return _reward, rbd1, rbd2, prg