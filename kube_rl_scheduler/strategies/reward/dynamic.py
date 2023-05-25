import os, sys

base_path = os.getcwd()
if base_path.split('/')[-1] != 'PROMES_colab':
    base_path = os.path.join(base_path, '..')
print(f"Base Path: {base_path}")
sys.path.append(base_path)

import numpy as np
from kube_hr_scheduler.scheduler.sim_hr_scheduler import SimHrScheduler
from kube_hr_scheduler.strategies.model.default import Model

def get_reward(env_prev, cluster, action, info, time, debug=False):

    # print(f"Action: {action}")
    # print(f"Info: {info}")
    # print(f"Time: {time}")

    # Weights for each factor
    w1 = 1
    w2 = 2
    w3 = 2
    w4 = 1
    w5 = 1

    default_scheduler = SimHrScheduler(env_prev, 'default.py')
    default_action = default_scheduler.decision(env_prev)
    _, _, _, default_info = env_prev.step(default_action)
    default_cluster = env_prev.cluster
    default_time = time

    # if debug:
    # print('=================================')
    # print('PROMES decision')
    # print(f"PROMES Action: {action}")
    # for node in cluster.nodes:
    #     print(f"Node {node.node_name}: {len(node.status['running_pods'])}")
    # print(f"info : {info['last_pod'].pod_name} / {info['is_scheduled']}")
    # print(f"time : {time}")
    # print('---------------------------------')
    # print('Default decision')
    # print(f"Default action: {default_action}")
    # for node in default_cluster.nodes:
    #     print(f"Node {node.node_name}: {len(node.status['running_pods'])}")
    # print(f"info : {default_info['last_pod'].pod_name} / {default_info['is_scheduled']}")
    # print(f"time : {default_time}")
    # print('=================================')

    reward, rur, rbd1, rbd2, pwd, prg = reward_helper(cluster, action, info, time, debug)
    d_reward, d_rur, d_rbd1, d_rbd2, d_pwd, d_prg = reward_helper(default_cluster, default_action, default_info, default_time, debug)

    # Prevent zero division
    if d_prg == 0:
        prg += 1
        d_prg += 1

    if debug:
        print(f"Reward: {reward}")
        print(f"Default Reward: {d_reward}")

    # f_rur = round(rur / d_rur - 1, 4)
    f_rur = round(rur, 4)

    if rbd1 == 0:
        f_rbd1 = round((d_rbd1 + 0.01) / (rbd1 + 0.01) -1, 4)
    else:
        f_rbd1 = round(d_rbd1 / rbd1 -1, 4)
    if rbd2 == 0:
        f_rbd2 = round((d_rbd2 + 0.01) / (rbd2 + 0.01) -1, 4)
    else:
        f_rbd2 = round(d_rbd2 / rbd2 -1, 4)
    f_pwd = round(pwd - d_pwd, 4)
    if d_prg == 0:
        f_prg = round((prg + 1) / (d_prg + 1) -1, 4)  
    else:
        f_prg = round(prg / d_prg - 1, 4) # If prg > d_prg, f_prg > 0, else if prg < d_prg, f_prg < 0

    # if debug:
    # print(f"Action: {action}, Default Action: {default_action}")
    # print(f"Reward details: rur({f_rur}: {rur}), rbd1({f_rbd1}: {d_rbd1} / {rbd1}), rbd2({f_rbd2}: {rbd2} / {d_rbd2}), pwd({f_pwd}: {pwd} - {d_pwd}), prg({f_prg}: {prg} / {d_prg} - 1)")

    # Rescale the factors : Set the first float number to locate at the first decimal place
    # Maybe needed???? 일단 보류

    # reward = reward - d_reward + f_prg
    reward = w1 * f_rur + w2 * f_rbd1 + w3 * f_rbd2 + w4 * f_pwd + w5 * f_prg

    # if debug:
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

    # rur = mean of cpu and mem utilization of all node
    rur_cpu = round(np.mean([util[node]["cpu"] for node in util]), 4)
    rur_mem = round(np.mean([util[node]["mem"] for node in util]), 4)
    rur = round((rur_cpu + rur_mem) / 2, 4)
    if debug:
        print(f"(Stragegy_Default) Avg CPU util: {rur_cpu}")
        print(f"(Stragegy_Default) Avg Mem util: {rur_mem}")
        print(f"(Stragegy_Default) Avg Util: {rur}")

    # rbd1 = summation of standard deviation of each resource in all nodes
    std_cpu = round(np.std([util[node]["cpu"] for node in util]), 4)
    std_mem = round(np.std([util[node]["mem"] for node in util]), 4)

    rbd1 = round(std_cpu + std_mem, 4)
    # if debug:
        # print(f"(Stragegy_Default) Std CPU util: {std_cpu}")
        # print(f"(Stragegy_Default) Std Mem util: {std_mem}")
        # print(f"(Stragegy_Default) Imbalance: {rbd1}")

    # rbd2 = Resource balance in each node between cpu and mem (1: best, 0: worst)
    # The worst case should be 1 (e.g. All cpu: 1 and All mem: 0)
    rbd2 = 0
    for node in util:
        rbd2 += abs(util[node]["cpu"] - util[node]["mem"])
        # print(f"{node}: {util[node]['cpu']} - {util[node]['mem']} = {abs(util[node]['cpu'] - util[node]['mem'])}")
    # rbd2 = round(rbd2 / len(util), 4)
    # rbd2 = 1 - rbd2

    # prg = The number of scheduled pods over time (1: best, 0: worst) - Competitive reward only
    prg = len(cluster.running_pods) + len(cluster.terminated_pods)

    # app = Average pending pods over time (1: best, 0: worst) - Competitive reward only
    
    # pwd = Penalty for the wrong decision (-5 for wrong decision, 0 for correct decision)
    if info is not None:
        if action != 0 and info['is_scheduled'] == False and info['last_pod'] != None: # 만석입니다.
            pwd = -5
        elif action != 0 and info['is_scheduled'] == False and info['last_pod'] == None: # 헛발질
            pwd = -5
        elif action == 0 and info['is_scheduled'] is None and info['last_pod'] != None: # 왜 안움직여
            pwd = -5
        else:
            pwd = 0
    else:
        pwd = 0

    rur = round(rur, 4)
    rbd1 = round(rbd1, 4)
    rbd2 = round(rbd2, 4)
    pwd = round(pwd, 4)

    # reward = w1 * rur + w2 * rbd1 + w3 * rbd2 + w4 * pwd # + w5 * prg
    _reward = rur + rbd1 + rbd2 + pwd # + prg

    return _reward, rur, rbd1, rbd2, pwd, prg