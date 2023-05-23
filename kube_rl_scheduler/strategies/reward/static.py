import numpy as np

def get_reward(env, cluster, action, is_scheduled, time, debug=False):

    # Weights for each factor
    w1 = 1
    w2 = 1
    w3 = 1
    w4 = 1
    
    util = {}
    for node in cluster.nodes:
        cpu_ratio, mem_ratio = node.get_node_rsrc_ratio()
        util[node.node_name] = {
            "cpu": cpu_ratio,
            "mem": mem_ratio
        }

    # rur = mean of cpu and mem utilization of all node
    rur_cpu = round(np.mean([util[node]["cpu"] for node in util]), 2)
    rur_mem = round(np.mean([util[node]["mem"] for node in util]), 2)
    rur = round((rur_cpu + rur_mem) / 2, 2)
    if debug:
        print(f"(Stragegy_Default) Avg CPU util: {rur_cpu}")
        print(f"(Stragegy_Default) Avg Mem util: {rur_mem}")
        print(f"(Stragegy_Default) Avg Util: {rur}")

    # rbd1 = summation of standard deviation of each resource in all nodes
    std_cpu = round(np.std([util[node]["cpu"] for node in util]), 2)
    std_mem = round(np.std([util[node]["mem"] for node in util]), 2)
    rbd1 = round(std_cpu + std_mem, 2)
    if debug:
        print(f"(Stragegy_Default) Std CPU util: {std_cpu}")
        print(f"(Stragegy_Default) Std Mem util: {std_mem}")
        print(f"(Stragegy_Default) Imbalance: {rbd1}")

    # rbd2 = Resource balance in each node between cpu and mem (1: best, 0: worst)
    # The worst case should be 1 (e.g. All cpu: 1 and All mem: 0)
    rbd2 = 0
    for node in util:
        rbd2 += abs(util[node]["cpu"] - util[node]["mem"])
    rbd2 = round(rbd2 / len(util), 2)
    rbd2 = 1 - rbd2

    # prg = The number of scheduled pods over time (1: best, 0: worst) - Competitive reward only

    # app = Average pending pods over time (1: best, 0: worst) - Competitive reward only
    
    # pwd = Penalty for the wrong decision (-5 for wrong decision, 0 for correct decision)
    if is_scheduled:
        pwd = 0
    else:
        pwd = -5

    reward = w1 * rur + w2 * rbd1 + w3 * rbd2 + w4 * pwd

    return reward