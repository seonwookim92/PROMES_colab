import numpy as np

def get_reward(env, cluster, action, is_scheduled, time, debug=False):

    # Resource balance in each node

    util = {}
    for node in cluster.nodes:
        cpu_ratio, mem_ratio = node.get_node_rsrc_ratio()
        util[node.node_name] = {
            "cpu": cpu_ratio,
            "mem": mem_ratio
        }

    # AvgUtil = mean of cpu and mem utilization of all node
    avg_cpu = round(np.mean([util[node]["cpu"] for node in util]), 2)
    avg_mem = round(np.mean([util[node]["mem"] for node in util]), 2)
    avg_util = round((avg_cpu + avg_mem) / 2, 2)

    reward = avg_util

    # if not reward:
    #     reward = 0

    # # Extra statement to get average reward
    # prev_acc_reward = env.acc_reward * env.time
    # reward = (prev_acc_reward + avg_util) / (env.time + 1)

    return reward