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

    # rbd1 : Resource Balance Degree across nodes
    std_cpu = round(np.std([util[node]["cpu"] for node in util]), 2)
    std_mem = round(np.std([util[node]["mem"] for node in util]), 2)
    rbd1 = (std_cpu ** 2 + std_mem ** 2) ** 0.5
    rbd1 = - round(rbd1, 2)

    reward = 1 + rbd1

    # if not reward:
    #     reward = 0

    # # Extra statement to get average reward
    # prev_acc_reward = env.acc_reward * env.time
    # reward = (prev_acc_reward + rbd1) / (env.time + 1)
    
    return reward