import numpy as np

def get_reward(env, cluster, action, is_scheduled, time, debug=False):

    last_cluster_state = env.last_cluster_state
    last_pod_state = last_cluster_state['pods'][1]

    # Resource balance in each node

    util = {}
    for node in cluster.nodes:
        cpu_ratio, mem_ratio = node.get_node_rsrc_ratio()
        util[node.node_name] = {
            "cpu": cpu_ratio,
            "mem": mem_ratio
        }

    # rbd2 : Resource Difference of the scheduled(tried) node
    if action == 0:
        rbd2 = 0
    else:
        _cpu = 1 - util[cluster.nodes[action-1].node_name]["cpu"] - last_pod_state[0]
        _mem = 1 - util[cluster.nodes[action-1].node_name]["mem"] - last_pod_state[1]
        rbd2 = abs(_cpu - _mem)
        rbd2 = - round(rbd2, 2)

    reward = 1 + rbd2

    # if not reward:
    #     reward = 0

    # # Extra statement to get average reward
    # prev_acc_reward = env.acc_reward * env.time
    # reward = (prev_acc_reward + rbd2) / (env.time + 1)
    
    return reward