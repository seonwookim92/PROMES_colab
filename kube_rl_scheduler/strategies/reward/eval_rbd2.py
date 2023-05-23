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

    rbd2_cpu = 0
    rbd2_mem = 0
    for node in util:
        rbd2_cpu += abs(util[node]["cpu"] - avg_cpu)
        rbd2_mem += abs(util[node]["mem"] - avg_mem)

    rbd2_cpu = round(rbd2_cpu / len(util), 2)
    rbd2_mem = round(rbd2_mem / len(util), 2)

    rbd2 = round((rbd2_cpu + rbd2_mem) / 2, 2)
    rbd2 = 1 - rbd2

    return rbd2
