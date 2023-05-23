import numpy as np

def get_reward(env, cluster, action, is_scheduled, time, debug=False):

    # Simple reward function
    # 1. Advantage or Penalty on more pods on a node.
    # 2. Penalty on the failure of scheduling
    # 3. Penalty on the pending pods which are not scheduled.
    # 4. Advantage if it stand by when there is no pending pods.

    # Initialize factors
    factor1_gap_n_pod = 0
    factor2_is_scheduled = 0
    factor3_is_pending = 0
    factor4_standby = 0

    # Default metric fetch
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

    pending_pods = cluster.pending_pods.copy()
    # If a pending pod is just listed(=just arrived), do not include it in the pending pods.
    if pending_pods:
        for pod in pending_pods:
            if pod.spec['arrival_time'] == time:
                pending_pods.remove(pod)

    # 1. Advantage or Penalty on more pods on a node
    n_pod_per_node = [len(node.status['running_pods']) for node in cluster.nodes]
    # Subtract 1 from the node deployed the pod
    if is_scheduled:
        n_pod_per_node[action-1] -= 1

    if debug:
        print(f"n_pod_per_node : {n_pod_per_node}")

    avg_n_pod_per_node = round(np.mean(n_pod_per_node), 2)

    # 0 0 0 0 -> avg=0 => if choose 0 : 0-0=0 -> no reward
    # 0 1 1 0 -> avg=0.5 -> if choose 1 -> 0.5-1=-0.5 / if choose 0 -> 0.5-0=0.5
    # 0 2 4 6 -> avg=3 => if choose 0 : 3-0=3 / if choose 2 : 3-2=1 / if choose 4 : 3-4=-1 / if choose 6 : 3-6=-3
    if action != 0: # If the action is not standby,
        factor1_gap_n_pod = avg_n_pod_per_node - n_pod_per_node[action-1]
        factor1_gap_n_pod = round(factor1_gap_n_pod, 2)

    # 2. Penalty on the failure of scheduling
    # If the pending pod is just listed, do not give disadvantage
    if not is_scheduled and action != 0:
        factor2_is_scheduled = -1

    # 3. Penalty on the pending pods which are not scheduled.
    if pending_pods:
        if pending_pods[0].spec['arrival_time'] < time:
            factor3_is_pending = pending_pods[0].spec['arrival_time'] - time

    # 4. Advantage if it stand by when there is no pending pods.
    if action == 0 and not pending_pods:
        factor4_standby = 1

    # Reward = a * factor1_gap_n_pod + b * factor2_is_scheduled + c * factor3_is_pending + d * factor4_standby
    a = 1
    b = 1
    c = 1
    d = 1

    reward = a * factor1_gap_n_pod + b * factor2_is_scheduled + c * factor3_is_pending + d * factor4_standby
    
    if debug:
        print(f"factor1_gap_n_pod : {factor1_gap_n_pod}")
        print(f"factor2_is_scheduled : {factor2_is_scheduled}")
        print(f"factor3_is_pending : {factor3_is_pending}")
        print(f"factor4_standby : {factor4_standby}")
        print(f"{a}*factor1 + {b}*factor2 + {c}*factor3 + {d}*factor4 = {reward}")

    return reward