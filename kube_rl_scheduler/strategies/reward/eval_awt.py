import numpy as np

def get_reward(env, cluster, action, is_scheduled, time, debug=False):

    # Accumulative delay of scheduling
    # If there are no pending pods, return 0
    # If there are pending pods, each pending pod has a delay of 1
    # If the pending pod is just listed, do not give disadvantage

    reward = 0
    pending_pods = cluster.pending_pods.copy()
    # If a pending pod is just listed(=just arrived), do not include it in the pending pods.
    if pending_pods:
        for pod in pending_pods:
            if pod.spec['arrival_time'] == time:
                pending_pods.remove(pod)

    if pending_pods:
        reward = len(pending_pods)

    return reward