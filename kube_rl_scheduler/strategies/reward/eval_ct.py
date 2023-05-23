import numpy as np

def get_reward(env, cluster, action, is_scheduled, time, debug=False):

    # Return completion time of the pod
    # Just returns the time from 0
    # The acc reward at the last step is the total time of the simulation
    
    return 1