# from kubernetes import client, config

import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(base_path)

# from kube_python_scheduler.common.monitor import Monitor
# from kube_python_scheduler.common.utils import convert_cpu_unit, convert_memory_unit

from utils.sim_utils.monitor import Monitor

class Filter:
    def __init__(self):
        self.monitor = Monitor()

        # Load the Kubernetes configuration
        # self.config = config.load_kube_config()
        # self.monitor = Monitor(cfg=self.config)
        # self.pending_pods = self.monitor.get_pending_pods()
        # self.running_pods = self.monitor.get_pods("Running")
        # self.nodes = self.monitor.get_nodes()

    def check_available(self, env, pod_name, node_name, debug=False):  
        # Get the node resources
        node_rsrc = self.monitor.get_node_rsrc(env, node_name)

        # Get the pod requests
        pod_rqsts = self.monitor.get_pod_rqsts(env, pod_name)

        # Check if the node has enough resources to run the pod
        # cpu_check = node_rsrc["cpu"][1] >= pod_rqsts["cpu"]
        # memory_check = node_rsrc["memory"][1] >= pod_rqsts["memory"]
        cpu_check = node_rsrc["cpu"][1] >= pod_rqsts["cpu_ratio"]
        mem_check = node_rsrc["mem"] >= pod_rqsts["mem_ratio"]
        # pod_cap_check = int(node_rsrc["pod_cap"][0]) >= 1

        # Print the result
        if debug:
            print(f"Node {node_name} availablity check:")
            print(f"CPU request: {pod_rqsts['cpu']}m, available: {node_rsrc['cpu'][1]}m")
            print(f"Memory request: {pod_rqsts['memory']}Ki, available: {node_rsrc['memory'][1]}Ki")
            # print(f"Pod capacity request: 1, available: {node_rsrc['pod_cap'][0]}")

        return cpu_check and mem_check # and pod_cap_check
    
    def get_available_nodes_name(self, env, pod_name):
        # Get all the nodes that have enough resources to run the pod
        # Return a list of nodes that have enough resources to run the pod
        nodes_name, _ = self.monitor.get_nodes(env)
        available_nodes_name = []
        for node_name in nodes_name:
            if self.check_available(pod_name, node_name):
                available_nodes_name.append(node_name)
        return available_nodes_name
