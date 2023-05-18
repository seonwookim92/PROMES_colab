from kubernetes import client, config

import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(base_path)

import importlib

# from kube_python_scheduler.common.monitor import Monitor
# from kube_python_scheduler.common.filter import Filter
from utils.real_utils.monitor import Monitor
from utils.real_utils.filter import Filter
# from kube_hr_scheduler.stratigies.fcfs.strategy import FCFS

from kube_real_gym.envs.real_kube_env import RealKubeEnv

import random

class RealHrScheduler:
    def __init__(self, cfg=None, model_fname="random.py"):
        # Load the Kubernetes configuration
        if cfg is None:
            self.config = config.load_kube_config()
        else:
            self.config = cfg

        self.env = RealKubeEnv()

        # Load the Kubernetes API client
        self.core_api = client.CoreV1Api()
        self.monitor = Monitor(cfg=self.config)
        self.filter = Filter()

        self.model_name = model_fname.split('.')[0]
        model_path = os.path.join("kube_hr_scheduler", "strategies", "model", self.model_name).replace("/", ".")
        self.model = importlib.import_module(model_path).Model(self.env)
        # self.strategy = strategy

    def decision(self, env, debug=False):
        
        return self.model.predict(env)




        # output = self.strategy.scoring()

        # if debug:
        #     print(f"scoring output: {output}")
        # pod = output['pod']
        # node_score = output['node_score']
        # print(f"pod: {pod}\nnode_score: {node_score}")
        
        # if pod is None:
        #     return (None, None)
        # else:
        #     # Get the node with the highest score
        #     # Randomly choose if there are multiple nodes with the same score
        #     max_score = max(node_score.values())
        #     max_score_nodes = [node for node, score in node_score.items() if score == max_score]
        #     node = random.choice(max_score_nodes)
        #     return (pod, node)

    
    # def scheduling(self, pod_name, node_name):
    #     pod = self.monitor.get_pod(pod_name)
    #     # Check if the pod is already scheduled
    #     if pod.status.phase == "Pending":
    #         print(f"Pod [{pod_name}] is scheduled to node [{node_name}]")
    #         try:
    #             # Binding the pod to the node
    #             body = client.V1Binding(
    #                 metadata=client.V1ObjectMeta(
    #                     name=pod_name,
    #                     namespace="default"
    #                 ),
    #                 target=client.V1ObjectReference(
    #                     api_version="v1",
    #                     kind="Node",
    #                     name=node_name,
    #                     namespace="default"
    #                 )
    #             )
    #             self.core_api.create_namespaced_binding(
    #                 body=body,
    #                 namespace="default"
    #             )
    #         except Exception as e:
    #             # print(f"Exception when calling CoreV1Api->create_namespaced_binding: {e}")
    #             pass
    #     else:
    #         print(f"Pod {pod_name} is already scheduled to node {pod.spec.node_name}")
