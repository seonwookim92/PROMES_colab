from kubernetes import client, config

import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(base_path)

# from kube_python_scheduler.common.monitor import Monitor
# from kube_python_scheduler.common.filter import Filter
from utils.real_utils.monitor import Monitor
from kube_hr_scheduler.stratigies.fcfs.strategy import FCFS

import random

class Deployer:
    def __init__(self, cfg=None, strategy=FCFS()):
        # Load the Kubernetes configuration
        if cfg is None:
            self.config = config.load_kube_config()
        else:
            self.config = cfg

        # Load the Kubernetes API client
        self.core_api = client.CoreV1Api()
        self.monitor = Monitor(cfg=self.config)
        self.strategy = strategy
    
    def deploy(self, pod_name, node_name):
        pod = self.monitor.get_pod(pod_name)
        # Check if the pod is already scheduled
        if pod.status.phase == "Pending":
            print(f"Pod [{pod_name}] is scheduled to node [{node_name}]")
            try:
                # Binding the pod to the node
                body = client.V1Binding(
                    metadata=client.V1ObjectMeta(
                        name=pod_name,
                        namespace="default"
                    ),
                    target=client.V1ObjectReference(
                        api_version="v1",
                        kind="Node",
                        name=node_name,
                        namespace="default"
                    )
                )
                self.core_api.create_namespaced_binding(
                    body=body,
                    namespace="default"
                )
            except Exception as e:
                # print(f"Exception when calling CoreV1Api->create_namespaced_binding: {e}")
                pass
        else:
            print(f"Pod {pod_name} is already scheduled to node {pod.spec.node_name}")
