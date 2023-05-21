import torch
import random
import importlib

import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(base_path)

class SimMmScheduler:
    def __init__(self, model='promes'):

        model_path = os.path.join("kube_mm_scheduler", "model", model).replace("/", ".")
        weight_path = os.path.join(base_path, "kube_mm_scheduler", "weight", f"{model}.pt") #.replace("/", ".")

        self.model = importlib.import_module(model_path).Model(True, True)
        # self.model.load_state_dict(torch.load(weight_path))

        self.model.eval()

    def decision(self, env):
        state = env.get_state()

        # If no pending pod, return 0
        if state[-1] == 0:
            return 0

        # Divide state into node state and pod quota
        node_state = state[:10]
        pod_quota = state[10:]

        # Convert to torch tensor
        node_state = torch.tensor(node_state, dtype=torch.float32)
        pod_quota = torch.tensor(pod_quota, dtype=torch.float32)

        # Unsqueeze to add batch dimension
        node_state = node_state.unsqueeze(0)
        pod_quota = pod_quota.unsqueeze(0)

        # Predict the score for each action
        output = self.model(node_state, pod_quota)
        print(f"output: {output}")

        # Argmax
        max_indices = torch.where(output == torch.max(output))[1]
        # Randomly choose one from max_indices
        max_idx = random.choice(max_indices)
        # print(f"max_idx: {max_idx}")

        # Convert to int
        action = int(max_idx) + 1
        print(f"action: {action}")

        return action