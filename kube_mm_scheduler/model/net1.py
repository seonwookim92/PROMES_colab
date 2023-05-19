import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
base_path = os.getcwd()
print(f"Base Path: {base_path}")
sys.path.append(base_path)

class Net1(nn.Module):
    def __init__(self, pretrained, freeze):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 6)

        if pretrained:
            # Load pretrained weights
            self.load_state_dict(os.path.join(base_path, "kube_mm_scheduler", "weight", "net1.pt"))

        if freeze:
            # Freeze those weights
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Unfreeze net5
            for param in self.net5.parameters():
                param.requires_grad = True

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x