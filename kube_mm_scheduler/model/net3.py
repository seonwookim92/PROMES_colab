import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
base_path = os.getcwd()
print(f"Base Path: {base_path}")
sys.path.append(base_path)

class Net3(nn.Module):
    def __init__(self, pretrained, freeze):
        super(Net3, self).__init__()
        self.fc1_1 = nn.Linear(10, 16) # 5 Nodes status (CPU, Memory)
        self.fc1_2 = nn.Linear(2, 16)   # Pod quota (CPU, Memory)
        self.fc2 = nn.Linear(32, 16)    # Concatenated vector
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 6)     # 6-sized vector

        if pretrained:
            # Load pretrained weights
            self.load_state_dict(torch.load(os.path.join(base_path, "kube_mm_scheduler", "weight", "net3.pt")))

        if freeze:
            # Freeze those weights
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Unfreeze net5
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1_1(x1))  
        x2 = F.relu(self.fc1_2(x2))
        x = torch.cat((x1, x2), dim=1) 
        x = F.relu(self.fc2(x))  
        x = F.relu(self.fc3(x))
        x = self.fc4(x)     
        # x = F.softmax(self.fc3(x), dim=1)
        # x = torch.argmax(x, dim=1)
        return x