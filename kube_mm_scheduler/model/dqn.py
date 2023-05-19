import os, sys

base_path = os.getcwd()
print(f"Base Path: {base_path}")
sys.path.append(base_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from kube_mm_scheduler.model.net3 import Net3
from kube_mm_scheduler.model.net5 import Net5

class Net3_(Net3):
    def __init__(self):
        super(Net3_, self).__init__()
        self.fc4 = None

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1_1(x1))  
        x2 = F.relu(self.fc1_2(x2))
        x = torch.cat((x1, x2), dim=1) 
        x = F.relu(self.fc2(x))  
        x = F.relu(self.fc3(x))
        return x

class Net5_(Net5):
    def __init__(self):
        super(Net5_, self).__init__()
        self.fc3_1 = None
        self.fc3_2 = None
        self.fc3_3 = None
        self.fc3_4 = None
        self.fc3_5 = None

    def forward(self, x1, x2):
        x12 = self.net3_(x1, x2)

        x3_1 = F.relu(self.fc1_3_1(x1[:, :2]))
        x3_2 = F.relu(self.fc1_3_2(x1[:, 2:4]))
        x3_3 = F.relu(self.fc1_3_3(x1[:, 4:6]))
        x3_4 = F.relu(self.fc1_3_4(x1[:, 6:8]))
        x3_5 = F.relu(self.fc1_3_5(x1[:, 8:10]))

        x4_1 = torch.cat((x12, x3_1), dim=1)
        x4_2 = torch.cat((x12, x3_2), dim=1)
        x4_3 = torch.cat((x12, x3_3), dim=1)
        x4_4 = torch.cat((x12, x3_4), dim=1)
        x4_5 = torch.cat((x12, x3_5), dim=1)

        x5_1 = self.fc2_1(x4_1)
        x5_2 = self.fc2_2(x4_2)
        x5_3 = self.fc2_3(x4_3)
        x5_4 = self.fc2_4(x4_4)
        x5_5 = self.fc2_5(x4_5)

        x6 = torch.cat((x5_1, x5_2, x5_3, x5_4, x5_5), dim=1) # 16 * 5 = 80

        return x6

        

# DQN model
# Based on net5, make a DQN model outputting 6 possible actions' Q-values
class DQN(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(DQN, self).__init__()
        self.net5_ = Net5_()
        # self.net5 = Net5()

        if pretrained:
            # Load pretrained weights
            net5_state_dict = torch.load(os.path.join(base_path, "kube_mm_scheduler", "weight", "net5.pt"))
            net5_state_dict.pop('fc3_1.weight')
            net5_state_dict.pop('fc3_1.bias')
            net5_state_dict.pop('fc3_2.weight')
            net5_state_dict.pop('fc3_2.bias')
            net5_state_dict.pop('fc3_3.weight')
            net5_state_dict.pop('fc3_3.bias')
            net5_state_dict.pop('fc3_4.weight')
            net5_state_dict.pop('fc3_4.bias')
            net5_state_dict.pop('fc3_5.weight')
            net5_state_dict.pop('fc3_5.bias')
            self.net5_.load_state_dict(net5_state_dict)
            # self.net5.load_state_dict(net5_state_dict)

        if freeze:
            # Freeze net5
            for param in self.net5_.parameters():
            # for param in self.net5.parameters():
                param.requires_grad = False
        else:
            # Unfreeze net5
            for param in self.net5_.parameters():
            # for param in self.net5.parameters():
                param.requires_grad = True

        # self.fc4 = nn.Linear(5, 16)        
        # self.fc5 = nn.Linear(16, 6)

        # self.fc4 = nn.Linear(5, 6)
        
        self.fc3 = nn.Linear(80, 20)
        self.fc4 = nn.Linear(20, 6)        

    def forward(self, x1, x2):
        x = self.net5_(x1, x2)
        # x = self.net5(x1, x2)

        # x = F.relu(self.fc4(x))
        # x = self.fc5(x)
        
        # x = self.fc4(x)

        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x