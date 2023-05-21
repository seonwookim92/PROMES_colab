import os, sys

base_path = os.path.join(os.getcwd())
print(f"Base Path: {base_path}")
sys.path.append(base_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from kube_mm_scheduler.model.net3 import Model as Net3



# class Net3(nn.Module):
#     def __init__(self):
#         super(Net3, self).__init__()
#         self.fc1_1 = nn.Linear(10, 16) # 5 Nodes status (CPU, Memory)
#         self.fc1_2 = nn.Linear(2, 16)   # Pod quota (CPU, Memory)
#         self.fc2 = nn.Linear(32, 16)    # Concatenated vector
#         self.fc3 = nn.Linear(16, 16)
#         self.fc4 = nn.Linear(16, 6)     # 6-sized vector

#     def forward(self, x1, x2):
#         x1 = F.relu(self.fc1_1(x1))  
#         x2 = F.relu(self.fc1_2(x2))
#         x = torch.cat((x1, x2), dim=1) 
#         x = F.relu(self.fc2(x))  
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)     
#         # x = F.softmax(self.fc3(x), dim=1)
#         # x = torch.argmax(x, dim=1)
#         return x

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

# Reuse Net4 to process input1 and input2
# We will take 5 nodes' state which is same as input3
# And will concatenate each output with the output from Net3_
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net3_ = Net3_()

        self.fc1_3_1 = nn.Linear(2, 16)   # Node1 state (CPU, Memory)
        self.fc1_3_2 = nn.Linear(2, 16)   # Node2 state (CPU, Memory)
        self.fc1_3_3 = nn.Linear(2, 16)   # Node3 state (CPU, Memory)
        self.fc1_3_4 = nn.Linear(2, 16)   # Node4 state (CPU, Memory)
        self.fc1_3_5 = nn.Linear(2, 16)   # Node5 state (CPU, Memory)

        # Node 1
        self.fc2_1 = nn.Linear(32, 16)
        self.fc3_1 = nn.Linear(16, 1)

        # Node 2
        self.fc2_2 = nn.Linear(32, 16)
        self.fc3_2 = nn.Linear(16, 1)

        # Node 3
        self.fc2_3 = nn.Linear(32, 16)
        self.fc3_3 = nn.Linear(16, 1)

        # Node 4
        self.fc2_4 = nn.Linear(32, 16)
        self.fc3_4 = nn.Linear(16, 1)

        # Node 5
        self.fc2_5 = nn.Linear(32, 16)
        self.fc3_5 = nn.Linear(16, 1)

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

        x5_1 = F.relu(self.fc2_1(x4_1))
        x5_2 = F.relu(self.fc2_2(x4_2))
        x5_3 = F.relu(self.fc2_3(x4_3))
        x5_4 = F.relu(self.fc2_4(x4_4))
        x5_5 = F.relu(self.fc2_5(x4_5))

        x6_1 = self.fc3_1(x5_1)
        x6_2 = self.fc3_2(x5_2)
        x6_3 = self.fc3_3(x5_3)
        x6_4 = self.fc3_4(x5_4)
        x6_5 = self.fc3_5(x5_5)

        x7 = torch.cat((x6_1, x6_2, x6_3, x6_4, x6_5), dim=1)
        return x7