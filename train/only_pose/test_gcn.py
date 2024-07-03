import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv

# Define the GCN model
class PoseGCN(nn.Module):
    def __init__(self, num_joints=17, num_features=3, hidden_channels=64, num_classes=16):
        super(PoseGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels * num_joints, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create a model instance
model = PoseGCN()

im = torch.randn((8, 15, 512, 512))
# # Class for pose 
num_joints = 17
batch_size, num_person, num_frames = 8, 1, 60

# #model.init_weights()
feature_pose = torch.randn(batch_size, num_person,
                     num_frames, num_joints, 3)

# # feature = torch.randn((8, 68))
y = model(im, feature_pose)
print(y.shape )

