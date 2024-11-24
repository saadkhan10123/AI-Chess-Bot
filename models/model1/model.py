import torch
from torch import nn

class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.linear1 = nn.Linear(12 * 8 * 8 + 4 + 8 + 1, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = self.linear5(x)
        return x