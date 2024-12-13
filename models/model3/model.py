class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.linear1 = nn.Linear(12 * 8 * 8 + 4 + 8 + 1, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        self.linear4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        self.linear5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        self.linear6 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.linear1(x)))
        x = torch.relu(self.bn2(self.linear2(x)))
        x = torch.relu(self.bn3(self.linear3(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn4(self.linear4(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn5(self.linear5(x)))
        x = self.dropout3(x)
        x = self.linear6(x)
        # Clamp the output to -50 and 50
        x = torch.clamp(x, -50, 50)
        return x