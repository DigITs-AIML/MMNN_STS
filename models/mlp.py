import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class MLP(nn.Module):

    def __init__(self, in_channels=1, out_channels=3, feature_channels=12, dropout_prob=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_channels = feature_channels
        self.dropout_prob = dropout_prob
        self.features = None

        self.relu = nn.ReLU()

        self.backbone = nn.Sequential(
            OrderedDict([
                ('dense0', nn.Linear(self.in_channels, 32)),
                ('bn0', nn.BatchNorm1d(32)),
                ('relu0', self.relu),
                ('drop0', nn.Dropout1d(dropout_prob)),
                ('dense1', nn.Linear(32, 16, 3)),
                ('bn1', nn.BatchNorm1d(16)),
                ('drop1', nn.Dropout1d(dropout_prob)),
                ('relu1', self.relu),
                ('dense2', nn.Linear(16, 8, 3)),
                ('bn2', nn.BatchNorm1d(8)),
                ('drop2', nn.Dropout1d(dropout_prob)),
                ('relu2', self.relu),
                ('dense3', nn.Linear(8, 8, 3)),
                ('bn3', nn.BatchNorm1d(8)),
                ('drop3', nn.Dropout1d(dropout_prob)),
                ('relu3', self.relu),
                ('dense4', nn.Linear(8, 8, 3)),
                ('bn4', nn.BatchNorm1d(8)),
                ('drop4', nn.Dropout1d(dropout_prob)),
                ('relu4', self.relu)
            ])
        )

        self.features = nn.Sequential(
            OrderedDict([
                ('dense5', nn.Linear(8, self.feature_channels)),
                ('bn5', nn.BatchNorm1d(self.feature_channels)),
                ('drop5', nn.Dropout1d(dropout_prob)),
                ('relu5', self.relu)
            ])
        )

        self.output_head = nn.Sequential(
            OrderedDict([
                ('dense6', nn.Linear(self.feature_channels, self.out_channels)),
            ])
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.features(x)
        x = self.output_head(x)

        return x 