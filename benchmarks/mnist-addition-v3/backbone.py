# mnist-addition-v3/backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepProbLogMNISTNet(nn.Module):
    """
    CNN architecture used in the DeepProbLog MNIST experiments:

    - 2 conv layers, kernel_size=5, channels 1→6→16
    - Each conv followed by MaxPool2d(2,2)
    - Then 3 fully connected layers: 16*4*4 → 120 → 84 → n_outputs
    - ReLU after all but the last layer; softmax will be applied outside if needed.
    """

    def __init__(self, n_outputs: int = 10):
        super().__init__()
        # Input: [B,1,28,28]
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # -> [B,6,24,24]
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # -> [B,16,8,8]
        self.pool = nn.MaxPool2d(2, 2)                # after each conv: 24→12, 8→4

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,28,28]
        x = self.pool(F.relu(self.conv1(x)))      # [B,6,12,12]
        x = self.pool(F.relu(self.conv2(x)))      # [B,16,4,4]
        x = x.view(x.size(0), -1)                 # [B,16*4*4]
        x = F.relu(self.fc1(x))                   # [B,120]
        x = F.relu(self.fc2(x))                   # [B,84]
        logits = self.fc3(x)                      # [B,n_outputs]
        return logits
