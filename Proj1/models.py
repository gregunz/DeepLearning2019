import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import IMAGE_HEIGHT, IMAGE_WIDTH


class SimpleFC(nn.Module):

    def __init__(self):
        super(SimpleFC, self).__init__()

        self.layer1 = nn.Linear(IMAGE_HEIGHT * IMAGE_WIDTH, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(20, 1)

    def fc_forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

    def forward(self, x):
        left, right = torch.chunk(x, chunks=2, dim=1)

        out_left = self.fc_forward(left)
        out_right = self.fc_forward(right)

        out = torch.cat((out_left, out_right), dim=1)
        aux_out = out

        out = F.relu(out)
        out = self.layer3(out)
        out = torch.sigmoid(out).squeeze(1)

        return out, aux_out


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 8, 3)
        self.fc1 = nn.Linear(8 * 2 * 2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(20, 1)

    def cnn_forward(self, x):
        # Convolution #1 with relu
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        # Convolution #2 with relu
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        # Reshaping BxCxHxW to a vector Bx(C*H*W) for fc
        x = x.view(x.shape[0], -1)

        # Fully connected #1
        x = F.relu(self.fc1(x))

        # Fully connected #2
        x = self.fc2(x)
        return x

    def forward(self, x):
        left, right = torch.chunk(x, chunks=2, dim=1)

        out_left = self.cnn_forward(left)
        out_right = self.cnn_forward(right)

        out = torch.cat((out_left, out_right), dim=1)
        aux_out = out

        out = F.relu(out)
        # Fully connected #2 with sigmoid
        out = torch.sigmoid(self.fc3(out)).squeeze(1)
        return out, aux_out
