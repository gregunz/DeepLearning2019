import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import IMAGE_HEIGHT, IMAGE_WIDTH


class SimpleNN(nn.Module):

    def __init__(self, r_l_same_net=False):
        super(SimpleNN, self).__init__()

        self.layer_left = nn.Linear(IMAGE_HEIGHT * IMAGE_WIDTH, 10)
        self.layer2_left = nn.Linear(10, 10)

        if r_l_same_net:
            self.layer_right = self.layer_left
            self.layer2_right = self.layer2_left
        else:
            self.layer_right = nn.Linear(IMAGE_HEIGHT * IMAGE_WIDTH, 10)
            self.layer2_right = nn.Linear(10, 10)

        self.final_layer = nn.Linear(20, 1)

    def forward(self, x):
        left, right = torch.chunk(x, chunks=2, dim=1)
        left = left.view(x.shape[0], -1)
        right = right.view(x.shape[0], -1)

        # left image
        out_left = self.layer_left(left)
        out_left = F.relu(out_left)
        out_left = self.layer2_left(out_left)

        # right image
        out_right = self.layer_right(right)
        out_right = F.relu(out_right)
        out_right = self.layer2_right(out_right)

        out = torch.cat((out_left, out_right), dim=1)
        aux_out = out

        out = F.relu(out)
        out = self.final_layer(out)
        out = torch.sigmoid(out).squeeze(1)

        return out, aux_out


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.fc1 = nn.Linear(16 * 2 * 2, 10)
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
