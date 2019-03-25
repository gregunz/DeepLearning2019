import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_HEIGHT = 14
IMAGE_WIDTH = 14

class Baseline(nn.Module):

    def __init__(self):
        super(Baseline, self).__init__()
        self.layer_left = nn.Linear(IMAGE_HEIGHT * IMAGE_WIDTH, 10)
        self.layer2_left = nn.Linear(10, 10)
        self.layer_right = nn.Linear(IMAGE_HEIGHT * IMAGE_WIDTH, 10)
        self.layer2_right = nn.Linear(10, 10)
        self.final_layer = nn.Linear(20, 1)
        
        self.relu = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        left, right = torch.chunk(x, chunks=2, dim=1)
        left = left.squeeze(1).view(x.shape[0], -1)
        right = right.squeeze(1).view(x.shape[0], -1)

        # left image
        out_left = self.layer_left(left)
        out_left = self.relu(out_left)
        out_left = self.layer2_left(out_left)
        out_left = self.relu(out_left)

        # right image
        out_right = self.layer_left(right)
        out_right = self.relu(out_right)
        out_right = self.layer2_left(out_right)
        out_right = self.relu(out_right)

        out = torch.cat((out_left, out_right), dim=1)
        out = self.final_layer(out)
        out = self.out_act(out).squeeze(1)
        return out
