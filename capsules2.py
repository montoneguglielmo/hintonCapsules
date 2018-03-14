import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import copy
import numpy as np

class primaryCapsule(nn.Module):

    def __init__(self, in_channels, out_channels, num_capsules, kernel_size, stride=1, padding=0, dilation=1):
        super(primaryCapsule, self).__init__()

        self.capsules = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding) for _ in range(num_capsules)])

        
    def forward(self, x):
        outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.squash(outputs)
        return outputs
        
    @staticmethod
    def squash(tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


class digitCapsule(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None, num_iterations=3):
        super(digitCapsule, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))

        
    @staticmethod
    def squash(tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    
    def forward(self, x):
        print x.shape
        priors = torch.matmul(x[None, :, :, None, :], self.route_weights[:, None, :, :, :])
        print priors.shape
        logits = Variable(torch.zeros(*priors.size()))
        if torch.cuda.is_available():
            logits = logits.cuda()
        for i in range(self.num_iterations):
            probs = F.softmax(logits, dim=0)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

            print outputs.shape
            if i != self.num_iterations - 1:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits

        return outputs    

    
class capsNet(nn.Module):
    def __init__(self):
        super(capsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = primaryCapsule(in_channels=256, out_channels=32, num_capsules=8, kernel_size=9, stride=2)
        self.digit_capsules = digitCapsule(num_capsules=10, num_route_nodes=32 * 6 * 6, in_channels=8,out_channels=16)
        
    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        return x

    
if __name__ == "__main__":

    cnet = capsNet()
    input = torch.randn(5, 1, 28, 28)
    input = Variable(input)
    output = cnet(input)
    print output.shape
    
