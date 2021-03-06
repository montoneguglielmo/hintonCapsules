import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable
import time

class CapsuleLayer(nn.Module):
    
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
            #print self.route_weights.shape
            #self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
            
            #stdv = 1. / np.sqrt(float(10000))
            #self.route_weights.data.uniform_(-stdv, stdv)
            
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = torch.matmul(x[None, :, :, None, :], self.route_weights[:, None, :, :, :])
            
            logits = Variable(torch.zeros(*priors.size()))
            if torch.cuda.is_available():
                logits = logits.cuda()

            for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=0)
                #print "max:", probs.max(), "min:", probs.min(), "mena:", probs.mean()# this should be dim=0
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class NetGram(nn.Module):
    def __init__(self, stdvW):
        super(NetGram, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=10, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        stdv = 1. / np.sqrt(float(stdvW))#np.sqrt(float(10000))
        self.digit_capsules.route_weights.data.uniform_(-stdv, stdv)

        
    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        print "here"
        print x.shape
        print "here"
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        return x

    def post_process(self):
        pass

    


if __name__ == "__main__":

    cnet = NetGram(stdvW=1e4)
    input = torch.randn(5, 1, 28, 28)
    input = Variable(input)
    output = cnet(input)

    #print output.shape
    
    #Gradient time estimation
    n_batches =  3
    batch_sz  =  100

    input = torch.randn(batch_sz, 1, 28, 28)
    input  = Variable(input)

    t_start = time.time()
    for cnt in range(n_batches):
         output = cnet(input)
         torch.sum(output).backward()

    t_elaps = (time.time() - t_start)/float(n_batches)
    print('Time to evaluate gradient for one batch of size %d: %.2f (s)' % (batch_sz, t_elaps))


    
