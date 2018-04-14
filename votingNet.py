import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable


class singleNet(nn.Module):

    def __init__(self, n_nodes=20):
        
        super(singleNet, self).__init__()
        self.n_nodes = n_nodes
        self.fc1     = nn.Linear(28*28, n_nodes)
        self.fc2     = nn.Linear(n_nodes, n_nodes)  
        
    def forward(self,x):
        x = x.view(x.shape[0], 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
        


class NetGram(nn.Module):
    
    def __init__(self, stdvW, num_nets=30, num_out=10, dim_out=16, num_iterations=3):

        super(NetGram, self).__init__()        
        self.nets = nn.ModuleList([singleNet() for _ in range(num_nets)])
        self.route_weights = nn.Parameter(torch.randn(num_out, num_nets, self.nets[0].n_nodes, dim_out))
        self.num_iterations = num_iterations

        
    def forward(self,x):
        outputs = torch.cat([self.squash(net(x)[:,None,:]) for net in self.nets], dim=1)
        #route_weight
        #10x30x16x20 -> 1x10x30x16x20

        #output
        #5x30x20     -> 5x1 x30x1x20
        priors = torch.matmul(outputs[:, None, :, None, :], self.route_weights[None,:, :, :, :])
        #print priors.shape
        
        logits = Variable(torch.zeros(*priors.size()))
        if torch.cuda.is_available():
                logits = logits.cuda()

        for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=1)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        return outputs.squeeze()
    
    def post_process(self):
        pass
    
    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)



if __name__ == "__main__":

    #net  = singleNet()
    net  = NetGram(30)
    input = torch.randn(5, 28*28)
    input = Variable(input)
    output = net(input)
