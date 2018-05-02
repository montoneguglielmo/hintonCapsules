import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable
from convMean import *
import time


from gugliNet2 import netCaps

if __name__ == "__main__":


    cnet = netCaps(stdWfc=10.0, stdWconv=10.0)
    cnet.post_process()

    if torch.cuda.is_available():
        cnet.cuda()

    
    # #Gradient time estimation
    n_batches =  3
    batch_sz  =  5

    input = torch.randn(batch_sz, 1, 28, 28)
    if torch.cuda.is_available():
        input = input.cuda()

    input  = Variable(input)

    t_start = time.time()
    for cnt in range(n_batches):
         output = cnet(input)
         #print torch.sum(output)
         torch.sum(output).backward()

    t_elaps = (time.time() - t_start)/float(n_batches)
    print('Time to evaluate gradient for one batch of size %d: %.2f (s)' % (batch_sz, t_elaps))
