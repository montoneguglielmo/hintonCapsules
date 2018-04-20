import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable
from convMean import *
import time

class convCapsuleLayer(nn.Module):

    def __init__(self,n_inp_caps, n_out_caps, dim_inp, dim_inp_caps, dim_out_caps, flt_sz, num_iterations):

        super(convCapsuleLayer, self).__init__()
        self.route_weights = nn.Parameter(torch.randn(n_out_caps, n_inp_caps, dim_inp, dim_inp, dim_inp_caps, dim_out_caps))
        
        self.inds  = range(0,dim_inp, flt_sz)
        self.num_iterations = num_iterations
        self.convMean = convMean(n_inp_caps, 1, flt_sz, stride=flt_sz)
        self.upSample = nn.Upsample(scale_factor=flt_sz)
        self.flt_sz = flt_sz

    def forward(self, x):
        priors = torch.matmul(x[:,None, :, :, :, None, :], self.route_weights[None, :, :, :, :, :, :])
        priors = priors.squeeze()
        priors = priors.permute(0,1,5,2,3,4).contiguous()
        logits = Variable(torch.zeros(priors.shape[0], priors.shape[1], 1, priors.shape[3], priors.shape[4], priors.shape[5]))

        if torch.cuda.is_available():
            logits = logits.cuda()

        print priors.shape
        for i in range(self.num_iterations):
            probs           = F.softmax(logits, dim=1)
            prior_weight    = probs*priors
            prior_conv      = prior_weight.view(-1, priors.shape[-3], priors.shape[-2], priors.shape[-1])
            print prior_conv.shape
            prior_conv      = self.convMean(prior_conv)
            prior_conv_mean = self.upSample(prior_conv)
            print prior_conv_mean.shape, probs.shape
            prior_conv_mean = prior_conv_mean.view(probs.shape[0], probs.shape[1], probs.shape[2], 1, probs.shape[4], probs.shape[5])
            prior_conv_mean = self.squash(prior_conv_mean, 2)
            delta_logits    = torch.sum(prior_conv_mean * priors, dim=2, keepdim=True)
            logits          = logits + delta_logits
            
        output = prior_conv.view(probs.shape[0], probs.shape[1], probs.shape[2], 1, prior_conv.shape[-2], prior_conv.shape[-1])
        return self.squash(output, 2) 


    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale        = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


    def post_process(self):
        route_mean  = torch.zeros(self.route_weights[:, :, :self.flt_sz, :self.flt_sz, :, :].shape)
        if torch.cuda.is_available():
            route_mean = route_mean.cuda()
            
        n_filter    = 0
        for indy in self.inds:
            for indx in self.inds:
               route_mean += self.route_weights.data[:, :, indx:indx+self.flt_sz, indy:indy+self.flt_sz, :, :]
               n_filter += 1

        route_mean /= float(n_filter) 

        for indy in self.inds:
            for indx in self.inds:
                self.route_weights.data[:, :, indx:indx+self.flt_sz, indy:indy+self.flt_sz, :, :] =  route_mean        


        
class startCapsuleLayer(nn.Module):
    
    def __init__(self, dim_out_capsules, n_inp_filters, n_capsules, kernel_size=None, stride=None):
        super(startCapsuleLayer, self).__init__()
        self.capsules = nn.ModuleList([nn.Conv2d(n_inp_filters, n_capsules, kernel_size=kernel_size, stride=stride, padding=0) for _ in range(dim_out_capsules)])

        self.dim_out_capsules = dim_out_capsules
        self.n_inp_filters    = n_inp_filters
        self.n_capsules       = n_capsules
        self.kernel_size      =  kernel_size
        self.stride           = stride
        
    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        outputs = [capsule(x) for capsule in self.capsules]
        outputs = torch.stack(outputs, dim=-1)
        outputs = self.squash(outputs, dim=4)
        return outputs

    def printInfo(self):
        n_params = (self.kernel_size**2)*self.n_inp_filters*self.dim_out_capsules*self.n_capsules
        return n_params



class fcCapsuleLayer(nn.Module):
    
    def __init__(self, n_out_caps, n_inp_caps, dim_inp_capsules, dim_out_capsules, num_iterations=3):
        super(fcCapsuleLayer, self).__init__()
        self.num_iterations = num_iterations
        self.route_weights = nn.Parameter(torch.randn(n_out_caps, n_inp_caps, dim_inp_capsules, dim_out_capsules))
        
    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        priors = torch.matmul(x[:, None, :, None, :], self.route_weights[None,:, :, :, :])
        logits = Variable(torch.zeros(*priors.size()))

        if torch.cuda.is_available():
            logits = logits.cuda()

        for i in range(self.num_iterations):
            probs        = F.softmax(logits, dim=1)
            outputs      = self.squash((probs * priors).sum(dim=2, keepdim=True))
            delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
            logits       = logits + delta_logits
            
        return outputs


    
class NetGram(nn.Module):
    
    def __init__(self, stdvWconv, stdvWffw, flt_sz=9):
        super(NetGram, self).__init__()
        self.flt_sz           = flt_sz
        self.conv1            = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=flt_sz, stride=1)
        self.primary_capsules = startCapsuleLayer(dim_out_capsules=8, n_inp_filters=256, n_capsules=32, kernel_size=flt_sz, stride=1)

        dim_inp               = 28 - 2 * (flt_sz-1)
        self.conv_capsules    = convCapsuleLayer(n_inp_caps=32, n_out_caps=20, dim_inp=dim_inp, dim_inp_caps=8, dim_out_caps=16, flt_sz=2, num_iterations=3)
        n_inp_caps            = ((28 - 2 * (flt_sz-1)) / 2)**2 * 20
        self.fc_capsules      = fcCapsuleLayer(n_out_caps=10, n_inp_caps=n_inp_caps, dim_inp_capsules=16, dim_out_capsules=20, num_iterations=3)

        stdvWffw  = np.sqrt(float(stdvWffw))
        stdvWconv = np.sqrt(float(stdvWconv))
        
        self.conv_capsules.route_weights.data.uniform_(-stdvWconv, stdvWconv)
        self.fc_capsules.route_weights.data.uniform_(-stdvWffw, stdvWffw)

        
    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.conv_capsules(x)
        x = x.squeeze()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.fc_capsules(x)
        x = x.squeeze()
        return x

    def post_process(self):
        self.conv_capsules.post_process()




class NetGram2(nn.Module):
    
    def __init__(self, stdvWconv, stdvWffw, flt_sz=5):
        super(NetGram2, self).__init__()
        self.flt_sz           = flt_sz
        self.primary_capsules = startCapsuleLayer(dim_out_capsules=4, n_inp_filters=1, n_capsules=32, kernel_size=flt_sz, stride=1)

        dim_inp                = 28 - flt_sz + 1
        self.conv_capsules0    = convCapsuleLayer(n_inp_caps=32, n_out_caps=20, dim_inp=dim_inp, dim_inp_caps=4, dim_out_caps=8, flt_sz=2, num_iterations=3)

        dim_inp                = dim_inp/2
        self.conv_capsules1    = convCapsuleLayer(n_inp_caps=20, n_out_caps=15, dim_inp=dim_inp, dim_inp_caps=8, dim_out_caps=16, flt_sz=2, num_iterations=3)

        n_inp_caps            = ((dim_inp/2)**2)*15
        self.fc_capsules      = fcCapsuleLayer(n_out_caps=10, n_inp_caps=n_inp_caps, dim_inp_capsules=16, dim_out_capsules=20, num_iterations=3)

        stdvWffw  = np.sqrt(float(stdvWffw))
        stdvWconv = np.sqrt(float(stdvWconv))
        
        self.conv_capsules0.route_weights.data.uniform_(-stdvWconv, stdvWconv)
        self.conv_capsules1.route_weights.data.uniform_(-stdvWconv, stdvWconv)
        self.fc_capsules.route_weights.data.uniform_(-stdvWffw, stdvWffw)

    def forward(self, x):
        x = self.primary_capsules(x)
        x = self.conv_capsules0(x)
        x = x.squeeze()
        x = x.permute(0,1,3,4,2).contiguous()
        x = self.conv_capsules1(x)
        x = x.squeeze()
        x = x.permute(0,1,3,4,2).contiguous()
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.fc_capsules(x)
        x = x.squeeze()
        return x
    
    def post_process(self):
        self.conv_capsules0.post_process()
        self.conv_capsules1.post_process()
        
if __name__ == "__main__":

    cnet = NetGram2(stdvWconv=1., stdvWffw=1e5)
    cnet.post_process()

    input = torch.randn(5, 1, 28, 28)
    input = Variable(input)
    output = cnet(input)

    print output.shape

    # #Gradient time estimation
    # n_batches =  3
    # batch_sz  =  100

    # input = torch.randn(batch_sz, 1, 28, 28)
    # input  = Variable(input)

    # t_start = time.time()
    # for cnt in range(n_batches):
    #      output = cnet(input)
    #      #print torch.sum(output)
    #      torch.sum(output).backward()

    # t_elaps = (time.time() - t_start)/float(n_batches)
    # print('Time to evaluate gradient for one batch of size %d: %.2f (s)' % (batch_sz, t_elaps))



    # # #Check that the weight change together
    # import torch.optim as optim
    # import copy
    # optimizer    = optim.Adam(cnet.parameters(), lr=0.01)

    # n_batches =  1
    # batch_sz  =  5

    # input = torch.randn(batch_sz, 1, 28, 28)
    # input  = Variable(input)

    # W_before = cnet.conv_capsules.route_weights.data.numpy()

    # flt_sz = cnet.conv_capsules.flt_sz
    # inds   = cnet.conv_capsules.inds
    
    # W_before_0 = W_before[:, :, inds[0]:inds[0]+flt_sz, inds[0]:inds[0]+flt_sz, :, :]
    # print W_before_0[0,0,0,0,0,0]
    
    # for ind in cnet.conv_capsules.inds[1:]:
    #     print np.all(W_before[:, :, ind:ind+flt_sz, ind:ind+flt_sz, :, :] == W_before_0)


    # for cnt in range(n_batches):
    #      output = cnet(input)
    #      optimizer.zero_grad()
    #      torch.sum(output).backward()
    #      optimizer.step()
        
    # W_before = cnet.conv_capsules.route_weights.data.numpy()

    # flt_sz = cnet.conv_capsules.flt_sz
    # inds   = cnet.conv_capsules.inds
    
    # W_before_0 = W_before[:, :, inds[0]:inds[0]+flt_sz, inds[0]:inds[0]+flt_sz, :, :]

    # for ind in cnet.conv_capsules.inds[1:]:
    #     print np.all(W_before[:, :, ind:ind+flt_sz, ind:ind+flt_sz, :, :] == W_before_0)

    # cnet.post_process()

    # W_before = cnet.conv_capsules.route_weights.data.numpy()

    # flt_sz = cnet.conv_capsules.flt_sz
    # inds   = cnet.conv_capsules.inds
    
    # W_before_0 = W_before[:, :, inds[0]:inds[0]+flt_sz, inds[0]:inds[0]+flt_sz, :, :]
    # print W_before_0[0,0,0,0,0,0]
    # for ind in cnet.conv_capsules.inds[1:]:
    #     print np.all(W_before[:, :, ind:ind+flt_sz, ind:ind+flt_sz, :, :] == W_before_0)




    
