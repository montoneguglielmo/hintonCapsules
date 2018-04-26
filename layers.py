import torch
import torch.nn.functional as F
from   torch import nn
import numpy as np
from   torch.autograd import Variable
from   convMean import *
import time


class simpleConv(nn.Conv2d):

    def __init__(self,**kwargs):
        super(simpleConv, self).__init__(**kwargs)

        
    def printInfo(self):
        n_params = (self.kernel_size[0]**2)*self.in_channels*self.out_channels
        return n_params

    def getstatejson(self):
        state = {
            'n_inp_filters': self.in_channels,
            'n_out_filter' : self.out_channels,
            'flt_sz'       : self.kernel_size,
            'stride'       : self.stride,
            'num_params'   : self.printInfo()
        }
        return state

    def post_process(self):
        pass


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

    def getstatejson(self):
        state = {
            'dim_out_caps' : self.dim_out_capsules,
            'n_inp_filters': self.n_inp_filters,
            'n_capsules'   : self.n_capsules,            
            'flt_sz'       : self.kernel_size,
            'stride'       : self.stride,
            'num_params'   : self.printInfo()
        }
        return state


    def post_process(self):
        pass

    
class downSampleCapsuleLayer(nn.Module):

    def __init__(self,n_inp_caps, dim_inp, dim_inp_caps, dim_out_caps, flt_sz, num_iterations):

        super(downSampleCapsuleLayer, self).__init__()
        self.route_weights = nn.Parameter(torch.randn(n_inp_caps, dim_inp, dim_inp, dim_inp_caps, dim_out_caps))

        self.n_inp_caps     = n_inp_caps
        self.dim_inp        = dim_inp
        self.dim_inp_caps   = dim_inp_caps
        self.dim_out_caps   = dim_out_caps
        self.flt_sz         = flt_sz
        self.num_iterations = num_iterations
        self.inds  = range(0,dim_inp, flt_sz)
        self.convMean = convMean(1, 1, flt_sz, stride=flt_sz)
        self.upSample = nn.Upsample(scale_factor=flt_sz)


    def forward(self, x):
        priors = torch.matmul(x[:, :, :, :, None, :], self.route_weights[None, :, :, :, :, :])
        priors = priors.squeeze()
        priors = priors.permute(0,4,1,2,3).contiguous()
        logits = Variable(torch.zeros(priors.shape[0], 1, priors.shape[2], priors.shape[3], priors.shape[4]))

        if torch.cuda.is_available():
            logits = logits.cuda()

        for i in range(self.num_iterations):
            probs           = F.softmax(logits, dim=1)
            prior_weighted  = probs*priors
            prior_conv      = prior_weighted.view(-1, 1, priors.shape[-2], priors.shape[-1])
            prior_conv      = self.convMean(prior_conv)
            prior_conv_mean = self.upSample(prior_conv)
            prior_conv_mean = prior_conv_mean.view(prior_weighted.shape[0], prior_weighted.shape[1], prior_weighted.shape[2], prior_weighted.shape[3], prior_weighted.shape[4])
            prior_conv_mean = self.squash(prior_conv_mean, 1)
            delta_logits    = torch.sum(prior_conv_mean * priors, dim=1, keepdim=True)
            logits          = logits + delta_logits

        output = prior_conv.view(prior_weighted.shape[0], prior_weighted.shape[1], prior_weighted.shape[2], prior_conv.shape[-2], prior_conv.shape[-1])
        output = output.permute(0,2,3,4,1).contiguous()
        return self.squash(output) 


    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale        = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


    def post_process(self):
        route_mean  = torch.zeros(self.route_weights[:, :self.flt_sz, :self.flt_sz, :, :].shape)
        if torch.cuda.is_available():
            route_mean = route_mean.cuda()
            
        n_filter    = 0
        for indy in self.inds:
            for indx in self.inds:
               route_mean += self.route_weights.data[:, indx:indx+self.flt_sz, indy:indy+self.flt_sz, :, :]
               n_filter += 1

        route_mean /= float(n_filter) 

        for indy in self.inds:
            for indx in self.inds:
                self.route_weights.data[:, indx:indx+self.flt_sz, indy:indy+self.flt_sz, :, :] =  route_mean        

    def printInfo(self):
        n_params = self.n_inp_caps * (self.flt_sz**2) * self.dim_inp_caps * self.dim_out_caps
        return n_params


    def getstatejson(self):
        state = {
            'n_inp_caps'  : self.n_inp_caps,
            'dim_inp'     : self.dim_inp,
            'dim_inp_caps': self.dim_inp_caps,
            'dim_out_caps': self.dim_out_caps,
            'flt_sz'      : self.flt_sz,
            'num_iter'    : self.num_iterations,
            'num_params'  : self.printInfo()            
        }
        return state

    
class convCapsuleLayer(nn.Module):

    def __init__(self,n_flt_inp, n_flt_out, inp_sz, dim_inp_caps, dim_out_caps, flt_sz, num_iterations):

        super(convCapsuleLayer, self).__init__()
        self.route_weights = nn.Parameter(torch.randn(n_flt_out, n_flt_inp, flt_sz*flt_sz, dim_inp_caps, dim_out_caps))

        self.n_flt_inp      = n_flt_inp
        self.n_flt_out      = n_flt_out
        self.inp_sz         = inp_sz
        self.dim_inp_caps   = dim_inp_caps
        self.dim_out_caps   = dim_out_caps
        self.flt_sz         = flt_sz
        self.num_iterations = num_iterations
 
        self.dx = self.inp_sz / self.flt_sz
        self.dy = self.inp_sz / self.flt_sz

        
    def forward(self, x):
        x = x.permute(0,1,4,2,3).contiguous()
        x = x.view(x.shape[0], x.shape[1], x.shape[2], self.dx, self.flt_sz, self.dy, self.flt_sz)
        x = x.permute(0,1,2,3,5,4,6).contiguous()
        x = x.view(x.shape[0], x.shape[1], x.shape[2], self.dx*self.dy, self.flt_sz * self.flt_sz)
        x = x.permute(0,1,3,4,2).contiguous()
        priors = torch.matmul(x[:, None, :, :, :, None, :], self.route_weights[None, :, :,None, :, :, :])
        priors = priors.squeeze()

        logits = Variable(torch.zeros(priors.shape[0], priors.shape[1], priors.shape[2], priors.shape[3], priors.shape[4], 1))

        for i in range(self.num_iterations):
            probs                = F.softmax(logits, dim=1)
            prior_weighted       = probs*priors
            prior_weighted_mean  = prior_weighted.mean(dim=2, keepdim=True).mean(dim=4, keepdim=True)
            prior_weighted_mean  = self.squash(prior_weighted_mean, dim=1)
            delta_logits         = (prior_weighted_mean * prior_weighted).sum(dim=-1, keepdim=True)
            logits               = logits + delta_logits 

        prior_weighted_mean = prior_weighted_mean.squeeze()
        pr_shape            = prior_weighted_mean.shape 
        prior_weighted_mean = prior_weighted_mean.view(pr_shape[0], pr_shape[1], self.dx, self.dy, pr_shape[-1])
        prior_weighted_mean = prior_weighted_mean.contiguous()

        return prior_weighted_mean
         
            
    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale        = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


    def printInfo(self):
        n_params = np.prod(self.route_weights.shape)
        return n_params

    
    def getstatejson(self):
        state={
            "n_flt_inp": self.n_flt_inp,
            "n_flt_out": self.n_flt_out,
            "dim_inp_caps": self.dim_inp_caps,
            "dim_out_caps": self.dim_out_caps,
            "flt_sz"      : self.flt_sz,
            "n_params": self.printInfo() 
        }
        return state

# class convCapsuleLayer(nn.Module):

#     def __init__(self,n_flt_inp, n_flt_out, inp_sz, dim_inp_caps, dim_out_caps, flt_sz, num_iterations):

#         super(convCapsuleLayer, self).__init__()
#         self.route_weights = nn.Parameter(torch.randn(n_flt_out, n_flt_inp, inp_sz, inp_sz, dim_inp_caps, dim_out_caps))
        
#         self.n_flt_inp      = n_flt_inp
#         self.n_flt_out      = n_flt_out
#         self.inp_sz         = inp_sz
#         self.dim_inp_caps   = dim_inp_caps
#         self.dim_out_caps   = dim_out_caps
#         self.flt_sz         = flt_sz
#         self.num_iterations = num_iterations
#         self.inds           = range(0,inp_sz, flt_sz)

#         self.convMean = convMean(n_flt_inp, 1, flt_sz, stride=flt_sz)
#         self.upSample = nn.Upsample(scale_factor=flt_sz)


#     def forward(self, x):
#         priors = torch.matmul(x[:,None, :, :, :, None, :], self.route_weights[None, :, :, :, :, :, :])
#         priors = priors.squeeze()
#         priors = priors.permute(0,1,5,2,3,4).contiguous()
#         logits = Variable(torch.zeros(priors.shape[0], priors.shape[1], 1, priors.shape[3], priors.shape[4], priors.shape[5]))

#         if torch.cuda.is_available():
#             logits = logits.cuda()

#         for i in range(self.num_iterations):
#             probs           = F.softmax(logits, dim=1)
#             prior_weighted  = probs*priors
#             prior_conv      = prior_weighted.view(-1, priors.shape[-3], priors.shape[-2], priors.shape[-1])
#             prior_conv      = self.convMean(prior_conv)
#             prior_conv_mean = self.upSample(prior_conv)
#             prior_conv_mean = prior_conv_mean.view(prior_weighted.shape[0], prior_weighted.shape[1], prior_weighted.shape[2], 1, prior_weighted.shape[4], probs.shape[5])
#             prior_conv_mean = self.squash(prior_conv_mean, 2)
#             delta_logits    = torch.sum(prior_conv_mean * priors, dim=2, keepdim=True)
#             logits          = logits + delta_logits

#         prior_conv = prior_conv.squeeze()
#         output = prior_conv.view(prior_conv_mean.shape[0], prior_conv_mean.shape[1], prior_conv_mean.shape[2], prior_conv.shape[-2], prior_conv.shape[-1])
#         output = output.permute(0,1,3,4,2).contiguous()
#         return self.squash(output) 


#     def squash(self, tensor, dim=-1):
#         squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
#         scale        = squared_norm / (1 + squared_norm)
#         return scale * tensor / torch.sqrt(squared_norm)


#     def post_process(self):
#         route_mean  = torch.zeros(self.route_weights[:, :, :self.flt_sz, :self.flt_sz, :, :].shape)
#         if torch.cuda.is_available():
#             route_mean = route_mean.cuda()
            
#         n_filter    = 0
#         for indy in self.inds:
#             for indx in self.inds:
#                route_mean += self.route_weights.data[:, :, indx:indx+self.flt_sz, indy:indy+self.flt_sz, :, :]
#                n_filter += 1

#         route_mean /= float(n_filter) 

#         for indy in self.inds:
#             for indx in self.inds:
#                 self.route_weights.data[:, :, indx:indx+self.flt_sz, indy:indy+self.flt_sz, :, :] =  route_mean        

#     def printInfo(self):
#         n_params = self.n_flt_out * self.n_flt_inp * self.dim_inp_caps * self.dim_out_caps * self.flt_sz**2
#         return n_params

    
#     def getstatejson(self):
#         state={
#             "n_flt_inp": self.n_flt_inp,
#             "n_flt_out": self.n_flt_out,
#             "dim_inp_caps": self.dim_inp_caps,
#             "dim_out_caps": self.dim_out_caps,
#             "flt_sz"      : self.flt_sz,
#             "n_params": self.printInfo() 
#         }
#         return state


    
class fcCapsuleLayer(nn.Module):
    
    def __init__(self, n_out_caps, n_inp_caps, dim_inp_capsules, dim_out_capsules, num_iterations=3):
        super(fcCapsuleLayer, self).__init__()
        self.num_iterations = num_iterations
        self.route_weights = nn.Parameter(torch.randn(n_out_caps, n_inp_caps, dim_inp_capsules, dim_out_capsules))

        self.n_out_caps = n_out_caps
        self.n_inp_caps = n_inp_caps
        self.dim_inp_capsules = dim_inp_capsules
        self.dim_out_capsules = dim_out_capsules
        
        
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
            
        outputs = outputs.squeeze()
        return outputs

    def printInfo(self):
        n_params = self.n_out_caps * self.n_inp_caps * self.dim_inp_capsules * self.dim_out_capsules
        return n_params

    
    def getstatejson(self):
        state={
            "n_out_caps": self.n_out_caps,
            "n_inp_caps": self.n_inp_caps,
            "dim_inp_caps": self.dim_inp_capsules,
            "dim_out_caps": self.dim_out_capsules,
            "n_params": self.printInfo() 
        }
        return state


    def post_process(self):
        pass
    
if __name__ == "__main__":

    input = torch.randn(5, 1, 28, 28)
    input = Variable(input)

    start = startCapsuleLayer(dim_out_capsules=8, n_inp_filters=1, n_capsules=32, kernel_size=9, stride=1)
    print "start capsule shape:" , start(input).shape
    print "start capsule params:", start.printInfo() 
    output = start(input)
    
    down  = downSampleCapsuleLayer(n_inp_caps=32, dim_inp=20, dim_inp_caps=8, dim_out_caps=16, flt_sz=2, num_iterations=3)
    print "down capsules shape", down(start(input)).shape
    print "down capsules params:", down.printInfo() 
    output = down(output)
    

    conv  = convCapsuleLayer(n_flt_inp=32, n_flt_out=32, inp_sz=10, dim_inp_caps=16, dim_out_caps=20, flt_sz=2, num_iterations=3)
    print "conv capsules shape",  conv(down(start(input))).shape
    print "conv capsules params", conv.printInfo()
    output = conv(output)

    x = output
    x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
    ffw  = fcCapsuleLayer(n_out_caps=10, n_inp_caps=800, dim_inp_capsules=20, dim_out_capsules=25)
    print "ffw capsules shape", ffw(x).shape
    print "ffw capsules params", ffw.printInfo()
    
