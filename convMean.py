import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn import functional as F

# class convMean(nn.Conv2d):
    
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        
#         super(convMean, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         avrg = 1. / n
#         self.weight.data[:] = avrg
#         self.weight.requires_grad = False
#         if self.bias is not None:
#             self.bias.data[:] = 0.
#             self.bias.requires_grad = False

class convMean(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):

        super(convMean, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = _pair(kernel_size)
        self.stride       = _pair(stride)
        self.padding      = _pair(padding)
        self.dilation     = _pair(dilation)
        self.groups       = groups
        
        self.weight = Variable(torch.Tensor(self.out_channels, self.in_channels // self.groups, *self.kernel_size))
        self.bias   = None
        self.reset_parameters()
        
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        avrg = 1. / n
        self.weight[:] = avrg
        

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



if __name__ == "__main__":


  inp   = Variable(torch.rand(5, 20, 16, 32, 6, 6))
  inp_v = inp.view(5*16*20, 32, 6, 6)

  
  mean = convMean(32, 1, 2, stride=2)


  up  = nn.Upsample(scale_factor=2)
  out = mean(inp_v)
  out = up(out)


  out = out.view(5,20,16,6,6)
  print inp.shape
  print out.shape

  out = torch.sum(out[:,:,:,None,:,:] * inp, dim=3, keepdim=True)
  print out.shape
