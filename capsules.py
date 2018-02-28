import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class primaryCapsule(nn.Conv2d):

    def __init__(self, in_channels, n_channels, dim_vector, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        
        out_channels = n_channels * dim_vector
        super(primaryCapsule, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.dim_vector = dim_vector
        self.n_channels = n_channels
        
    def forward(self, input):
        output = super(primaryCapsule, self).forward(input)
        output = output.view([output.shape[0], self.n_channels, self.dim_vector, output.shape[-2], output.shape[-1]])
        output = self.squash(output)
        return output

    @staticmethod
    def squash(input, dim=2):
        s_square_norm = torch.sum((input * input), dim=dim, keepdim=True)
        output  = s_square_norm/(1. + s_square_norm) * input/torch.sqrt(s_square_norm)
        return output
        


class digitCapsule(nn.Module):

    def __init__(self, n_inp_capsules, dim_inp_capsules, n_out_capsules, dim_out_capsules, num_routing=1):
        super(digitCapsule,self).__init__()
        self.n_inp_capsules   = n_inp_capsules
        self.dim_inp_capsules = dim_inp_capsules
        self.n_out_capsules   = n_out_capsules
        self.dim_out_capsules = dim_out_capsules
        self.num_routing      = num_routing
        
        self.weight = nn.Parameter(torch.Tensor(dim_inp_capsules, dim_out_capsules, n_inp_capsules, n_out_capsules))
        self.weight.data.uniform_(-0.1,0.1)

                                   

    def forward(self, input):

        b     = torch.zeros(input.shape[0], 1, self.n_inp_capsules, self. n_out_capsules)
        u_hat = torch.Tensor(input.shape[0], self.dim_out_capsules, self.n_inp_capsules, self.n_out_capsules)

        if torch.cuda.is_available():
            b, u_hat = b.cuda(), u_hat.cuda()

        b, u_hat = Variable(b), Variable(u_hat)
            
        for i in range(self.n_inp_capsules):
            for j in range(self.n_out_capsules):
                u_hat[:,:,i,j] = torch.mm(input[:, :, i], self.weight[:,:, i,j])


        c = F.softmax(b, dim=2)
        
        for cnt in range(self.num_routing):
            c     = F.softmax(b.clone(), dim=2)
            v_out = self.squash(torch.sum(u_hat*c, dim=2), dim=1)
            v     = v_out.view(v_out.shape[0], v_out.shape[1], 1, -1)
            v     = torch.cat([v] * self.n_inp_capsules, dim=2)
            b    +=  torch.sum(v * u_hat, dim=1, keepdim=True)
            
        return v_out

    @staticmethod
    def squash(input, dim=2):
        s_square_norm = torch.sum((input * input), dim=dim, keepdim=True)
        output  = s_square_norm/(1. + s_square_norm) * input/torch.sqrt(s_square_norm)
        return output


if __name__ == "__main__":

    class capsNet(nn.Module):

        def __init__(self):
            super(capsNet, self).__init__()
            self.conv1 = nn.Conv2d(1,256,9)
            self.caps1 = primaryCapsule(in_channels=256, n_channels=32, dim_vector=8, kernel_size=9, stride=2)
            self.caps2 = digitCapsule(n_inp_capsules=1152, dim_inp_capsules=8, n_out_capsules=10, dim_out_capsules=16)

        
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.caps1(x)
            x = x.permute(0,2,1,3,4).contiguous()
            x = x.view(x.shape[0], x.shape[1],-1)
            x = self.caps2(x)
            return x



    cnet = capsNet()
    input = torch.randn(5, 1, 28, 28)
    input = Variable(input)
    output = cnet(input)
    print output.shape
