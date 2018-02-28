import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gzip
import cPickle as pickle
import torch.optim as optim
import itertools
import copy
import numpy as np
from capsules import *
import matplotlib.pyplot as plt
from torch.nn.modules.loss import _Loss
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


class recNet(nn.Module):

    def __init__(self):
        super(recNet, self).__init__()

        self.fc1  = nn.Linear(16*10, 512)
        self.fc2  = nn.Linear(512, 1024)
        self.fc3  = nn.Linear(1024, 784)

    def forward(self, x):
        x = x.view(x.shape[0], 160)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = x.view(x.shape[0], 28, 28)
        return x


class MarginLoss(_Loss):

    def __init__(self):
        super(MarginLoss, self).__init__(size_average=True)
        self.m_plus  = 0.9
        self.m_minus = 0.1
        self.lambd   = 0.5
        
    def forward(self, input, target):
        output  = torch.sqrt(torch.sum(input * input, dim=1))
        out_plus  = square(F.relu(self.m_plus - output))
        out_minus = square(F.relu(output - self.m_minus))
        l = out_plus * target  + out_minus * (1 - target) * self.lambd
        return torch.sum(l)
    

def square(x):
    return torch.mul(x,x)


class mnist(Dataset):

    def __init__(self, inp, trg, transform=None):

        self.inp       = inp
        self.trg       = trg
        self.transform = transform
        
    def __len__(self):
        return self.inp.shape[0]

    def __getitem__(self, idx):
        img    = self.inp[idx]
        trg    = self.trg[idx]

                
        if self.transform:
            img = self.transform(img)
            
        return img, trg


    

class ToTensor(object):
    def __call__(self, image):
        return torch.from_numpy(image).float()



if __name__ == "__main__":

    epoch = 100
    batch_size = 20

    datafile = '/home/guglielmo/dataset/mnist.pkl.gz'
    with gzip.open(datafile, 'rb') as f:
        data = pickle.load(f)
        
    images = data[0].reshape(data[0].shape[0],1,28,28)
    labels = data[1]

    n_test_samples  = 10000
    n_valid_samples = 10000
    n_train_samples = 50000
        
    images_test  = images[:n_test_samples]
    images_valid = images[n_test_samples:n_test_samples+n_valid_samples]
    images_train = images[n_test_samples+n_valid_samples:n_test_samples+n_valid_samples+n_train_samples]

    labels_test  = labels[:n_test_samples]
    labels_valid = labels[n_test_samples:n_test_samples+n_valid_samples]
    labels_train = labels[n_test_samples+n_valid_samples:n_test_samples+n_valid_samples+n_train_samples]
    mnistPartTest   = mnist(images_test, labels_test, transform=ToTensor())
    mnistPartValid  = mnist(images_valid, labels_valid, transform=ToTensor())
    mnistPartTrain  = mnist(images_train, labels_train, transform=ToTensor())

    testloader  = DataLoader(mnistPartTest,  batch_size=500, shuffle=False, num_workers=1)
    validloader = DataLoader(mnistPartValid, batch_size=500, shuffle=False, num_workers=1)
    trainloader = DataLoader(mnistPartTrain, batch_size=batch_size, shuffle=True, num_workers=1)

    cnet = capsNet()
    rnet = recNet()
    loss_r    = nn.MSELoss()
    loss_c    = MarginLoss()

    
    if torch.cuda.is_available():
        cnet.cuda()
        rnet.cuda()
    
    optimizer = optim.SGD(itertools.chain(cnet.parameters(), rnet.parameters()), lr=0.00001, momentum=0.9)

    n_epoch = 10
    cnt_epc = 0
    while cnt_epc < n_epoch:
        cnt_epc += 1
        total   = 0
        correct = 0
        cnt_batch = 0
        for input, target in trainloader:
            cnt_batch += 1
            print cnt_batch
            if torch.cuda.is_available():
                input = input.cuda()
        
            input_c  = Variable(input)
            output_c = cnet(input_c)
            
            mask     = torch.zeros(output_c.shape)
            target_d = torch.zeros((input.shape[0], 10))
            for cnt in range(target.shape[0]):
                mask[cnt, :, target[cnt]] = 1.
                target_d[cnt, target[cnt]] = 1.

            if torch.cuda.is_available():
                target, mask, target_d     = target.cuda(), mask.cuda(), target_d.cuda()
            mask     = Variable(mask, requires_grad=False)
            input_r  = torch.mul(output_c, mask)
            output_r = rnet(input_r)

            target_d = Variable(target_d, requires_grad=False)
            loss_cap = loss_c(output_c, target_d)

            loss_rec = loss_r(output_r, input_c)
            loss_t   = 0.0005 * loss_rec + loss_cap

            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

            _, predicted  = torch.max(torch.sum(output_c * output_c, dim=1), dim=1)
            correct += torch.sum(predicted.data == target)
            total   += input.shape[0]
            
            miss = (1.- float(correct)/float(total)) * 100.
            print("Epoch %d" % cnt_epc)
            print("Missclass (Train): %.2f" % miss)


        correct = 0
        total   = 0
        for input, target in testloader:
            if torch.cuda.is_available():
                input = input.cuda()
            input_c  = Variable(input)
            output_c = cnet(input_c)
            _, predicted  = torch.max(torch.sum(output_c * output_c, dim=1), dim=1)

            if torch.cuda.is_available():
                target = target.cuda()
            correct += torch.sum(predicted.data == target)
            total   += input.shape[0]

        miss = (1.- float(correct)/float(total)) * 100.
        print("Missclass (Test): %.2f" % miss)
        
        mask = torch.zeros(output_c.shape)
        for cnt in range(target.shape[0]):
            mask[cnt, :, target[cnt]] = 1.
        
        if torch.cuda.is_available():
            mask  = mask.cuda()
            
        mask     = Variable(mask, requires_grad=False)    
        output_r = rnet(output_c * mask)
        
        fig , axes = plt.subplots(2,4)
        for cnt_r in range(2):
            if cnt_r == 0:
                dt = input_c.data[:,0,:,:]
            if cnt_r == 1:
                dt = output_r.data
                
            for cnt_c in range(4):
                axes[cnt_r, cnt_c].imshow(dt[cnt_c])

        #plt.show()
        fig.savefig('result' + str(cnt_epc) + '.png')
