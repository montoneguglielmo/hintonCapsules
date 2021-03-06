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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn.modules.loss import _Loss
from torchvision import datasets, transforms
import copy
from torch.optim import lr_scheduler
import json
import time

#from capsulesCedrickchee import *
#from capsulesGramAi import *
#from convCaps import *
from votingNet import *

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

    def __init__(self, size_average=True):
        super(MarginLoss, self).__init__(size_average=size_average)
        self.m_plus  = 0.9
        self.m_minus = 0.1
        self.lambd   = 0.5
        self.size_average = size_average
    
        
    def forward(self, input, target):
        output = torch.sqrt(torch.sum(input**2, dim=2))
        zero   = Variable(torch.zeros(1))
        if torch.cuda.is_available():
            zero = zero.cuda()
        out_plus  = torch.max(self.m_plus - output, zero)**2
        out_minus = torch.max(output - self.m_minus, zero)**2
        loss      = (out_plus * target  + out_minus * (1. - target) * self.lambd).sum(dim=1)
        if self.size_average:
            loss = loss.mean()
        return loss
    
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

class shift(object):

    def __call__(self, image, n_pixel=2):
        image_sz    = image.shape
        image_frame = np.zeros((1, image_sz[1] + n_pixel * 2, image_sz[2] + n_pixel * 2))
        image_frame[0, n_pixel:-n_pixel, n_pixel:-n_pixel] = image        
        shift       = np.random.randint(0, n_pixel+1, 2) * (np.random.randint(0,2, 2) * 2 - 1) 
        start       = n_pixel + shift
        image       = image_frame[:, start[0]:start[0]+image_sz[1], start[1]:start[1]+image_sz[2]]
        return image
        

if __name__ == "__main__":
    
    batch_size = 128
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

    labels_test     = labels[:n_test_samples]
    labels_valid    = labels[n_test_samples:n_test_samples+n_valid_samples]
    labels_train    = labels[n_test_samples+n_valid_samples:n_test_samples+n_valid_samples+n_train_samples]

    #transformations = transforms.Compose([shift(), ToTensor()])
    transformations = ToTensor()
    
    mnistPartTest   = mnist(images_test, labels_test, transform=transformations)
    mnistPartValid  = mnist(images_valid, labels_valid, transform=transformations)
    mnistPartTrain  = mnist(images_train, labels_train, transform=transformations)

    testloader  = DataLoader(mnistPartTest,  batch_size=batch_size, shuffle=False, num_workers=1)
    validloader = DataLoader(mnistPartValid, batch_size=batch_size, shuffle=False, num_workers=1)
    trainloader = DataLoader(mnistPartTrain, batch_size=batch_size, shuffle=True, num_workers=1)

    stdvW   = 1e5
    init_lr = 0.001
    lr_step = 1
    gamma   = 0.95
    n_epoch = 250

    results = {'stdvW': stdvW}
    results['init_lr'] = init_lr
    results['lr_step'] = lr_step
    results['gamma']   = gamma
    results['n_epoch'] = n_epoch
    results['name']    = time.strftime("%H%M%S")

    cnet = NetGram(stdvW)
    
    rnet = recNet()
    loss_r    = nn.MSELoss()
    loss_c    = MarginLoss()

    
    if torch.cuda.is_available():
        cnet.cuda()
        rnet.cuda()

    optimizer    = optim.Adam(itertools.chain(cnet.parameters(), rnet.parameters()), lr=init_lr)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=gamma)

    best_valid    = np.inf
    log_interval  = 100
    cnt_epc  = 0
    while cnt_epc < n_epoch:
        lr_scheduler.step()
        cnt_epc   += 1
        cnt_batch  = 0 
        total      = 0
        correct    = 0
        avrg_loss  = 0
        for input, target in trainloader:
            #W_old = copy.deepcopy(cnet.caps2.weight.data)
            cnt_batch += 1
            #print cnt_batch*batch_size
            if torch.cuda.is_available():
                input = input.cuda()
        
            input_c  = Variable(input)
            output_c = cnet(input_c)
        
            target_d = torch.zeros((input.shape[0], 10))
            for cnt in range(target.shape[0]):
                target_d[cnt, target[cnt]] = 1.

            if torch.cuda.is_available():
                target, target_d = target.cuda(), target_d.cuda()

            target_d   = Variable(target_d, requires_grad=False)
            loss_t     = loss_c(output_c, target_d)
            avrg_loss += loss_t.data[0]
            
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

            cnet.post_process()

            #W_new = copy.deepcopy(cnet.caps2.weight.data)
            #print np.linalg.norm(W_new - W_old)
            _, predicted  = torch.max(torch.sum(output_c**2, dim=2), dim=1)
            
            correct += torch.sum(predicted.data == target)
            total   += input.shape[0]

            if np.mod(cnt_batch, log_interval)==0:
                miss = (1.- float(correct)/float(total)) * 100.
                mean_avrg_loss = avrg_loss/float(log_interval)
                print("Epoch %d, data processed %d" % (cnt_epc, total))
                print("Missclass (Train): %.3f" % miss)
                print("Mean Loss: %.4f" % mean_avrg_loss)
                
        correct = 0
        total   = 0
        for input, target in validloader:
            if torch.cuda.is_available():
                input = input.cuda()
            input_c  = Variable(input)
            output_c = cnet(input_c)
            _, predicted  = torch.max(torch.sum(output_c**2, dim=2), dim=1)

            if torch.cuda.is_available():
                target = target.cuda()
            correct += torch.sum(predicted.data == target)
            total   += input.shape[0]

        miss_v = (1.- float(correct)/float(total)) * 100.
        
                
        correct = 0
        total   = 0
        for input, target in testloader:
            if torch.cuda.is_available():
                input = input.cuda()
            input_c  = Variable(input)
            output_c = cnet(input_c)
            _, predicted  = torch.max(torch.sum(output_c**2, dim=2), dim=1)

            if torch.cuda.is_available():
                target = target.cuda()
            correct += torch.sum(predicted.data == target)
            total   += input.shape[0]

        miss_t = (1.- float(correct)/float(total)) * 100.
        print("Missclass Valid:%.3f ,   Test: %.3f" % (miss_v, miss_t))

        
        if miss_v < best_valid:
            best_valid = miss_v
            results['best_test'] = miss_t
            results['curr_epc']  = cnt_epc
            results['curr_lr']   = lr_scheduler.get_lr()[0]

            with open(results['name'] + '.json', 'w') as f:
               json.dump(results, f, indent=3, sort_keys=True)


        #print("Missclass Best Test: %.3f" % best_test)
        print("Current Learning Rate: %.7f" % lr_scheduler.get_lr()[0])
