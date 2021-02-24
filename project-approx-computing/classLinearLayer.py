import torch
#torch.cuda.current_device()
import argparse
from numba import jit, prange
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from collections import OrderedDict
import sys
import resource
import torch
import torch.nn as nn
import math
import numpy as np
import os
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torchvision import datasets, transforms
from compress4 import compress_top4

#@jit(parallel = True, nopython=True)
#@jit(parallel = True)
def mult_implement( ip, w, res, ip_batch_size , weight_size, ip_neurons, sign_res):
        abs_ip = np.abs(ip)
        abs_w = np.abs(w)
        sign_ip = np.sign(ip)
        sign_w = np.sign(w)

        print("created")
        #print(k, l, z)
        for i in range(ip_batch_size):
           for j in range(weight_size):
              sign_res[i][j] = sign_ip[i] * sign_w[j]
              for y in range(ip_neurons):
                  #res[i][j][y] = mymul(abs_ip[i][y], abs_w[j][y])
                  res[i][j][y] = compress_top4((int)(abs_ip[i][y]), (int)(abs_w[j][y]), (int)(4)) 

        print("done")

        return res*sign_res
#@jit(parallel = True)
def mult_implement_dummy( a, b, res, k , l, z):
        #print(res.shape)
        #p1      = np.ndarray(shape = (int(4), 8 ), dtype = bool)
        #print("crreating empty matrix")
        #print(a.size()[0], b.size()[0])
        #res = torch.zeros(a.size()[0]* b.size()[0], device=torch.device('cuda:0'))
        #prod = np.zeros((z,1), dtype = float)
        print("created")

        for i in prange( k):
           #print("i")
           for j in prange(l):
      
              #res[i][j] = torch.sum(torch.Tensor(list(map(mymul, a[i], b[j], prod ))))
              #print(len(a[i]), len(b[j]))
              res[i][j] = sum((list(map(mymul, a[i], b[j])))) 
              #print(res[i].shape)
              #print(res[i][j].shape)
           #print("i-done")
              #print(i, j)
        #res = torch.reshape(res, (a.size()[0], b.size()[0]))
        #print(res)
        return res

@jit(parallel = True)
def mymul(a, b):
        #prod = a*b
        return (float)(a*b)
class MyLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, weight_sf, size_in, size_out, counter):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weight = torch.Tensor(size_out, size_in)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        self.weight_sf = weight_sf
        self.act_sf = 0
        self.counter = counter

    def forward(self, input):
      print("layer")
      if self.counter>0:
        self.counter = self.counter-1
        k = input.size()[0]
        l = self.weight.size()[0]
        z = input.size()[1]
        wx = np.zeros((k, l), dtype = int)
        x1 = input.cpu().detach().numpy()
        x2 = self.weight.cpu().detach().numpy()
        #x2 = w.cpu().detach().numpy()        
        mmm = mult_implement_dummy(x1, x2, wx, k, l ,z)
        w_times_x= torch.from_numpy(mmm)
        #x = torch.flip(x, [0,1])
        repeat_bias = self.bias.repeat(input.size()[0], 1)
        w2 = w_times_x.add(repeat_bias)
        y = w_times_x.add(repeat_bias)
        return w2  # w times x + b
      else:
        ip_batch_size = input.size()[0]
        weight_size = self.weight.size()[0]
        ip_neurons = input.size()[1]
        wx = np.zeros((ip_batch_size, weight_size, ip_neurons), dtype = int)
        res_sign = np.zeros((ip_batch_size, weight_size, ip_neurons), dtype = int)
        #print(input.size())
        #print(self.weight.size())
        input1 = input/(math.pow(2, -self.act_sf))
        w = self.weight/(math.pow(2,-self.weight_sf))

        input1=input1.int()
        w=w.int()

        #print("---------------check if integers----------------------")
        #print(input1)
        #print(w)
        x1 = input1.cpu().detach().numpy()
        #x2 = self.weight.cpu().detach().numpy()
        x2 = w.cpu().detach().numpy()
        sign_ip = np.sign(x1)
        abs_ip  = np.abs(x1)
        sign_w  = np.sign(x2)
        abs_w   = np.abs(x2)  
        mmm = mult_implement(x1, x2, wx, ip_batch_size, weight_size ,ip_neurons, res_sign)
        mmm = np.sum(mmm, axis = 2)
        w_times_x= torch.from_numpy(mmm)
        w_times_x = w_times_x * (math.pow(2, -self.act_sf)) * (math.pow(2,-self.weight_sf))
        #x = torch.flip(x, [0,1])
        repeat_bias = self.bias.repeat(input.size()[0], 1)
        #w_times_x = torch.matmul(input, torch.transpose(self.weight, 0,1))
        #w_times_x = input.matmul(self.weight.t())
        w2 = w_times_x.add(repeat_bias)
        #w_times_x = input @ self.weight.t() + self.bias
        #y = input @ self.weight.t() + self.bias#
        y = w_times_x.add(repeat_bias)
        #assert torch.eq(w_times_x, y), 'not equai'
        return w2  # w times x + b

    def assign_act_sf(self, act_sf):
        print("--------inside--------")
        self.act_sf = act_sf
