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

@jit(parallel = True)
def mymul(a, b):
        #prod = a*b
        return (float)(a*b)
@jit(nopython = True)
def my_calc_dummy(w2, ready_img, number_of_kernels, batch_size, out_mat, input_image_positions, weights_size):
    for kernel_no in range((int)(number_of_kernels)):
         #print(kernel_no)
         for img_no in range(batch_size):
                 for pos in range(input_image_positions):
                    for wpos in range(weights_size):
                      #print("debug")
                      #print(mymul(ready_img[img_no][pos*weights_size+wpos], w2[kernel_no][pos]))
                      out_mat[kernel_no][img_no][pos*weights_size+wpos] = mymul((int)(ready_img[img_no][pos*weights_size+wpos]), (int)(w2[kernel_no][wpos]))

    #print("returned from 1")
    #print(out_mat.shape)
    #print(out_mat)
    return out_mat
              

#@jit(parallel = True)
#@jit
def my_convolvedd_dummy(o, batch_size, o_size, weights_size, out_channels, bias, w_sf, act_sf):
    out_mat2=np.reshape(o, (out_channels, batch_size,  o_size*o_size, weights_size))
    out_mat2=np.sum(out_mat2, axis=3)
    out_mat2=out_mat2*(w_sf * act_sf)
    for b in range(out_channels):
         out_mat2[b]=out_mat2[b] + bias[b]
    out_mat2=np.transpose(out_mat2, (1,0,2))
    out_mat2=np.reshape(out_mat2, (batch_size, out_channels, o_size, o_size))
    return out_mat2

#@jit(nopython = True)
def my_calc(w2, ready_img, number_of_kernels, batch_size, out_mat, input_image_positions, weights_size):
    for kernel_no in range((int)(number_of_kernels)):
         print(kernel_no)
         for img_no in range(batch_size):
                 for pos in range(input_image_positions):
                    for wpos in range(weights_size):
                      #print("debug")
                      #print(mymul(ready_img[img_no][pos*weights_size+wpos], w2[kernel_no][pos]))
                      out_mat[kernel_no][img_no][pos*weights_size+wpos] = compress_top4((int)(ready_img[img_no][pos*weights_size+wpos]), (int)(w2[kernel_no][wpos]), 4)

    #print("returned from 1")
    #print(out_mat.shape)
    #print(out_mat)
    return out_mat
              

#@jit(parallel = True)
#@jit
def my_convolvedd(o, batch_size, o_size, weights_size, out_channels, bias, w_sf, act_sf):
    out_mat2=np.reshape(o, (out_channels, batch_size,  o_size*o_size, weights_size))
    out_mat2=np.sum(out_mat2, axis=3)
    out_mat2=out_mat2*(w_sf * act_sf)
    for b in range(out_channels):
         out_mat2[b]=out_mat2[b] + bias[b]
    out_mat2=np.transpose(out_mat2, (1,0,2))
    out_mat2=np.reshape(out_mat2, (batch_size, out_channels, o_size, o_size))
    return out_mat2

class my_Conv2d(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, in_channels, out_channels, weight_sf, kernel, padding,counter ):
        super().__init__()
        self.padding=padding
        self.in_channels=in_channels
        self.out_channels=out_channels
        weight = torch.Tensor(out_channels, in_channels, kernel, kernel)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(out_channels)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.weight_sf=weight_sf;
        self.act_sf = 0;
        self.counter= counter
        self.kernel_size = kernel

    def forward(self, x):
      if self.counter > 0:
        self.counter = self.counter - 1
        delta_weight = math.pow(2.0, -self.weight_sf)
        delta_act = math.pow(2.0, -self.act_sf)
        w = self.weight
        w = w/delta_weight
        #print("------------------is it integerss---------------------------_")
        #print(delta_act)
        x = x/delta_act
        #print(x)
        number_of_kernels = w.size()[0]
        number_of_channels= w.size()[1]
        #weights_size = number_of_channels * 9
        print("layer")
        #print("-----------------------input is ---------------------------------")
        #print(x)
        batch_size = x.size()[0]
        unfold = torch.nn.Unfold(kernel_size=(self.kernel_size,self.kernel_size), padding=self.padding)        
        ready_img = unfold(x).transpose(1,2)
 
        r0 = ready_img.size()[0]
        number_of_patches = ready_img.size()[1]
        patch_size=ready_img.size()[2]
 
        ready_img = torch.reshape(ready_img, (r0, number_of_patches*patch_size))
        ready_img2 = ready_img.cpu().detach().numpy()

        w2 = w.view(number_of_kernels, patch_size)
        b2 = self.bias.cpu().detach().numpy()
        o_size= x.size()[2] - (2 * (self.kernel_size//2)) + (2 * self.padding)
        out_placeholder = np.zeros((self.out_channels, batch_size, number_of_patches * patch_size), dtype = np.float32)
        #w2=w2.repeat(1, input_image_size)
        assert (patch_size == number_of_channels * self.kernel_size * self.kernel_size), "patch size is wrong"
        assert (o_size*o_size == number_of_patches), "number of patches is wrong"
        w3=w2.cpu().detach().numpy()
 
        output = my_calc_dummy(w3, ready_img2, number_of_kernels, batch_size, out_placeholder, number_of_patches, patch_size)
        output = my_convolvedd_dummy(output, batch_size, o_size, patch_size, self.out_channels, b2, delta_weight, delta_act)
        output = torch.from_numpy(output)    
        output=output.float()
        return output
      else:
        #print("layer")
        delta_weight = math.pow(2.0, -self.weight_sf)
        delta_act = math.pow(2.0, -self.act_sf)
        w = self.weight
        w = w/math.pow(2.0, -self.weight_sf)
        #print("------------------is it integerss---------------------------_")
        #print(delta_act)
        x = x/math.pow(2.0, -self.act_sf)
        #print(x)
        number_of_kernels = w.size()[0]
        number_of_channels= w.size()[1]
        #weights_size = number_of_channels * 9
        print("layer")
        #print("-----------------------input is ---------------------------------")
        #print(x)
        batch_size = x.size()[0]
        unfold = torch.nn.Unfold(kernel_size=(self.kernel_size,self.kernel_size), padding=self.padding)        
        ready_img = unfold(x).transpose(1,2)
 
        r0 = ready_img.size()[0]
        number_of_patches = ready_img.size()[1]
        patch_size=ready_img.size()[2]
 
        ready_img = torch.reshape(ready_img, (r0, number_of_patches*patch_size))
        ready_img2 = ready_img.cpu().detach().numpy()

        w2 = w.view(number_of_kernels, patch_size)
        b2 = self.bias.cpu().detach().numpy()
        o_size= x.size()[2] - (2 * (self.kernel_size//2)) + (2 * self.padding)
        out_placeholder = np.zeros((self.out_channels, batch_size, number_of_patches * patch_size), dtype = np.float32)
        #w2=w2.repeat(1, input_image_size)
        assert (patch_size == number_of_channels * self.kernel_size * self.kernel_size), "patch size is wrong"
        assert (o_size*o_size == number_of_patches), "number of patches is wrong"
        w3=w2.cpu().detach().numpy()
 
        output = my_calc(w3, ready_img2, number_of_kernels, batch_size, out_placeholder, number_of_patches, patch_size)
        output = my_convolvedd(output, batch_size, o_size, patch_size, self.out_channels, b2, delta_weight, delta_act)
        output = torch.from_numpy(output)    
        output=output.float()
        return output

    def assign_act_sf(self,sf):
        #print("here iam")
        #print(sf)
        self.act_sf = sf
        #print(self.act_sf)
