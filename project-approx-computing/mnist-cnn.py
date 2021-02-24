# based on https://github.com/aaron-xichen/pytorch-playground
import torch
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
sys.path.append('.')
from classLinearLayer import MyLinearLayer
from classConv2d import my_Conv2d
import torch.nn.functional as F

def eval_model(model, ds, n_sample=None, ngpu=0, is_imagenet=False):
    import tqdm
    import torch
    from torch import nn
    from torch.autograd import Variable

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        def forward(self, input):
            input.data.div_(255.)
            input.data[:, 0, :, :].sub_(self.mean[0]).div_(self.std[0])
            input.data[:, 1, :, :].sub_(self.mean[1]).div_(self.std[1])
            input.data[:, 2, :, :].sub_(self.mean[2]).div_(self.std[2])
            return self.model(input)

    correct1, correct5 = 0, 0
    n_passed = 0
    if is_imagenet:
        model = ModelWrapper(model)
    model = model.eval()

    c = 1
    dataiter = iter(ds)
    images, labels = dataiter.next()
    n_sample = len(ds) if n_sample is None else n_sample
    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        c = c + 1
        n_passed += len(data)
        data =  Variable(torch.FloatTensor(data))
        indx_target = torch.LongTensor(target)
        output = model(data)

        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]


        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += idx_pred[:, :1].cpu().eq(idx_gt1).sum()
        correct5 += idx_pred[:, :5].cpu().eq(idx_gt5).sum()
        if idx >= n_sample - 1:
            break

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5

def eval_model1(model, ds, n_sample=None, ngpu=0, is_imagenet=False):
    import tqdm
    import torch
    from torch import nn
    from torch.autograd import Variable

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        def forward(self, input):
            input.data.div_(255.)
            input.data[:, 0, :, :].sub_(self.mean[0]).div_(self.std[0])
            input.data[:, 1, :, :].sub_(self.mean[1]).div_(self.std[1])
            input.data[:, 2, :, :].sub_(self.mean[2]).div_(self.std[2])
            return self.model(input)

    correct1, correct5 = 0, 0
    n_passed = 0
    if is_imagenet:
        model = ModelWrapper(model)
    model = model.eval()

    c = 1
    dataiter = iter(ds)
    images, labels = dataiter.next()
    n_sample = len(ds) if n_sample is None else n_sample
    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        c = c + 1
        n_passed += len(data)
        data =  Variable(torch.FloatTensor(data))
        indx_target = torch.LongTensor(target)
        output = model(data)

        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]


        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += idx_pred[:, :1].cpu().eq(idx_gt1).sum()
        correct5 += idx_pred[:, :5].cpu().eq(idx_gt5).sum()
        if idx >= (n_sample/50) - 1:
            break

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5

def get(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
    kwargs.pop('input_size', None)
    num_workers = kwargs.setdefault('num_workers', 1)
    print("Building MNIST data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=False, download=True,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

class LinearQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LinearQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            sf_new = self.bits - 1 - compute_integral_part(input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            print("//////////////////////////////////////////////////////////////")
            output = linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

def duplicate_model_with_quant(model, bits, overflow_rate=0.0, counter=10, type='linear'):
    """assume that original model has at least a nn.Sequential"""
    assert type in ['linear', 'minmax', 'log', 'tanh']
    if isinstance(model, nn.Sequential):
        l = OrderedDict()
        for k, v in model._modules.items():

            if isinstance(v, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.AvgPool2d, )):
                l[k] = v
                if type == 'linear':
                    quant_layer = LinearQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                elif type == 'log':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=log_minmax_quantize)
                elif type == 'minmax':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=min_max_quantize)
                else:
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=tanh_quantize)
                l['{}_{}_quant'.format(k, type)] = quant_layer

            elif isinstance(v, (MyLinearLayer, my_Conv2d )):
                quant_layer = LinearQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                if type == 'linear':
                    quant_layer = LinearQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                elif type == 'log':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=log_minmax_quantize)
                elif type == 'minmax':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=min_max_quantize)
                else:
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=tanh_quantize)

                l['{}_{}_quant'.format(k, type)] = quant_layer
                l[k] = v

            else:
                l[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
        m = nn.Sequential(l)
        return m
    else:
        for k, v in model._modules.items():
            model._modules[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
        return model


def compute_integral_part(input, overflow_rate):
    import math
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = float(v.data.cpu())
    sf = math.ceil(math.log2(v+1e-12))
    return sf

def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)
    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
   
    return clipped_value


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        layers = OrderedDict()
        layers['conv1']= nn.Conv2d(1, 10, kernel_size=5)
        layers['M1']   = nn.MaxPool2d(2)
        layers['ReLU1'] = nn.ReLU()
        layers['conv2']= nn.Conv2d(10, 20, kernel_size=5)
        layers['Dropout1'] = nn.Dropout2d()
        layers['M2'] = nn.MaxPool2d(2)
        layers['ReLU2'] = nn.ReLU()
        self.features = nn.Sequential(layers)

        layers1 = OrderedDict()
        layers1['fc1'] = nn.Linear(320, 50)
        layers1['ReLU1'] = nn.ReLU()
        layers1['Dropout1'] = nn.Dropout()
        layers1['fc2'] = nn.Linear(50, 10)
        self.classifier = nn.Sequential(layers1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 320)
        x = self.classifier(x)
        return F.log_softmax(x)

class myNet(nn.Module):
    def __init__(self, weights_sf, counter):
        super(myNet, self).__init__()
        self.weights_sf = weights_sf
        layers = OrderedDict()
        layers['conv1']=my_Conv2d(1, 10,  weights_sf.pop(0), kernel=5, padding=0, counter=counter)
        layers['M1']   = nn.MaxPool2d(2)
        layers['ReLU1'] = nn.ReLU()
        layers['conv2']=my_Conv2d(10, 20, weights_sf.pop(0), kernel=5, padding=0, counter=counter)
        layers['Dropout1'] = nn.Dropout2d()
        layers['M2'] = nn.MaxPool2d(2)
        layers['ReLU2'] = nn.ReLU()
        self.features = nn.Sequential(layers)

        layers1 = OrderedDict()
        layers1['fc1'] = MyLinearLayer(weights_sf.pop(0), 320, 50, counter=counter)
        layers1['ReLU1'] = nn.ReLU()
        layers1['Dropout1'] = nn.Dropout()
        layers1['fc2'] = MyLinearLayer(weights_sf.pop(0), 50, 10, counter=counter)
        self.classifier = nn.Sequential(layers1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 320)
        x = self.classifier(x)
        return F.log_softmax(x)

# initialise a standard network
# load weights from the pth file
# quantize weights, store scaling factor for use in forward pass

def main():

    model_raw = Net()
    prev_state_dict = torch.load("/home/uma/Desktop/project-approx-computing/results/model.pth")
    param_bits = 4
    bn_bits = 4
    fwd_bits =4
    overflow_rate = 0.0
    n_sample = 100
    input_size = 224
    quant_method = 'linear'
    model_root = '~/.torch/models/'
    data_root = '/home/uma/public_ataset/pytorch/'
    typer = 'mnist'
    batch_size = 100
    torch.manual_seed(1)
    ngpu = 1
    counter = 100

    if param_bits < 32:
        state_dict = model_raw.state_dict()
        sd = state_dict
        state_dict_quant = OrderedDict()
        sf_dict = OrderedDict()
        weights_sf = []
        for k, v in state_dict.items():
            if 'running' in k:
                if bn_bits >=32:
                    print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits = bn_bits
            else:
                bits = param_bits

            if quant_method == 'linear':
                sf = bits - 1. - compute_integral_part(v, overflow_rate=overflow_rate)
                v_quant  = linear_quantize(v, sf, bits=bits)
                if "weight" in k:
                    weights_sf += [sf] 
                    print("yes")
            
            state_dict_quant[k] = v_quant
            print(k, bits)
        model_raw.load_state_dict(state_dict_quant)

# introducing quantization layers into the network for activations
    if fwd_bits < 32:
        model_raw = duplicate_model_with_quant(model_raw, bits=fwd_bits, overflow_rate=overflow_rate, counter=n_sample, type=quant_method)
# run forward pass a few times to collect activation sf parameters
        print(model_raw)
        val_ds_tmp = get(10, data_root=data_root, train=False, input_size=input_size)
        eval_model(model_raw, val_ds_tmp, ngpu=ngpu, n_sample=100)
        print(model_raw)


        my_model = myNet(weights_sf, counter)
        my_model.load_state_dict(model_raw.state_dict())
        my_model = duplicate_model_with_quant(my_model, bits=fwd_bits, overflow_rate=overflow_rate, counter=n_sample, type=quant_method)
        val_ds_tmp2 = get(10, data_root=data_root, train=False, input_size=input_size)
        eval_model(my_model, val_ds_tmp2, ngpu=ngpu, n_sample=100)
        print(my_model)
        for i in [0, 4]:
            sf = my_model.features[i].sf
            my_model.features[i+1].assign_act_sf(sf)

        for i in [0, 4]:
            sf = my_model.classifier[i].sf
            my_model.features[i+1].assign_act_sf(sf)
    val_ds = get(batch_size, data_root=data_root, train=False, input_size=input_size) 
    val_ds2 = get(batch_size, data_root=data_root, train=False, input_size=input_size) 
    print("before accuracu eval \n\nmy model\n\n\n")
    acc1a, acc5a = eval_model1(my_model, val_ds, ngpu = ngpu)
    print("after accuracy eval")
    res_str2 = "type={}, quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
        typer, quant_method, param_bits, bn_bits, fwd_bits, overflow_rate, acc1a, acc5a)
    print(res_str2)

    acc1, acc5 = eval_model1(model_raw, val_ds2, ngpu=ngpu)
    # print sf
    print(model_raw)
    res_str = "type={}, quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
        typer, quant_method, param_bits, bn_bits, fwd_bits, overflow_rate, acc1, acc5)
    print("approximate results")
    print(res_str)

    acc1, acc5 = eval_model1(model_raw, val_ds2, ngpu=ngpu)
    # print sf
    print(model_raw)
    res_str = "type={}, quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
        typer, quant_method, param_bits, bn_bits, fwd_bits, overflow_rate, acc1, acc5)
    print("Exact results")
    print(res_str)
        

if __name__ == '__main__':
    main()

