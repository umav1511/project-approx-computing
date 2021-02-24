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
from nopythonhenkel import top

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
        if idx >= (n_sample) - 1:
            break

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5

def load_state_dict(model, model_urls, model_root):
    from torch.utils import model_zoo
    from torch import nn
    import re
    from collections import OrderedDict
    own_state_old = model.state_dict()
    own_state = OrderedDict() # remove all 'group' string
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = model_zoo.load_url(model_urls, model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            print(own_state.keys())
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

    missing = set(own_state.keys()) - set(state_dict.keys())
    no_use = set(state_dict.keys()) - set(own_state.keys())
    if len(no_use) > 0:
        raise KeyError('some keys are not used: "{}"'.format(no_use))

#@jit(parallel = True, nopython=True)
#@jit(parallel = True)
def mult_implement( ip, w, res, ip_batch_size , weight_size, ip_neurons, sign_res):
        abs_ip = np.abs(ip)
        abs_w = np.abs(w)
        sign_ip = np.sign(ip)
        sign_w = np.sign(w)

        for i in range(ip_batch_size):
           for j in range(weight_size):
              sign_res[i][j] = sign_ip[i] * sign_w[j]
              for y in range(ip_neurons):
                  res[i][j][y] = top((int)(abs_ip[i][y]), (int)(abs_w[j][y]), (int)(4)) 

        return res*sign_res

#@jit(parallel = True)
def mult_implement_dummy( a, b, res, k , l, z):
        for i in prange( k):
           for j in prange(l):
              res[i][j] = sum((list(map(mymul, a[i], b[j])))) 
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
      if self.counter>0:
        self.counter = self.counter-1
        k = input.size()[0]
        l = self.weight.size()[0]
        z = input.size()[1]
        wx = np.zeros((k, l), dtype = int)
        x1 = input.cpu().detach().numpy()
        x2 = self.weight.cpu().detach().numpy()
        mmm = mult_implement_dummy(x1, x2, wx, k, l ,z)
        w_times_x= torch.from_numpy(mmm)
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
        print(input.size())
        print(self.weight.size())
        input1 = input/(math.pow(2, -self.act_sf))
        w = self.weight/(math.pow(2,-self.weight_sf))

        input1=input1.int()
        w=w.int()

        x1 = input1.cpu().detach().numpy()
        x2 = w.cpu().detach().numpy()
        sign_ip = np.sign(x1)
        abs_ip  = np.abs(x1)
        sign_w  = np.sign(x2)
        abs_w   = np.abs(x2)  
        mmm = mult_implement(x1, x2, wx, ip_batch_size, weight_size ,ip_neurons, res_sign)
        
        mmm = np.sum(mmm, axis = 2)
        w_times_x= torch.from_numpy(mmm)
        w_times_x = w_times_x * (math.pow(2, -self.act_sf)) * (math.pow(2,-self.weight_sf))
        repeat_bias = self.bias.repeat(input.size()[0], 1)
        w2 = w_times_x.add(repeat_bias)
        y = w_times_x.add(repeat_bias)
        return w2  # w times x + b

    def assign_act_sf(self, act_sf):
        self.act_sf = act_sf

def min_max_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    min_val, max_val = input.min(), input.max()
    if isinstance(min_val, Variable):
        max_val = float(max_val.data.cpu().numpy())
        min_val = float(min_val.data.cpu().numpy())

    input_rescale = (input - min_val) / (max_val - min_val)

    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n

    v =  v * (max_val - min_val) + min_val
    return v

def log_minmax_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = min_max_quantize(input0, bits-1)
    v = torch.exp(v) * s
    return v

class NormalQuant(nn.Module):
    def __init__(self, name, bits, quant_func):
        super(NormalQuant, self).__init__()
        self.name = name
        self.bits = bits
        self.quant_func = quant_func

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        output = self.quant_func(input, self.bits)
        return output

    def __repr__(self):
        return '{}(bits={})'.format(self.__class__.__name__, self.bits)


class MyApproxModel(nn.Module):
    def __init__(self, weight_sf, input_dims, n_hiddens, n_class, counter):
        super(MyApproxModel, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = MyLinearLayer(weight_sf.pop(0), current_dims, n_hidden, counter)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = MyLinearLayer(weight_sf.pop(0), current_dims, n_class, counter)

        self.model= nn.Sequential(layers)
        #print(self.model)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)

model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
}

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model= nn.Sequential(layers)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)

def mnist_model(input_dims=784, n_hiddens=[256,256], n_class=10, pretrained=None):
    model = MLP(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['mnist'], map_location=torch.device('cpu'))
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model



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


def mnist(cuda=False, model_root=None):
    print("Building and initializing mnist parameters")
    
    m = mnist_model(pretrained=os.path.join(model_root, 'mnist.pth'))
    if cuda:
        m = m.cuda()
    return m, get, False

def select(model_name, **kwargs):
    kwargs.setdefault('model_root', os.path.expanduser('~/.torch/models'))
    return mnist(**kwargs)




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

def tanh_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    input = torch.tanh(input) # [-1, 1]
    input_rescale = (input + 1.0) / 2 #[0, 1]
    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = 2 * v - 1 # [-1, 1]

    v = 0.5 * torch.log((1 + v) / (1 - v)) # arctanh
    return v

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
            elif isinstance(v, (MyLinearLayer )):
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


def main():
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
  
    torch.manual_seed(1)
    model_raw, ds_fetcher, is_imagenet = select(mnist, model_root= model_root)

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
            elif quant_method == 'log':
                v_quant = log_minmax_quantize(v, bits=bits)
            elif quant_method == 'minmax':
                v_quant = min_max_quantize(v, bits=bits)
            else:
                v_quant = tanh_quantize(v, bits=bits)
           
            state_dict_quant[k] = v_quant
            print(k, bits)
        model_raw.load_state_dict(state_dict_quant)

    if fwd_bits < 32:

        print(model_raw.state_dict())
        model_raw = duplicate_model_with_quant(model_raw, bits=fwd_bits, overflow_rate=overflow_rate, counter=n_sample, type=quant_method)
        my_model = MyApproxModel( weights_sf, input_dims=784, n_hiddens=[256, 256], n_class=10, counter=n_sample)

        my_model.load_state_dict(model_raw.state_dict())
        my_model = duplicate_model_with_quant(my_model, bits=fwd_bits, overflow_rate=overflow_rate, counter=n_sample, type=quant_method)
        val_ds_tmp = ds_fetcher(10, data_root=data_root, train=False, input_size=input_size)
        val_ds_tmp2 = ds_fetcher(10, data_root=data_root, train=False, input_size=input_size)
        eval_model(model_raw, val_ds_tmp, ngpu=ngpu, n_sample=100, is_imagenet=is_imagenet)
        eval_model(my_model, val_ds_tmp2, ngpu=ngpu, n_sample=100, is_imagenet=is_imagenet)
        for i in [0, 4, 8]:
            sf = my_model.model[i].sf
            my_model.model[i+1].assign_act_sf(sf)

    # eval model
    val_ds = ds_fetcher(batch_size, data_root=data_root, train=False, input_size=input_size) 
    val_ds2 = ds_fetcher(batch_size, data_root=data_root, train=False, input_size=input_size) 
    acc1a, acc5a = eval_model1(my_model, val_ds, ngpu = ngpu, is_imagenet = is_imagenet)
    res_str2 = "type={}, quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
        typer, quant_method, param_bits, bn_bits, fwd_bits, overflow_rate, acc1a, acc5a)
    print("approximate results")
    print(res_str2)

    acc1, acc5 = eval_model1(model_raw, val_ds2, ngpu=ngpu, is_imagenet=is_imagenet)
    # print sf
    res_str = "type={}, quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
        typer, quant_method, param_bits, bn_bits, fwd_bits, overflow_rate, acc1, acc5)
    print("exact results")
    print(res_str)

if __name__ == '__main__':
    main()


