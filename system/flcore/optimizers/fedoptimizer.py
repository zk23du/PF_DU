import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from collections import defaultdict

       
class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cs, client_cs):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])




class DFW(optim.Optimizer):
    def __init__(self, params, global_params,lr=required, momentum=0.9, weight_decay=0, eps=1e-5, mu=0.0):
        if lr is not required and lr <= 0.0:
            raise ValueError("Invalid eta: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        #defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, mu=mu)

        super(DFW, self).__init__(params, defaults)
        self.eps = eps
        self.global_params = global_params

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)
        
         
    @torch.autograd.no_grad()
    def step(self, global_params , device, closure=None):
        loss = float(closure())

        w_dict = defaultdict(dict)
        for group in self.param_groups:
            wd = group['weight_decay']
            for param in group['params']:
                if param.grad is None:
                    continue
                w_dict[param]['delta_t'] = param.grad.data
                w_dict[param]['r_t'] = wd * param.data

        self._line_search(loss, w_dict)
        for group in self.param_groups:
            lr = group['lr']
            m = group['momentum']
            mu = group['mu']
            for param, g in zip(group['params'], global_params):
                if param.grad is None:
                    continue
                state = self.state[param]
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']

                param.data -= lr * (r_t + self.gamma * delta_t)

                if m:
                    z_t = state['momentum_buffer']
                    z_t *= m
                    z_t -= lr * self.gamma * (delta_t + r_t)
                    param.data += m * z_t

                    g = g.to(device)  # Ensure global_param is on the same device as param
                    #g = torch.tensor(global_params, device=device)
                    d_p = param.grad.data + mu * (param.data - g.data)
                    # Use global_params
                    param.data.add_(d_p, alpha=-lr)

    @torch.autograd.no_grad()
    def _line_search(self, loss, w_dict):
        """
        Computes the line search in closed form.
        """
        num = loss
        denom = 0

        for group in self.param_groups:
            lr = group['lr']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']
                num -= lr * torch.sum(delta_t * r_t)
                denom += lr * delta_t.norm() ** 2

        self.gamma = float((num / (denom + self.eps)).clamp(min=0, max=1))


class SC(optim.Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0, eps=1e-5):
        if lr is not required and lr <= 0.0:
            raise ValueError("Invalid eta: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SC, self).__init__(params, defaults)
        self.eps = eps

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

    @torch.autograd.no_grad()
    def step(self, server_cs, client_cs, closure):
        loss = float(closure())

        w_dict = defaultdict(dict)
        for group in self.param_groups:
            wd = group['weight_decay']
            for param in group['params']:
                if param.grad is None:
                    continue
                w_dict[param]['delta_t'] = param.grad.data
                w_dict[param]['r_t'] = wd * param.data

        self._line_search(loss, w_dict)

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            gamma = self.gamma
            for param, sc, cc in zip(group['params'], server_cs, client_cs):
                if param.grad is None:
                    continue
                state = self.state[param]
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']
                param.data -= lr * (r_t + self.gamma * delta_t)

                if mu:
                    z_t = state['momentum_buffer']
                    z_t *= mu
                    z_t -= lr * self.gamma * (delta_t + r_t)
                    param.data += mu * z_t
                    param.data.add_(other=(delta_t + sc - cc), alpha=-group['lr'])
                                     
    @torch.autograd.no_grad()
    def _line_search(self, loss, w_dict):
        """
        Computes the line search in closed form.
        """
        num = loss
        denom = 0

        for group in self.param_groups:
            lr = group['lr']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']
                num -= lr * torch.sum(delta_t * r_t)
                denom += lr * delta_t.norm() ** 2

        self.gamma = float((num / (denom + self.eps)).clamp(min=0, max=1))

import torch
import numpy as np
import torch.linalg as linalg
import torch.distributed as dist
from torch.distributed import ReduceOp

class DINSGD():
    def __init__(self, params, p=1, q=10, beta=0.9, rho=5, compression_method="quantize"):
        """
        Default values have been taken from the paper https://arxiv.org/pdf/2002.04130.pdf
        Rho is number of workers - Total number of GPUs
        compression_method: ['quantize', 'one_bit', 'sparse_top_k', 'sparse_randomized']
        Compression methods have been taken from below gihub repo.
        - https://github.com/scottjiao/Gradient-Compression-Methods/blob/master/utils.py 
        """
        self.param_groups = []

        if compression_method == "quantize":
            self.compress = self.quantize
        elif compression_method == "one_bit":
            self.compress = self.one_bit
        elif compression_method == "sparse_top_k":
            self.compress = self.sparse_top_k
        elif compression_method == "sparse_randomized":
            self.compress = self.sparse_randomized
        else:
            raise('COMPRESSION METHOD NOT DEFINED')

        defaults = {"P":p, "Q":q, "Beta":beta, "Rho":rho}
        defaults.update({"params":list(params), "memory": [], "M":[]})
        
        self.param_groups.append(defaults)

        for g in self.param_groups:
            for pars in g['params']:
                g['memory'].append(torch.zeros_like(pars.data))
                g['M'].append(torch.zeros_like(pars.data))

    def compress_gradients(self, gradients):
        """
        Compress the gradients using the selected compression method.
        """
        compressed_gradients = []
        for grad in gradients:
            compressed_grad = self.compress(grad)
            compressed_gradients.append(compressed_grad)
        return compressed_gradients

    def step(self):
        """
        Perform a single training step.
        """
        rand_ = self.randomize()

        for groups in self.param_groups:
            P = groups['P']
            Q = groups['Q']
            Beta = groups['Beta']
            Rho = groups['Rho']

            for par in range(len(groups['params'])):

                grad = groups['params'][par].grad.data  #worker
                m_prev =  groups['M'][par]

                corrected_grad, groups['memory'][par] = self.correct_gradients(grad, groups['memory'][par])  #worker

                # Compress the gradients before sending them to the server
                
                compressed_grad = self.compress_gradients([corrected_grad])[0]

                # Send the compressed gradients to the server for federated averaging
                # (replace this with actual code for sending gradients)

                # Receive the averaged gradients from the server
                # (replace this with actual code for receiving gradients)

                # Update the model parameters with the averaged gradients
                groups['M'][par] = Beta * m_prev  + (1-Beta)*compressed_grad/Rho
                eta = 1/(P*linalg.norm(groups['M'][par]) + Q)
                v_t = (groups['params'][par].data - eta*groups['M'][par])*(1-rand_)

                groups['params'][par].data.multiply_(rand_)
                groups['params'][par].data.add_(v_t, alpha=1)

    def randomize(self):  #worker
        return torch.rand(1, device=torch.device('cuda'))

    def correct_gradients(self, x, mem):  #worker

        corrected_gradient = self.compress(mem + x)
        mem = mem + x - corrected_gradient

        return corrected_gradient, mem

    def one_bit(self, x):  #worker
        
        x_norm=torch.norm(x,p=float('inf'))
        sgn_x=((x>0).float()-0.5)*2
        compressed_x=x_norm*sgn_x
        
        return compressed_x 

    def quantize(self,x,input_compress_settings={}):  #BEST
        compress_settings={'n':6}
        # compress_settings.update(input_compress_settings)
        #assume that x is a torch tensor
        
        n=compress_settings['n']
        #print('n:{}'.format(n))
        x=x.float()
        x_norm=torch.norm(x,p=float('inf'))
        
        sgn_x=((x>0).float()-0.5)*2
        
        p=torch.div(torch.abs(x),x_norm)
        renormalize_p=torch.mul(p,n)
        floor_p=torch.floor(renormalize_p)
        compare=torch.rand_like(floor_p)
        final_p=renormalize_p-floor_p
        margin=(compare < final_p).float()
        xi=(floor_p+margin)/n
        
        Tilde_x=x_norm*sgn_x*xi
        
        return Tilde_x

    def sparse_top_k(self, x, input_compress_settings={}):
        compress_settings = {'k': 1 / 16}
        compress_settings.update(input_compress_settings)
        k = compress_settings['k']
        vec_x = x.flatten()
        d = int(len(vec_x))
        k = int(np.ceil(d * k))
        indices = torch.abs(vec_x).topk(k)[1]
        out_x = torch.zeros_like(vec_x)
        out_x[indices] = vec_x[indices]
        out_x = out_x.reshape(x.shape)
        return out_x

