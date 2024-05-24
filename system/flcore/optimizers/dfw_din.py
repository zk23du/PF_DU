import torch
from torch.optim import Optimizer
import torch
import torch.optim as optim
from torch.optim.optimizer import required
from collections import defaultdict
import numpy as np
import torch.linalg as linalg
import torch.distributed as dist
from torch.distributed import ReduceOp

class DFWDin(optim.Optimizer):
    def __init__(self, params, lr=required, momentum=0, weight_decay=0, eps=1e-5, p=1, q=10, beta=0.9, rho=1, compression_method="quantize"):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps, P=p, Q=q, Beta=beta, Rho=rho)
        super(DFWDin, self).__init__(params, defaults)

        self.p = p
        self.q = q
        self.beta = beta
        self.rho = rho
        self.eps = eps

        if compression_method == "quantize":
            self.compress = self.quantize
        elif compression_method == "one_bit":
            self.compress = self.one_bit
        elif compression_method == "sparse_top_k":
            self.compress = self.sparse_top_k
        elif compression_method == "sparse_randomized":
            self.compress = self.sparse_randomized
        else:
            raise ValueError('COMPRESSION METHOD NOT DEFINED')

        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('weight_decay', 0)
            group.setdefault('P', p)
            group.setdefault('Q', q)
            group.setdefault('Beta', beta)
            group.setdefault('Rho', rho)

            for p in group['params']:
                self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)
                self.state[p]['memory'] = torch.zeros_like(p.data, device=torch.device('cuda'))
                self.state[p]['M'] = torch.zeros_like(p.data, device=torch.device('cuda'))

    @torch.autograd.no_grad()
    def step(self, closure):
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
            P = group['P']
            Q = group['Q']
            Beta = group['Beta']
            Rho = group['Rho']

            for param in group['params']:
                if param.grad is None:
                    continue

                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']
                m_prev = self.state[param]['M']

                corrected_grad, self.state[param]['memory'] = self.correct_gradients(delta_t, self.state[param]['memory'])
                dist.all_reduce(corrected_grad, op=ReduceOp.SUM)

                self.state[param]['M'] = Beta * m_prev + (1 - Beta) * corrected_grad / Rho
                lr = 1 / (P * linalg.norm(self.state[param]['M']) + Q)
                v_t = (param.data - lr * self.state[param]['M']) * (1 - self.randomize())

                param.data.multiply_(self.randomize())
                param.data.add_(v_t, alpha=1)

                if mu:
                    z_t = self.state[param]['momentum_buffer']
                    z_t *= mu
                    z_t -= lr * self.gamma * (delta_t + r_t)
                    param.data += mu * z_t

    @torch.autograd.no_grad()
    def _line_search(self, loss, w_dict):
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

    def randomize(self):
        return torch.rand(1, device=torch.device('cuda'))

    def correct_gradients(self, x, mem):
        corrected_gradient = self.compress(mem + x)
        mem = mem + x - corrected_gradient
        return corrected_gradient, mem

    def one_bit(self, x):
        x_norm = torch.norm(x, p=float('inf'))
        sgn_x = ((x > 0).float() - 0.5) * 2
        compressed_x = x_norm * sgn_x
        return compressed_x

    def quantize(self, x, input_compress_settings={}):
        compress_settings = {'n': 6}
        n = compress_settings['n']
        x = x.float()
        x_norm = torch.norm(x, p=float('inf'))
        sgn_x = ((x > 0).float() - 0.5) * 2
        p = torch.div(torch.abs(x), x_norm)
        renormalize_p = torch.mul(p, n)
        floor_p = torch.floor(renormalize_p)
        compare = torch.rand_like(floor_p)
        final_p = renormalize_p - floor_p
        margin = (compare < final_p).float()
        xi = (floor_p + margin) / n
        Tilde_x = x_norm * sgn_x * xi
        return Tilde_x

    def sparse_top_k(self, x, input_compress_settings={}):
        compress_settings = {'k': 1 / 16}
        k = compress_settings['k']
        vec_x = x.flatten()
        d = int(len(vec_x))
        k = int(np.ceil(d * k))
        indices = torch.abs(vec_x).topk(k)[1]
        out_x = torch.zeros_like(vec_x)
        out_x[indices] = vec_x[indices]
        out_x = out_x.reshape(x.shape)
        return out_x
