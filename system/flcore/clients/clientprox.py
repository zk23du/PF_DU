import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import DFW


class clientProxDFW(Client):
    def __init__(self, args, id, train_samples, test_samples, enable_memory_management=True,**kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu

        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = self.optimizer = DFW(self.model.parameters(),self.global_params, lr=0.1, momentum=0.9, mu = self.mu)
        self.round_counter = 0
        self.memory_management = enable_memory_management
        
    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # Zero gradients only at the start of every 'local_steps' batches
                if self.round_counter % self.local_steps == 0:
                    self.optimizer.zero_grad()
                
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

                # Step the optimizer (i.e., update model parameters) every 'local_steps' batches
                if self.round_counter % self.local_steps == 0:
                    self.optimizer.step(self.global_params, self.device, lambda: float(loss))
                self.round_counter += 1 
                
            if self.memory_management:  # Assuming a flag to control this behavior
                torch.cuda.empty_cache()
                
        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()
