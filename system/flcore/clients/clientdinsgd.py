import torch
import torch.nn as nn
import numpy as np
import copy
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
from flcore.optimizers.fedoptimizer import DFW, DINSGD
from flcore.optimizers.dfw import DFW1
from flcore.optimizers.dfw_din import DFWDin
import torch.distributed as dist
from flcore.optimizers.sls import Sls
from torch.optim.optimizer import required
import numpy as np
import torch

   
class clientDinsgd(Client):
    def __init__(self, args, id, train_samples, test_samples,enable_memory_management=True, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        nprocs = torch.cuda.device_count()
        self.round_counter = 0
        self.loss = nn.CrossEntropyLoss()
        self.mu = args.mu
        self.memory_management = enable_memory_management
        self.global_params = copy.deepcopy(list(self.model.parameters()))
        self.optimizer = DINSGD(self.model.parameters(),p=0.001,q=0.5,beta=0.9,rho=nprocs,compression_method='quantize')
        


    def train(self):
        #print("Client Training")
        trainloader = self.load_train_data()        
        start_time = time.time()
        self.model.train()
        max_local_steps = 1
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
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
                #self.optimizer.zero_grad()
                loss.backward()
                #self.optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                if self.round_counter % self.local_steps == 0:
                    self.optimizer.step() 
                self.round_counter += 1 
                    


        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            res, DELTA = get_dp_params(self.optimizer)
            print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")
