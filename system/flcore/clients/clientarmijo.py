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


def get_optimizer(params, n_batches_per_epoch=None):
    n_batches_per_epoch = 32   
    opt = Sls(params, c = 0.1, n_batches_per_epoch=n_batches_per_epoch,line_search_fn="armijo")
    return opt

class clientArmijo(Client):
    def __init__(self, args, id, train_samples, test_samples,enable_memory_management=True, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        nprocs = torch.cuda.device_count()
        self.round_counter = 0
        self.loss = nn.CrossEntropyLoss()
        self.mu = args.mu
        self.global_params = copy.deepcopy(list(self.model.parameters()))
        self.memory_management = enable_memory_management   
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.98)     
        self.optimizer = get_optimizer( params=self.model.parameters(), n_batches_per_epoch = 32)


    def train(self):
        trainloader = self.load_train_data()        
        start_time = time.time()
        self.model.train()
        
        # differential privacy
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
                
        max_local_steps = self.local_epochs
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
                
                closure = lambda : nn.CrossEntropyLoss()(self.model(x), y)
                self.optimizer.zero_grad()
                loss = self.optimizer.step(closure=closure)

                    
            # if self.memory_management:  # Assuming a flag to control this behavior
            #     torch.cuda.empty_cache() 
                
            # self.scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            res, DELTA = get_dp_params(self.optimizer)
            print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")
