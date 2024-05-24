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

   

class clientAVGDFW(Client):
    def __init__(self, args, id, train_samples, test_samples,enable_memory_management=True, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        nprocs = torch.cuda.device_count()
        self.round_counter = 0
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = DFW1(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.98)  # Example scheduler
        self.mu = args.mu
        self.memory_management = enable_memory_management
        self.global_params = copy.deepcopy(list(self.model.parameters()))

        
    def train(self):
        #print("Client Training")
        trainloader = self.load_train_data()        #i got batches for 5 clients divided into a batch of 32
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

        def train(self):
            trainloader = self.load_train_data()
            self.model.train()

            for step in range(self.local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    x = x.to(self.device) if isinstance(x, torch.Tensor) else [x_.to(self.device) for x_ in x]
                    y = y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    loss = self.loss(output, y)
                    loss.backward()
                    self.optimizer.step()

            # Compute differences between local and global parameters
            param_diffs = [local_param.data - global_param.data for local_param, global_param in zip(self.model.parameters(), self.global_params)]
            self.param_diffs = param_diffs  # Store the differences for sending to the server

                
                    
            # if self.memory_management:  # Assuming a flag to control this behavior
            #      torch.cuda.empty_cache()
                 
            # self.scheduler.step()
            
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            res, DELTA = get_dp_params(self.optimizer)
            print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")


