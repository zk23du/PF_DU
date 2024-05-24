import time
from flcore.clients.clientavgdfw import clientAVGDFW
from flcore.servers.serverbasedfw import Server
from threading import Thread
from flcore.optimizers.dfw import DFW1
import copy
import torch

class FedAvg1(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVGDFW)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        # self.load_model()


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])


        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()


# class FedAvg(Server):
#     def __init__(self, args, times,model):
#         super().__init__(args, times)
#         # Other initialization steps...        
#         # select slow clients
#         self.set_slow_clients()
#         self.set_clients(clientAVG)

#         print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
#         print("Finished creating server and clients.")
#         self.Budget = []
#         # self.load_model()
#         self.model = model
#         self.optimizer = DFW1(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

#     def train(self):
#         for i in range(self.global_rounds + 1):
#             s_t = time.time()
#             self.selected_clients = self.select_clients()
#             self.send_models()

#             if i % self.eval_gap == 0:
#                 print(f"\n-------------Round number: {i}-------------")
#                 print("\nEvaluate global model")
#                 self.evaluate()

#             gradients = []  # List to store gradients/model updates from each client

#             for client in self.selected_clients:
#                 client_gradients = client.compute_gradients()  # Modify client to return gradients instead of updating locally
#                 gradients.append(client_gradients)

#             global_gradients = self.aggregate_gradients(gradients)  # Aggregate gradients from all clients

#             # Perform global optimization step using DFW1 algorithm
#             self.optimizer.step(global_gradients)  # Modify to perform DFW1 optimization

#             self.Budget.append(time.time() - s_t)
#             print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            
#         print("\nBest global accuracy.")
#         # self.print_(max(self.rs_test_acc), max(
#         #     self.rs_train_acc), min(self.rs_train_loss))
#         print(max(self.rs_test_acc))

#         self.save_results()
#         self.save_global_model()
        
        
#     def aggregate_gradients(self, gradients):
#         if len(gradients) == 0:
#             return None

#         # Determine the maximum number of parameters based on the first valid client gradients
#         for client_grads in gradients:
#             if client_grads is not None:
#                 max_params = len(client_grads)
#                 break

#         aggregated_gradients = []
#         for param_index in range(max_params):
#             valid_tensors = []
#             for client_grads in gradients:
#                 if client_grads is not None and len(client_grads) > param_index and isinstance(client_grads[param_index], torch.Tensor):
#                     valid_tensors.append(client_grads[param_index])
#                 else:
#                     print(f"Invalid or missing gradient at client {gradients.index(client_grads)}, param index {param_index}")

#             if valid_tensors:
#                 mean_grad = torch.stack(valid_tensors).mean(dim=0)
#                 aggregated_gradients.append(mean_grad)
#             else:
#                 # Ensure that at least one valid gradient tensor exists to use as a template for zeros_like
#                 if any(isinstance(grads[param_index], torch.Tensor) for grads in gradients if grads is not None):
#                     template_tensor = next(grads[param_index] for grads in gradients if isinstance(grads[param_index], torch.Tensor))
#                     aggregated_gradients.append(torch.zeros_like(template_tensor))
#                 else:
#                     print(f"No valid template for zero gradients found at parameter index {param_index}")
#                     # As a fallback, you might need to define a default tensor shape and type
#                     # Example: aggregated_gradients.append(torch.zeros(default_shape, dtype=default_dtype, device=default_device))
#                     # You will need to define default_shape, default_dtype, and default_device based on your model's requirements

#         return aggregated_gradients


# import torch
# import time

# class FedAvg(Server):
#     def __init__(self, args, times, model):
#         super().__init__(args, times)
#         self.model = model
#         self.optimizer = DFW1(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
#         # Initialize other necessary DFW-related variables here if necessary

#     def aggregate_parameters(self, models, client_updates):
#         # Here you need to aggregate updates such as gradients in a way that DFW can utilize them
#         # Assuming client_updates contain 'delta_t', 'r_t' among other things as necessary for DFW

#         with torch.no_grad():
#             for name, param in self.model.named_parameters():
#                 aggregated_tensor = torch.zeros_like(param.data)
#                 total_weight = 0

#                 for model, update in zip(models, client_updates):
#                     # Assuming client_updates are structured with necessary components
#                     weight = update[name]['weight']  # Define how weights are determined (could be based on data size, error, etc.)
#                     delta_t = update[name]['delta_t']
#                     r_t = update[name]['r_t']
#                     aggregated_tensor += (weight * delta_t) / len(models)  # Example: weighted mean of deltas
#                     total_weight += weight

#                 # Here you might want to use your optimizer's step function to update each parameter
#                 self.optimizer.step(aggregated_tensor, r_t)  # You need to adjust or modify your optimizer to handle this

#     def train(self):
#         for i in range(self.global_rounds + 1):
#             s_t = time.time()
#             self.selected_clients = self.select_clients()
#             self.send_models()

#             # Additional server-side operations
#             if i % self.eval_gap == 0:
#                 print(f"\n-------------Round number: {i}-------------")
#                 self.evaluate()

#             for client in self.selected_clients:
#                 client.train()

#             models = self.receive_models()
#             client_updates = self.receive_client_updates()  # This must include the necessary DFW components
#             self.aggregate_parameters(models, client_updates)

#             self.Budget.append(time.time() - s_t)
#             print('-'*25, 'time cost', '-'*25, self.Budget[-1])

#             if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
#                 break

#         print("\nBest accuracy.")
#         print(max(self.rs_test_acc))
#         print("\nAverage time cost per round.")
#         print(sum(self.Budget[1:])/len(self.Budget[1:]))

#         self.save_results()
#         self.save_global_model()

#         if self.num_new_clients > 0:
#             self.eval_new_clients = True
#             self.set_new_clients(clientAVG)
#             print(f"\n-------------Fine tuning round-------------")
#             self.evaluate()






            
            