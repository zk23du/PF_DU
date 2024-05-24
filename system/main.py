#!/usr/bin/env python
import copy
#import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
from flcore.servers.serveravg import FedAvg
from flcore.servers.serversgd import FedSGD
from flcore.servers.serverprox import FedProxDFW
from flcore.servers.serverscaffold import SCAFFOLDDFW
from flcore.servers.serverdinsgd import FedDinsgd
from flcore.servers.serverscaff import SCAFFOLD
from flcore.servers.serverpgd import FedPGD
from flcore.trainmodel.models import *
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.transformer import *
from flcore.trainmodel.models import TextCNN, LSTMNet
from flcore.trainmodel.models import resnet20_cifar
#from flcore.trainmodel.res import *
from flcore.trainmodel.resnet import resnet20
from flcore.trainmodel.models import fastText
from utils.result_utils import average_data
from utils.mem_utils import MemReporter
import random
import numpy

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
#torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 387114   #98635 for AG_News and 399198 for Sogou_News
max_len=200
emb_dim=32

def set_random_seeds(random_seed=149131323):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(random_seed)
    random.seed(random_seed)
set_random_seeds()

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
        
        elif model_str == "resnet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)
            
        elif model_str == "resnet20":
            args.model = resnet20().to(args.device)

            
        elif model_str == "Resnet20":
            args.model = resnet20_cifar().to(args.device)
        
        elif model_str == "resnet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)
                 
        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim, output_size=args.num_classes, 
                        num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                        embedding_length=emb_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size, 
                            num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, d_hid=emb_dim, nlayers=2, 
                            num_classes=args.num_classes).to(args.device)
        
        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
            
        if args.algorithm == "FedAvg":
            server = FedAvg(args,i)
            
        elif args.algorithm == "FedSGD":
            server = FedSGD(args, i)

        elif args.algorithm == "FedProx":
            server = FedProxDFW(args, i)

        elif args.algorithm == "FedPGD":
            server = FedPGD(args, i)

        elif args.algorithm == "SCAFFOLDFW":
            server = SCAFFOLDDFW(args, i)

        elif args.algorithm == "SCAFF":
            server = SCAFFOLD(args, i)
            
            
        elif args.algorithm == "Dinsgd":
            server = FedDinsgd(args, i)
            
       
        else:
            print(args.algorithm)
            raise NotImplementedError

        server.train()
        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="Resnet20")
    parser.add_argument('-lbs', "--batch_size", type=int, default=32)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.001,
                        help="Local learning rate")
    parser.add_argument('-wd', "--weight_decay", type=float, default=1e-4,
                        help="Weight Decay")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=150)
    parser.add_argument('-le', "--local_epochs", type=int, default=2, 
                        help="Multiple update steps in one local epoch.")    
    
    parser.add_argument('-ls', "--local_steps", type=int, default=1, 
                        help="Multiple update steps in one local steps.")   
         
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAdam")
    parser.add_argument('-jr', "--join_ratio", type=float, default=20,help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=True,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=100,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    
    # FedAdam
    parser.add_argument('--beta1', type=float, default=0.9, help='Coefficient for first moment in Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Coefficient for second moment in Adam optimizer')
    parser.add_argument('--server_lr', type=float, default=1, help='Server Learning Rate') 
    parser.add_argument('--tau', type=float, default=0.001, help='Adaptivity')   


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local Steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    print("=" * 50)

    run(args)
