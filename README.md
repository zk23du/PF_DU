# PF_DU
Personalized Federated Learning 
This repository conatins the personalized framework of fedearted learning with modifications and inclution of new optimizers. (Origainlly propsed by PFLib [https://github.com/TsingZ0/PFLlib]) 
Here I propose a user-friendly algorithm library (with an integrated evaluation platform) for beginners who intend to start federated learning (FL) study. A detailed explantion is stated below.

## Creating the conda environment
```sh
conda create --name PF_DU
conda activate PF_DU
## install the necessary libarries as in requiremnets.txt
### navigate to your folder.
```

## Datasets
For the label skew scenario, I have introduced some famous datasets: MNIST, EMNIST, Fashion-MNIST, Cifar10, Cifar100, AG News, Sogou News and Shakespeare which can be easily splitted into IID and non-IID versions. In the non-IID scenario, 2 situations exist. The first one is the pathological non-IID scenario, the second one is the practical non-IID scenario. In the pathological non-IID scenario, for example, the data on each client only contains the specific number of labels (maybe only 2 labels), though the data on all clients contains 10 labels such as the MNIST dataset. In the practical non-IID scenario, Dirichlet distribution is utilized . We can input balance for the iid scenario, where the data are uniformly distributed.

### Examples for MNIST dataset.
Navigate to dataset folder and the required .py file.
```sh
# python generate_MNIST.py iid - - # for iid and unbalanced scenario
# python generate_MNIST.py iid balance - # for iid and balanced scenario
# python generate_MNIST.py noniid - pat # for pathological noniid and unbalanced scenario
python generate_MNIST.py noniid - dir # for practical noniid and unbalanced scenario
# python generate_MNIST.py noniid - exdir # for Extended Dirichlet strategy 
```
### Output
```sh
Number of classes: 10
Client 0         Size of data: 2630      Labels:  [0 1 4 5 7 8 9]
                 Samples of labels:  [(0, 140), (1, 890), (4, 1), (5, 319), (7, 29), (8, 1067), (9, 184)]
--------------------------------------------------
Client 1         Size of data: 499       Labels:  [0 2 5 6 8 9]
                 Samples of labels:  [(0, 5), (2, 27), (5, 19), (6, 335), (8, 6), (9, 107)]
--------------------------------------------------
Client 2         Size of data: 1630      Labels:  [0 3 6 9]
                 Samples of labels:  [(0, 3), (3, 143), (6, 1461), (9, 23)]
--------------------------------------------------
```
## Models Used
Navigate to system --> flcore --> trainmodel. 
The default models are used here, yu can add you own models in the models.py file or create a new .py file and import it.

1. for MNIST and Fashion-MNIST
Mclr_Logistic(1*28*28)
LeNet()
DNN(1*28*28, 100) # non-convex

2. for Cifar10, Cifar100 and Tiny-ImageNet
Mclr_Logistic(3*32*32)
FedAvgCNN()
DNN(3*32*32, 100) # non-convex
ResNet18 (and other versions), AlexNet, MobileNet, GoogleNet, etc.

3.for AG_News and Sogou_News
LSTM()
fastText() in Bag of Tricks for Efficient Text Classification
TextCNN() in Convolutional Neural Networks for Sentence Classification
TransformerModel() in Attention is all you need

## Optimizers Used.
We have a client and server setup for our federated learning. The optimizers are used in the client side for achieving the results. One can extend it to the server side as well.
Navigate to system --> flcore --> optimizers. (here you can add new optimizers and import them in client class.
Optimizers used in this repo.
1. SGD
2. Armijo (Sls) [Link to paper : [Armijo](https://arxiv.org/abs/1905.09997)
3. Deep Franke Wolfe [Link to paper : [DFW](https://arxiv.org/abs/1811.07591)
4. Variations of DFW used fro FedProx and Scaffold.
5. DinSGD

## Running the code.
1. Download the datasets as described above.
2. Navigate to system folder.
3. Running the main.py file :
   The arguments in the main.py file can be adjusted depending upon the 
   algorithm to be used. One can set the hyperparmeters and run this file.
   ``` python3 main.py ```
4. Another file Experiments.py file contains the code to run multiple experiments simultaneously. Specify the model, algorithms and other details int the file to run it.
   ``` python3 Experiments.py ```
5. for running code one can also use the below command to run codes and store them in txt file.
   ``` nohup python3 main.py > example.txt & ```



