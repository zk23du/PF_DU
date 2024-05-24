import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 16


class LocalModel(nn.Module):
    def __init__(self, base, predictor):
        super(LocalModel, self).__init__()

        self.base = base
        self.predictor = predictor

        self.gamma, self.beta = None, None
        
    def forward(self, x):
        out = self.base(x)
        if self.gamma != None:
            out = (1 + self.gamma) * out + self.beta
        out = self.predictor(out)

        return out
        

# class FedAvgCNN(nn.Module):
#     def __init__(self, in_features=1, num_classes=10, dim=1024, dropout=0.0):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_features,
#                                32,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.conv2 = nn.Conv2d(32,
#                                64,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.fc1 = nn.Linear(dim, 512)
#         self.fc = nn.Linear(512, num_classes)

#         self.act = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, x):
#         x = self.act(self.conv1(x))
#         x = self.maxpool(x)
#         x = self.dropout(x)
#         x = self.act(self.conv2(x))
#         x = self.maxpool(x)
#         x = self.dropout(x)
#         x = torch.flatten(x, 1)
#         x = self.act(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x


# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/cnn.py
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features,
                               32,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.fc1 = nn.Linear(dim, 512)
        self.fc = nn.Linear(512, num_classes)

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc(x)
        return x

# ====================================================================================================================

# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
class FedAvgMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

# ====================================================================================================================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, batch_size, 2, 1)
        self.conv2 = nn.Conv2d(batch_size, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

# ====================================================================================================================

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=1*28*28, num_classes=10):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

# ====================================================================================================================

class DNN(nn.Module):
    def __init__(self, input_dim=1*28*28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

class CifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGGbatch_size': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         output = F.log_softmax(out, dim=1)
#         return output

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

# ====================================================================================================================

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class LeNet(nn.Module):
    def __init__(self, feature_dim=50*4*4, bottleneck_dim=256, num_classes=10, iswn=None):
        super(LeNet, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

# class CNNCifar(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, batch_size, 5)
#         self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 100)
#         self.fc3 = nn.Linear(100, num_classes)

#         # self.weight_keys = [['fc1.weight', 'fc1.bias'],
#         #                     ['fc2.weight', 'fc2.bias'],
#         #                     ['fc3.weight', 'fc3.bias'],
#         #                     ['conv2.weight', 'conv2.bias'],
#         #                     ['conv1.weight', 'conv1.bias'],
#         #                     ]
                            
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, batch_size * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)
#         return x

# ====================================================================================================================

class LSTMNet(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        dims = hidden_dim*2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        text, text_lengths = x
        
        embedded = self.embedding(text)
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        #unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:,-1,:])
        out = self.dropout(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
            
        return out

# ====================================================================================================================

class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = F.log_softmax(z, dim=1)

        return out

# ====================================================================================================================

class TextCNN(nn.Module):
    def __init__(self, hidden_dim, num_channels=100, kernel_size=[3,4,5], max_len=200, dropout=0.8, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super(TextCNN, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0]+1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1]+1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2]+1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels*len(kernel_size), num_classes)
        
    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text).permute(0,2,1)
        
        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)
        
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)

        return out

# class TextCNN(nn.Module):
#     def __init__(self, hidden_dim, num_channels=100, kernel_size=[3,4,5], max_len=200, dropout=0.8, 
#                 padding_idx=0, vocab_size=98635, num_classes=10):
#         super(TextCNN, self).__init__()
        
#         # Embedding Layer
#         self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
#         # This stackoverflow thread clarifies how conv1d works
#         # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
#             nn.ReLU(),
#             nn.MaxPool1d(max_len - kernel_size[0]+1)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
#             nn.ReLU(),
#             nn.MaxPool1d(max_len - kernel_size[1]+1)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
#             nn.ReLU(),
#             nn.MaxPool1d(max_len - kernel_size[2]+1)
#         )
        
#         self.dropout = nn.Dropout(dropout)
        
#         # Fully-Connected Layer
#         self.fc = nn.Linear(num_channels*len(kernel_size), num_classes)
    
#     def shape_matcher(self, T1, T2):
#         if T1.shape == T2.shape:
#             return T1, T2
#         elif T1.shape[-1] < T2.shape[-1]:
#             T1 = F.pad(T1, (0,T2.shape[-1]-T1.shape[-1]))
#             return T1, T2
#         elif T2.shape[-1] < T1.shape[-1]:
#             T2 = F.pad(T2, (0,T1.shape[-1]-T2.shape[-1]))
#             return T1, T2
    
#     def forward(self, x):
#         if not isinstance(x, tuple):
#             text = x
#             # If text_lengths are not provided, use a dummy value or compute if possible
#             text_lengths = torch.tensor([len(text)]).to(text.device)
#         else:
#             text, text_lengths = x
            
#         text = text.long()
            
#         #text, text_lengths = x

#         embedded_sent = self.embedding(text).permute(0,2,1)
#         #print(embedded_sent.shape)
#         conv_out1 = self.conv1(embedded_sent).squeeze(2)
#         conv_out2 = self.conv2(embedded_sent).squeeze(2)
#         conv_out3 = self.conv3(embedded_sent).squeeze(2)

#         #print(conv_out1.shape, conv_out2.shape, conv_out3.shape)
        
#         conv_out1, conv_out2 = self.shape_matcher(conv_out1, conv_out2)
#         conv_out2, conv_out3 = self.shape_matcher(conv_out2, conv_out3)
#         conv_out3, conv_out1 = self.shape_matcher(conv_out3, conv_out1)
            
#         all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
#         #print(all_out.shape)
#         final_feature_map = self.dropout(all_out)
#         # print(final_feature_map.shape)
#         # print(final_feature_map.mean(1).shape)
#         # out = self.fc(final_feature_map.mean(1))
#         out = self.fc(final_feature_map)
#         out = F.log_softmax(out, dim=1)

#         return out


# ====================================================================================================================


# class linear(Function):
#   @staticmethod
#   def forward(ctx, input):
#     return input
  
#   @staticmethod
#   def backward(ctx, grad_output):
#     return grad_output


import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
# from distiller.modules import EltwiseAdd

__all__ = [
    'resnet20_cifar', 'resnet32_cifar', 'resnet44_cifar', 'resnet56_cifar'
]

NUM_CLASSES = 10


class EltwiseAdd(nn.Module):
    def __init__(self, inplace=False):
        super(EltwiseAdd, self).__init__()

        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 block_gates,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.block_gates = block_gates
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(
            inplace=False)  # To enable layer removal inplace must be False
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.residual_eltwiseadd = EltwiseAdd()

    def forward(self, x):
        residual = out = x

        if self.block_gates[0]:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)

        if self.block_gates[1]:
            out = self.conv2(out)
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.residual_eltwiseadd(residual, out)
        out = self.relu2(out)

        return out


class ResNetCifar(nn.Module):
    def __init__(self, block, layers, num_classes=NUM_CLASSES):
        self.nlayers = 0
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(3):
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 16  # 64
        super(ResNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3,
                               self.inplanes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 16,
                                       layers[0])
        self.layer2 = self._make_layer(self.layer_gates[1],
                                       block,
                                       32,
                                       layers[1],
                                       stride=2)
        self.layer3 = self._make_layer(self.layer_gates[2],
                                       block,
                                       64,
                                       layers[2],
                                       stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(layer_gates[0], self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20_cifar(**kwargs):
    model = ResNetCifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet18_cifar(**kwargs):
    model = ResNetCifar(BasicBlock, [2 , 2 , 2], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNetCifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNetCifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNetCifar(BasicBlock, [9, 9, 9], **kwargs)
    return model