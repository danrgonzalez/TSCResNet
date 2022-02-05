import time
from config import NUM_CLASSES

import torch
import torch.nn as nn

from architecture import config

class CNNBlock(nn.Module):

    #Tuple is structured by (filters, kernel_size, stride) 
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):

        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        
        self.use_bn_act = bn_act #batch normalization, True or False

    def forward(self, x):

        #Question, when do we not need batch norm? 1) In ScaledPred (it appears this is the only place)
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x) 

class ResidualBlock(nn.Module):

    #List is structured by "B" indicating a residual block followed by the number of repeats
    def __init__(self, channels, use_residual=True, num_repeats=1):

        super().__init__()

        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1)
                    )
                ]

        self.use_residual = use_residual #Is use_residual ever false?
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x

class PredictionBlock(nn.Module): #Conv layers reshaped for prediction

    def __init__(self, in_channels, num_classes = 18):

        super().__init__()

        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, 1, bn_act=False, kernel_size=1) #Key: X Cin to 1 Cout
        )
        self.num_classes = num_classes

    def forward(self, x):

        if DEBUG:
            print ('Prediction Block Shape, before Reshaping:', self.pred(x).shape)

        return self.pred(x).reshape(x.shape[0], self.num_classes)

class TSCResNet(nn.Module):

    def __init__(self, in_channels = 3):

        super().__init__()

        self.in_channels = in_channels #gets updated throughout the passes/layers

        self.layers = self._create_layers()

    def _create_layers(self):

        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config: #see architecture.py
            
            if isinstance(module, tuple):

                    out_channels, kernel_size, stride = module

                    layers.append(
                        CNNBlock(in_channels, 
                                out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride, 
                                padding=1 if kernel_size == 3 else 0) #to maintain output shape
                    )
                    in_channels = out_channels

            elif isinstance(module, list):

                num_repeats = module[1]

                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

            elif module == 'P':

                layers.append(PredictionBlock(in_channels))
        
        return layers

    def forward(self, x):

        for layer in self.layers:

            if DEBUG:
                print ('~~~~~~~~~~~~~~~~~~~~')
                print ('Input to Block shape: ', x.shape)

            x = layer(x)

            if DEBUG:
                print (layer)
                print ('Output of Block shape: ', x.shape)
                print ('~~~~~~~~~~~~~~~~~~~~')
                print ('')

        return x

if __name__ == '__main__':

    DEBUG = True

    samples, features, timesteps = 16, 40, 172

    x = torch.randn((samples, features, timesteps)) #1 batch

    print ('INPUT SHAPE: ', x.shape)

    model = TSCResNet(in_channels = features)

    t0 = time.time()

    out = model(x)

    print ('OUTPUT SHAPE: ', out.shape)

    print ('Time: ', time.time() - t0)

    print("DONE")
