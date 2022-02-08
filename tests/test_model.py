
from app import *
from app.model import TSCResNet

import torch

def test_model():

    DEBUG = False

    samples, features, timesteps = 16, 40, 172

    x = torch.randn((samples, features, timesteps)) #1 batch

    #print ('INPUT SHAPE: ', x.shape)

    model = TSCResNet(in_channels = features, DEBUG=DEBUG)

    #t0 = time.time()

    out = model(x)

    #print ('OUTPUT SHAPE: ', out.shape)

    #print ('Time: ', time.time() - t0)

    #print("DONE")

    assert out.shape == torch.Size([16, 18])
