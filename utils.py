import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix

from model import *

def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    if dataset == 'IMDB':
        pass
    elif dataset == 'AG_NEWS':
        pass
    elif dataset == 'DBpedia':
        pass
    elif dataset == 'SST2':
        pass
    
