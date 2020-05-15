import scipy.io
import os
import h5py
import numpy as np
import os
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch import optim
import sys
from torch.utils.data import TensorDataset ,DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.nn import Sigmoid

# -*- coding: utf-8 -*-
from joblib import Parallel, delayed
from time import time
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import torch.nn.init
from sklearn.model_selection import cross_val_predict 
from sklearn.model_selection import cross_val_score
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import requests
from torch.nn import Softmax
import matplotlib.pyplot as plt
from torch.nn import LogSoftmax
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from torchvision import datasets, transforms

import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch import optim

from torch.utils.data import TensorDataset ,DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.nn import Sigmoid
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sklearn

import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import torch.nn.init
from sklearn.model_selection import cross_val_predict 
from sklearn.model_selection import cross_val_score
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import requests
from torch.nn import Softmax
import matplotlib.pyplot as plt
from torch.nn import LogSoftmax
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import gc 
import torch.nn.functional as F
import random
import argparse

#自作モジュール
from utils import seed_everything
from utils import load_metanalysis,load_subject_path
import dataset
import models
#ArgumentParser
#https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0

parser = argparse.ArgumentParser("Task")
#seed
parser.add_argument('--seed', type=int, default=2019, help="seed for seed_everything")

#datapath
parser.add_argument('--datapath', type=str, default="/mnt/koyama/koyama/data_old/langMSMALL_csv/complete/", help="MSMALL data")
parser.add_argument('--banddatapath', type=str, default="/mnt/koyama/koyama/data_old/langBandPassedcsv/0.01-0.1Hz/complete/", help="BANDPASS data")
parser.add_argument('--metanst', type=str, default="/mnt/koyama/koyama/data_old/NST_csv/", help="NST_csv")
parser.add_argument('--metanqt', type=str, default="/mnt/koyama/koyama/data_old/NQT_csv/", help="NQT_csv")

#which metaanalysis
parser.add_argument('--metaanalysis', type=str, choices=['NQT', 'NST', '12'], help="select metanalysis")

parser.add_argument('--datatype', type=str, choices=['MSMALL', 'BandPassed'], help="MSMALL or Bandpassed")
parser.add_argument('--runtype', type=str, choices=['mixrun', 'separate_run'], help="mixrun or separate_run")

parser.add_argument('--modeltype', type=str, default="Simple", help="modeltype")

# dataset


"""parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='cnn')
# misc
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")"""

args = parser.parse_args()

def main():
    seed_everything(args.seed)
    metanalysis = load_metanalysis(args.metaanalysis,args.metanqt,args.metanst)
    group1_sub_path,group2_sub_path = load_subject_path(args.datapath)
    
    
    train_dataloader,test_dataloader = dataset.create(args.datatype,
                                                      args.runtype,
                                                      30,
                                                      True,
                                                      False,
                                                      args.datapath,
                                                      group1_sub_path,
                                                      group2_sub_path)
    
    model = models.create(args.modeltype,59412,5,args.seed,xx=None)
    
    print(model)
if __name__ == '__main__':
    main()