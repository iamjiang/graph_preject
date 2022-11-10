from collections  import OrderedDict
import copy
import argparse
import itertools
import os
import numpy as np
from numpy import save,load,savetxt,loadtxt,savez_compressed
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder, label_binarize
import catboost
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, sum_models

import pandas as pd
import scipy.sparse as sp
import time
from tqdm import tqdm, tqdm_notebook,tnrange
tqdm.pandas(position=0, leave=True)
import math 
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import edge_subgraph
from dgl.nn.functional import edge_softmax
import dgl.nn as dglnn
import dgl.function as fn

from GraphSage_Model import *
from evaluation import *
from inductive_graph import *
from graph_to_dataframe import *
from MLP_Model import *
from print_func import *

from MLP_run import *
from catboost_run import *
from GraphSage_run import *
from GraphSage_featureless_run import *

import functools
import seaborn as sns
import pickle
import random

import warnings
warnings.filterwarnings('ignore')
import utils

print("torch version is {}".format(th.__version__))
print("DGL version is {}".format(dgl.__version__))


def seed_everything(seed):
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    np.random.seed(seed)
    dgl.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
        
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--h_dim", type=int, default=32,
            help="number of hidden units")
#     parser.add_argument("--out_dim", type=int, default=1,
#             help="output dimension")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--featureless_lr", type=float, default=1e-4,
            help='Learning Rate for featureless graph model')
    parser.add_argument("--num_bases", type=int, default=5,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--num_layers", type=int, default=1,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n_epochs", type=int, default=3,
            help="number of training epochs")
#     parser.add_argument("--model_path", type=str, default="/workspace/cjiang/eagle_project/CAP_graph/CAP_without_zipcode/rgcn_model_param.pt",
#             help='path for save the model')
    parser.add_argument("--l2norm", type=float, default=1e-3,
            help="l2 norm coef")
    parser.add_argument("--use_self_loop", default=True, action='store_true',
            help="include self feature as a special relation")
    parser.add_argument("--batch-size", type=int, default=10240,
            help="Mini-batch size. If -1, use full graph training.")
    parser.add_argument("--fanout", type=int, default=15,
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")
    parser.add_argument("--loss_weight",  type=bool,default=True,  ## number of label=0/number of label=1
            help="weight for unbalance data")
    parser.add_argument("--num_worker",  type=int,default=0,  
            help="number of worker for neighbor sampling") 
    parser.add_argument("--train_test_split", type=float, default=0.1,
            help="the proportion of test dataset")
    
    parser.add_argument("--loss_function", type=str, default="MultiClass",
            help='Loss function for Catboost')    
    parser.add_argument("--CatBoost_LR", type=float, default=0.01,
            help='Learning Rate for Catboost')  
    parser.add_argument("--iterations", type=int, default=3000,
            help='epochs iterations for Catboost')  
    parser.add_argument("--early_stopping", type=int, default=200,
            help='early_stopping rounds for Catboost') 
    parser.add_argument("--device_type", type=str, default="GPU",
            help='GPU utilization for Catboost training')      
    parser.add_argument("--verbose", type=int, default=200,
            help='verbose details for Catboost training')  
    
    args=parser.parse_args()
    print(args)
    print()
    
    seed_everything(args.seed)
    
    
    KG_dir="/workspace/cjiang/eagle_project/CAP_graph/BGNN/"

    start=time.time()
    with open(os.path.join(KG_dir,'CAP_Graph_v1'), 'rb') as f:
        G,multi_label,binary_label,\
        train_mask_multi_label,  val_mask_multi_label,  test_mask_multi_label,\
        train_mask_binary_label, val_mask_binary_label, test_mask_binary_label= pickle.load(f)
    end=time.time()
    print("It took {:0.4f} seconds to load graph".format(end-start))

    usaanr_feat=[]
    for key, scheme in G.node_attr_schemes(ntype="usaanr").items():
        usaanr_feat.append(key)

    usaanr_feat=[x for x in usaanr_feat if x not in 
                 ['usaanr','cmpyelig','ACTCORP','Segment','train_mask','val_mask','test_mask','label','_ID']]

    print()
    print("The features associated with USAA Member are\n ")
    for i in usaanr_feat:
        print(i)
    print()
    
    G.nodes['usaanr'].data['label']=binary_label
    
#     dict_edges={}
#     for etype in G.etypes:
#         dict_edges[etype]=th.arange(G.num_edges(etype))[0:5000]
#     sg=dgl.edge_subgraph(G,dict_edges)
    
    subgraph_class=create_inductive_graph(G,args.train_test_split,args.seed)
    train_g, test_g=subgraph_class.subgraph_func()
    test_idx=subgraph_class.nodes_idx(train_g,test_g)
    
    assert train_g.num_nodes()+test_idx.shape[0]==test_g.num_nodes()
    
    device="cpu"
    use_cuda=args.gpu>=0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        device='cuda:%d' % args.gpu
    
    data=G, train_g, test_g, test_idx
    
    train_graph_v1, test_graph_v1=graph_run_featureless(args,usaanr_feat,device,data)    
    train_graph_v2, test_graph_v2=graph_run(args,usaanr_feat,device,data)
    train_catboost, test_catboost=catboost_run(args,device,data)
    train_mlp, test_mlp=MLP_run(args,usaanr_feat,device,data)
    
    print()
    func_print(train_catboost, train_mlp, train_graph_v1, train_graph_v2, "train_output.txt")
    print()
    func_print(test_catboost, test_mlp, test_graph_v1, test_graph_v2, "test_output.txt")