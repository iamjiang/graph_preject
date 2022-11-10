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

import functools
import seaborn as sns
import pickle
import random

from GraphSage_Model import *

def lift_gain_eval(logit,label,topk):
    DF=pd.DataFrame(columns=["pred_score","actual_label"])
    DF["pred_score"]=logit
    DF["actual_label"]=label
    DF.sort_values(by="pred_score", ascending=False, inplace=True)
    gain={}
    for p in topk:
        N=math.ceil(int(DF.shape[0]*p))
        DF2=DF.nlargest(N,"pred_score",keep="first")
        gain[str(int(p*100))+"%"]=round(DF2.actual_label.sum()/(DF.actual_label.sum()),2)
    return gain

def get_class_count_weight(y,n_classes):
    classes_count=[]
    weight=[]
    for i in range(n_classes):
        count=np.sum(y.squeeze()==i)
        classes_count.append(count)
        weight.append(len(y)/(n_classes*count))
    return classes_count,weight

def eval_loop_func(model, loader, labels, device, loss_weight, num_classes):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    losses=[]
    for input_nodes_raw, seeds_raw, blocks in tqdm(loader, position=0, leave=True):
        
        blocks = [blk.to(device) for blk in blocks]
        seeds = seeds_raw.to(device)
        
        input_nodes={}
        input_nodes["usaanr"]=input_nodes_raw
        input_nodes={k : e.to(device) for k, e in input_nodes.items()}

        lbl = labels[seeds].squeeze().to(device)
        
        with th.no_grad():
            logits,h = model(input_nodes,blocks)
            if loss_weight is None:
                loss = F.cross_entropy(logits.view(-1, num_classes), lbl.to(device))
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes), lbl.to(device),weight=loss_weight.float())        
            losses.append(loss.item())
        fin_targets.append(lbl.cpu().detach().numpy())
        fin_outputs.append(logits.cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses

def eval_loop_MLP(model, loader, device, loss_weight, num_classes):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    losses=[]
    
    for df_batch,labels_batch in tqdm(loader, position=0, leave=True):
        
        df_batch=df_batch.to(device)
        labels_batch=labels_batch.to(device)

        with th.no_grad():
            logits = model(df_batch)
            if loss_weight is None:
                loss = F.cross_entropy(logits.view(-1, num_classes), labels_batch.squeeze().long().to(device))
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes), 
                                       labels_batch.squeeze().long().to(device),weight=loss_weight.float().to(device))        
            losses.append(loss.item())

        fin_targets.append(labels_batch.squeeze().cpu().detach().numpy())
        fin_outputs.append(logits.cpu().detach().numpy())
        
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses

def evaluate(target, predicted):
    true_label_mask=[1 if (np.argmax(x)-target[i])==0 else 0 for i,x in enumerate(predicted)]
    nb_prediction=len(true_label_mask)
    true_prediction=sum(true_label_mask)
    false_prediction=nb_prediction-true_prediction
    accuracy=true_prediction/nb_prediction
    
    precision, recall, fscore, support = precision_recall_fscore_support(target, predicted.argmax(axis=1))
    auc = roc_auc_score(target.ravel(), th.sigmoid(th.from_numpy(predicted))[:,1].numpy().ravel())
    
    prec,rec,_ = precision_recall_curve(target.ravel(), th.sigmoid(th.from_numpy(predicted))[:,1].numpy().ravel())
    
    pr_auc=auc_score(rec,prec)
    
    arg1=predicted[:,1]
    arg2=target
    gain = lift_gain_eval(arg1,arg2,topk=[0.01,0.05,0.10])
    
    return {
        "nb_example":len(target),
        "true_prediction":true_prediction,
        "false_prediction":false_prediction,
        "accuracy":accuracy,
        "precision":precision[1], 
        "recall":recall[1], 
        "f1_score":fscore[1],
        "AUC":auc,
        "pr_auc":pr_auc,
        "GAIN":gain
    }