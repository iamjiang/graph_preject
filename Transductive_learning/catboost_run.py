import math 
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import edge_subgraph
from dgl.nn.functional import edge_softmax
import dgl.nn as dglnn
import dgl.function as fn

import catboost
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, sum_models

from GraphSage_Model import *
from evaluation import *
from mask_function import *
from MLP_Model import *
from print_func import *

import time
import utils

def catboost_run(args,device,data):
    
    G, train_mask, test_mask=data
    graph_df_class=transductive_graph(G,args.train_test_split,args.seed)
    DF, LABEL, USAANR, train_idx, test_idx, train_classes_weight, categorical_index = graph_df_class.graph_2_df()

    train_y=LABEL[train_idx].squeeze()
    test_y=LABEL[test_idx].squeeze()
    
    df_train=DF.loc[train_idx]
    df_test=DF.loc[test_idx]
    
    usaanr_train=USAANR[train_idx]
    usaanr_test=USAANR[test_idx]

    params = {'loss_function':args.loss_function,
         'learning_rate' : args.CatBoost_LR,
         'iterations' : args.iterations,
         'cat_features' : categorical_index,
         'early_stopping_rounds'  : args.early_stopping,
         'random_seed' : args.seed,
         'task_type' : args.device_type,
         'class_weights':train_classes_weight,
         'verbose' : args.verbose}

    print()
    print("******************************************** ")
    print("========= Training For CatBoosting ========= ")
    print("******************************************** ")
    print()
    model=CatBoostClassifier(**params)
    model.fit(df_train, train_y, eval_set=(df_test,test_y),use_best_model=True,early_stopping_rounds=50)
    
    LOGIT_train = model.predict_proba(df_train)
    train_output=evaluate(train_y, LOGIT_train)
    
    LOGIT_test = model.predict_proba(df_test)
    test_output=evaluate(test_y, LOGIT_test)
    

    train_catboost=OrderedDict()
    train_catboost['model']="catboost"
    train_catboost['% test']=str(args.train_test_split*100)+"%"
    train_catboost['nb_example']=train_output['nb_example']
    train_catboost['true_prediction']=train_output['true_prediction']
    train_catboost['false_prediction']=train_output['false_prediction']
    train_catboost['accuracy']=train_output['accuracy']
    train_catboost['precision']=train_output['precision']
    train_catboost[ 'recall']=train_output[ 'recall']
    train_catboost[ 'f1_score']=train_output[ 'f1_score']
    train_catboost[ 'AUC']=train_output[ 'AUC']
    train_catboost[ 'pr_auc']=train_output[ 'pr_auc']
    
    test_catboost=OrderedDict()
    test_catboost['model']="catboost"
    test_catboost['% test']=str(args.train_test_split*100)+"%"
    test_catboost['nb_example']=test_output['nb_example']
    test_catboost['true_prediction']=test_output['true_prediction']
    test_catboost['false_prediction']=test_output['false_prediction']
    test_catboost['accuracy']=test_output['accuracy']
    test_catboost['precision']=test_output['precision']
    test_catboost[ 'recall']=test_output[ 'recall']
    test_catboost[ 'f1_score']=test_output[ 'f1_score']
    test_catboost[ 'AUC']=test_output[ 'AUC']
    test_catboost[ 'pr_auc']=test_output[ 'pr_auc']
    
    return train_catboost, test_catboost