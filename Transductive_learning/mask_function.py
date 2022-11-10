import copy
import pandas as pd
import numpy as np
import math 
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import edge_subgraph
from dgl.nn.functional import edge_softmax
import dgl.nn as dglnn
import dgl.function as fn

from sklearn.preprocessing import LabelEncoder, label_binarize
from tqdm import tqdm
import evaluation 

class transductive_graph:
    def __init__(self,g,train_test_split,seed):
        super(transductive_graph,self).__init__()
        self.g=g
        self.label=self.g.nodes['usaanr'].data['label']
        self.train_test_split=train_test_split
        self.seed=seed
        
    def remove_isolated(self,g):
        connected_nodes=[]
        for etype in g.etypes:
            idx=((g.in_degrees(etype=etype) !=0) | (g.out_degrees(etype=etype)!=0)).nonzero().squeeze().tolist()
            connected_nodes.extend(idx)
        connected_nodes=set(connected_nodes)
        all_nodes=set(th.arange(g.num_nodes()).tolist())
        isolated_nodes=all_nodes-connected_nodes
        if len(isolated_nodes)>0:
            g.remove_nodes(th.tensor(list(isolated_nodes)))

        return g
    
    def train_test_mask_func(self):
        
        self.g=self.remove_isolated(self.g)
        LABEL=self.g.nodes['usaanr'].data['label'].numpy().squeeze()
        
        train_idx=[]
        test_idx=[]
        
        IDX=np.arange(LABEL.shape[0])
        prod_list=np.unique(LABEL).tolist()
        for i in range(len(prod_list)):
            _idx=IDX[LABEL==prod_list[i]]
            np.random.seed(self.seed)
            np.random.shuffle(_idx)
            train_idx.extend(_idx[:int((1-self.train_test_split)*len(_idx))])
            test_idx.extend(_idx[int((1-self.train_test_split)*len(_idx)):])
            
        train_idx=th.LongTensor(train_idx).squeeze()
        test_idx=th.LongTensor(test_idx).squeeze()
        
        train_mask=th.zeros(self.g.num_nodes(),dtype=th.bool)
        test_mask=th.zeros(self.g.num_nodes(),dtype=th.bool)
        
        train_mask[train_idx]=True
        test_mask[test_idx]=True
        
        g=self.g
        
        return g, train_mask, test_mask
    
    def graph_2_df(self):
        
        usaanr_feat=[]
        for key, scheme in self.g.node_attr_schemes(ntype="usaanr").items():
            usaanr_feat.append(key)
            usaanr_feat=[x for x in usaanr_feat if x not in \
                         ['cmpyelig','ACTCORP','Segment','train_mask','val_mask','test_mask','label','_ID']]
            
        g, train_mask, test_mask=self.train_test_mask_func()
        LABEL=g.nodes['usaanr'].data['label'].numpy().squeeze()
        
        train_idx=th.nonzero(train_mask).squeeze().numpy()
        test_idx=th.nonzero(test_mask).squeeze().numpy()     
        
        DF=pd.DataFrame()
        for i,col in enumerate(usaanr_feat):
            ndata=g.nodes['usaanr'].data[col].squeeze().numpy()
            DF[col]=ndata

        DF['target_variable']=LABEL
        
        class_le=LabelEncoder()

        for i in tqdm(range(len(DF.columns)),position=0,leave=True):
            col=DF.columns[i]
            if col not in ["target_variable"]:
                DF[col]=DF[col].astype('str')
                DF[col]=class_le.fit_transform(DF[col])
                DF[col]=DF[col].astype('str')
       
    
        train_y=DF.loc[train_idx,['target_variable']].values
        num_classes=th.unique(th.from_numpy(train_y)).shape[0]
        train_classes_num, train_classes_weight = evaluation.get_class_count_weight(train_y,num_classes)
        
        USAANR=DF.loc[:,['usaanr']].values.astype(int)
        
        train_y=DF.loc[train_idx,['target_variable']].values  
        num_classes=th.unique(th.from_numpy(train_y)).shape[0]
        train_classes_num, train_classes_weight = evaluation.get_class_count_weight(train_y,num_classes)
        
        DF.drop(['usaanr','target_variable'], axis=1,inplace=True)
        categorical_index=np.where(DF.dtypes==object)[0]
                
        return DF, LABEL, USAANR, train_idx, test_idx, train_classes_weight, categorical_index