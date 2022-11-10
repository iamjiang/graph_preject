import pandas as pd
import numpy as np
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


class graph_2_df:
    def __init__(self,train_g,test_g,test_idx):
        
        self.train_g=train_g
        self.test_g=test_g
        self.test_idx=test_idx
        
    def df_creation(self):
        usaanr_feat=[]
        for key, scheme in self.train_g.node_attr_schemes(ntype="usaanr").items():
            usaanr_feat.append(key)
        usaanr_feat=[x for x in usaanr_feat if x not in 
                     ['usaanr','cmpyelig','ACTCORP','Segment','train_mask','val_mask','test_mask','label','_ID']]
             
        train_idx=th.arange(self.train_g.num_nodes()).squeeze()        
        test_idx=th.from_numpy(self.test_idx).squeeze()

        train_label=self.train_g.nodes['usaanr'].data['label'][train_idx]
        test_label=self.test_g.nodes['usaanr'].data['label'][test_idx]

        label_train=train_label.squeeze().numpy()
        label_test=test_label.squeeze().numpy()

        
        df_train=pd.DataFrame()
        for i,col in enumerate(usaanr_feat):
            ndata=self.train_g.nodes['usaanr'].data[col].squeeze().numpy()
            df_train[col]=ndata
        df_train['target_variable']=label_train
        df_train['mask']=0
        
        df_test=pd.DataFrame()
        for i,col in enumerate(usaanr_feat):
            ndata=self.test_g.nodes['usaanr'].data[col].squeeze().numpy()
            df_test[col]=ndata[self.test_idx]
        df_test['target_variable']=label_test
        df_test['mask']=1
        
        df_all=df_train.append(df_test)
        
        class_le=LabelEncoder()

        for i in tqdm(range(len(df_all.columns)),position=0,leave=True):
            col=df_all.columns[i]
            if col not in ["target_variable","mask"]:
                df_all[col]=df_all[col].astype('str')
                df_all[col]=class_le.fit_transform(df_all[col])
                df_all[col]=df_all[col].astype('str')

        df_train=df_all[df_all["mask"]==0]
        df_test=df_all[df_all["mask"]==1]
        
        train_y=df_train['target_variable'].values
        num_classes=th.unique(th.from_numpy(train_y)).shape[0]
        train_classes_num, train_classes_weight = evaluation.get_class_count_weight(train_y,num_classes)
        
        test_y=df_test['target_variable'].values
        
        df_train.drop(['target_variable', 'mask'], axis=1,inplace=True)
        df_test.drop(['target_variable', 'mask'], axis=1,inplace=True)
        categorical_index=np.where(df_train.dtypes==object)[0]
                
        return df_train, df_test, train_y, test_y, categorical_index,train_classes_weight