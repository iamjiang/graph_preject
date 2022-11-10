import copy
import pandas as pd
import math 
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import edge_subgraph
from dgl.nn.functional import edge_softmax
import dgl.nn as dglnn
import dgl.function as fn

class create_inductive_graph:
    def __init__(self,g,train_test_split,seed):
        super(create_inductive_graph,self).__init__()
        self.g=g
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
    
    def subgraph_func(self):
        
        node_train={}
        train_mask=th.zeros(self.g.num_nodes(),dtype=th.bool)
        th.manual_seed(self.seed)
        perm=th.randperm(self.g.num_nodes())
        train_idx=perm[:int((1-self.train_test_split)*self.g.num_nodes())]
        train_mask[train_idx]=True
        node_train['usaanr']=train_mask
        
        train_g=dgl.node_subgraph(self.g,node_train)
        test_g=copy.deepcopy(self.g)
        
        train_g=self.remove_isolated(train_g)
        test_g=self.remove_isolated(test_g)
        
        return train_g, test_g
    
    def nodes_idx(self,train_g,test_g):
        train_node=pd.DataFrame()
        dst_curr=[]
        dst_orig=[]
        
        for etype in train_g.etypes:
            u,v=train_g.all_edges(form="uv",etype=etype)
            u_orig=train_g.ndata[dgl.NID][u]
            v_orig=train_g.ndata[dgl.NID][v]
            curr=th.unique(th.cat((u,v)).squeeze())
            orig=th.unique(th.cat((u_orig,v_orig)).squeeze())
            dst_curr.extend(curr.squeeze().tolist())
            dst_orig.extend(orig.squeeze().tolist())
            
        train_node['v_curr']=dst_curr
        train_node['v_orig']=dst_orig
        train_node.drop_duplicates(inplace=True)
    
        test_node=pd.DataFrame()
        dst_curr=[]
        dst_orig=[]
        
        for etype in test_g.etypes:
            u,v=test_g.all_edges(form="uv",etype=etype)
            u_orig=u
            v_orig=v
            curr=th.unique(th.cat((u,v)).squeeze())
            orig=th.unique(th.cat((u_orig,v_orig)).squeeze())
            dst_curr.extend(curr.squeeze().tolist())
            dst_orig.extend(orig.squeeze().tolist())
            
        test_node['v_curr']=dst_curr
        test_node['v_orig']=dst_orig
        test_node.drop_duplicates(inplace=True)
        
        common=train_node.merge(test_node,on="v_orig",how="inner")
        
#         train_idx_map={}

        test_idx=test_node[~test_node["v_orig"].isin(common.v_orig)]["v_orig"].values
        
        return test_idx