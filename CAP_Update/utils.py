import time
import os
import pandas as pd
import numpy as np
import time
import datetime
import pickle
from tqdm import tqdm
tqdm.pandas(position=0, leave=True)
import math
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.multiprocessing as mp

import dgl
import dgl.nn as dglnn
import dgl.function as F

from _thread import start_new_thread
from functools import wraps
import traceback

def read_csv(data_dir, file):
    start=time.time()
    df=pd.read_csv(os.path.join(data_dir,file))
    df.drop_duplicates(inplace=True)
    end=time.time()
    print("Dataloading running time is {:0.4f}".format(end-start))
    print("The Shape of Dataset is {}".format(df.shape))
    return df

def to_pickle(data_dir,file_in,file_out):
    start=time.time()
    file_in.to_pickle(os.path.join(data_dir,file_out))
    end=time.time()
    print("pickle time is {:0.4f}".format(end-start))
    
def read_pickle(data_dir,file):
    start=time.time()
    df=pd.read_pickle(os.path.join(data_dir,file))
    end=time.time()
    print("loading time is {:0.4f}".format(end-start))
    print("The Shape of Dataset is {}".format(df.shape))
    return df

def node2txt(file,src,dst,rel):
    with open(file,"w") as f:
        edge_num=src.shape[0]
        for i in range(edge_num):
            f.write(src[i].item().strip()+"-->"\
                    +rel+"-->"\
                    +dst[i].item()+"\n")
def format_time(elapsed):
    #### Takes a time in seconds and returns a string hh:mm:ss
    elapsed_rounded=int(round(elapsed)) ### round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded))
def compute_pagerank(g,K,DAMP):
    N=g.number_of_nodes()
    g.ndata['pv'] = torch.ones(N) / N
    degrees = g.out_degrees(g.nodes()).type(torch.float32)
    for k in range(K):
        g.ndata['pv'] = g.ndata['pv'] / degrees
        g.update_all(message_func=F.copy_src(src='pv', out='m'),
                     reduce_func=F.sum(msg='m', out='pv'))
        g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['pv']
    return g.ndata['pv']
def graph_show(G):
    print('*'*50)
    print("Node_types: " , G.ntypes)
    print("Edge_types: " , G.etypes)
    print('*'*50)
    print("Canonical Etypes of Graph is:\n")
    for srctype, etype, dsttype in G.canonical_etypes:
        print("{:<20}{:<20}{:<20}".format(srctype, etype, dsttype))
    print('*'*50)
    Total_ntype_num=0
    for i in G.ntypes:
        print(f"number of ntype={i:<20}  {G.number_of_nodes(i):<15,}")
        Total_ntype_num+=G.number_of_nodes(i)
    print('*'*50)
    print("Total number of nodes is {:,}".format(Total_ntype_num))
    print('*'*50)
    Total_edge_num=0
    for j in G.etypes:
        print(f"number of etype={j:<20}  {G.number_of_edges(j):<15,}")
        Total_edge_num+=G.number_of_edges(j)
    print('*'*50)
    print("Total number of edges is {:,}".format(Total_edge_num))
    print('*'*50)
    for nty in G.ntypes:
        if G.nodes[nty].data!={}:
            print('*'*50)
            print(f"The attributes for the node type={nty}")
            print('*'*50)
            for key, scheme in G.node_attr_schemes(ntype=nty).items():
                print("{:<40}{}".format(key,G.nodes[nty].data[key].shape))


def prepare_mp(graph):
    graph.in_degrees(0)
    graph.out_degrees(0)
    graph.find_edges([0])
    
def fix_openmp(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

from dgl.nn import edge_softmax
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.utils import Identity

# pylint: enable=W0235
class GATConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""
        Description
        -----------
        Set allow_zero_in_degree flag.
        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        r"""
        Description
        -----------
        Compute graph attention network layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(F.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(F.u_mul_e('ft', 'a', 'm'),
                             F.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)
                
            if rst.dim()>2:
                rst=torch.mean(rst,dim=1)
                
            return rst

