import math 
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import edge_subgraph
from dgl.nn.functional import edge_softmax
import dgl.nn as dglnn
import dgl.function as fn

class FeatureLess_Embedding(nn.Module):
    def __init__(self,g,embed_size,device,dropout=0.0):
        super(FeatureLess_Embedding,self).__init__()
        self.device=device
        self.G=g
        self.embed_size=embed_size
        self.dropout=nn.Dropout(dropout)
        ## Embedding matrices for features of nodes.
        
        self.emb = nn.ModuleDict()
        for ntype in self.G.ntypes:
            embed=nn.Embedding(self.G.num_nodes(ntype)+1, self.embed_size)
            self.emb[ntype]=embed
        
#         self.emb = nn.ParameterDict()
#         for ntype in self.G.ntypes:
#             embed=nn.Parameter(th.FloatTensor(self.G.num_nodes(ntype)+1, self.embed_size))
# #             initrange=1.0/self.embed_size
# #             nn.init.uniform(embed,-initrange,initrange)
#             nn.init.xavier_uniform_(embed,gain=nn.init.calculate_gain('relu'))
#             self.emb[ntype]=embed
    
    def forward(self,sg, nid):
#         nid=nid.to("cpu")
#         self.emb=self.emb.to("cpu")
        idx=sg.nodes['usaanr'].data[dgl.NID][nid].squeeze().to(self.device)
        out_feature=self.emb['usaanr'](idx).squeeze().to(self.device)
        
        return out_feature

## USAA Members Features Embedding
class USAANR_Embedding(nn.Module):
    def __init__(self,g,feature_size,feat_list,device):
        super(USAANR_Embedding,self).__init__()
        self.device=device
        self.G=g
        self.feature_size=feature_size
        self.feat_list=feat_list
        ## Embedding matrices for features of nodes.
        self.emb = nn.ModuleDict()
        
        for i,col in enumerate(self.feat_list):
            self.emb[col]=nn.Embedding(self.G.nodes['usaanr'].data[col].max().item()+1, self.feature_size)
    
    def forward(self,sg, nid):
        nid=nid.to(self.device)
        extra_repr=[]
        for i,col in enumerate(self.feat_list):
            ndata=sg.nodes['usaanr'].data[col].to(self.device)
            extra_repr.append(self.emb[col].to(self.device)(ndata[nid]).squeeze())
        return th.stack(extra_repr, 0).sum(0)


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.conv = dglnn.HeteroGraphConv({
#                 rel : dglnn.GraphConv(in_feat, out_feat, norm="both", weight=False, bias=False)
                rel : dglnn.SAGEConv(in_feat, out_feat, aggregator_type='mean',feat_drop=0.,bias=True,norm=None)
                for rel in rel_names
            })
        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)
        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(dropout)
    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i] : {'weight' : w.squeeze(0)}
                     for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}
        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs
        hs = self.conv(g, inputs)
        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}
    

class Entity_Classify(nn.Module):
    def __init__(self,
                 g,
                 feat_list,
                 device,
                 h_dim,
                 out_dim,
                 num_bases,
#                  embed_layer,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(Entity_Classify, self).__init__()
        self.g = g
        self.feat_list=feat_list
        self.device=device
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
#         self.num_bases = None if num_bases < 0 else num_bases
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
            
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        
#         self.node_embed={}
        self.node_embed=nn.ModuleDict()
        self.node_embed['usaanr'] = USAANR_Embedding(self.g,self.h_dim,self.feat_list,self.device)
#         self.node_embed['zipcode'] = Zipcode_Embedding(self.g,self.h_dim)
        self.layers = nn.ModuleList()
        #i2h
        self.layers.append(RelGraphConvLayer(
                    self.h_dim, self.h_dim, self.rel_names,
                    self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                    dropout=self.dropout, weight=True))
        # h2h
        if self.num_hidden_layers>1:
            for i in range(0,self.num_hidden_layers-1):
                self.layers.append(RelGraphConvLayer(
                    self.h_dim, self.h_dim, self.rel_names,
                    self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                    dropout=self.dropout))
        # h2o
#         self.layers.append(RelGraphConvLayer(
#             self.h_dim, self.out_dim, self.rel_names, 
#             self.num_bases, activation=partial(F.softmax, dim=1),
#             self_loop=self.use_self_loop))
        self.classifier = nn.Linear(self.h_dim, self.out_dim)
    
    def forward(self, sg, input_nodes, blocks=None):
        H={}
        for ntype, nid in input_nodes.items():
            nid = input_nodes[ntype]
            H[ntype] = self.node_embed[ntype](sg,nid)
        if blocks is None:
            for layer in self.layers:
                H = layer(sg, H)
        else:
            for layer, block in zip(self.layers, blocks):
                H = layer(block, H)
                
#         h=(H["usaanr"] - H["usaanr"].mean(0, keepdims=True)) / H["usaanr"].std(0, keepdims=True)
        
        output = self.classifier(H["usaanr"])
    
        return output, H
    

class Entity_Classify_FeatureLess(nn.Module):
    def __init__(self,
                 g,
                 device,
                 h_dim,
                 out_dim,
                 num_bases,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(Entity_Classify_FeatureLess, self).__init__()
        self.g = g
        self.device=device
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
#         self.num_bases = None if num_bases < 0 else num_bases
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
            
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        
#         self.node_embed={}
        self.node_embed=nn.ModuleDict()
        self.node_embed['usaanr'] = FeatureLess_Embedding(self.g,self.h_dim,self.device,self.dropout)
#         self.node_embed['zipcode'] = Zipcode_Embedding(self.g,self.h_dim)
        self.layers = nn.ModuleList()
        #i2h
        self.layers.append(RelGraphConvLayer(
                    self.h_dim, self.h_dim, self.rel_names,
                    self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                    dropout=self.dropout, weight=True))
        # h2h
        if self.num_hidden_layers>1:
            for i in range(0,self.num_hidden_layers-1):
                self.layers.append(RelGraphConvLayer(
                    self.h_dim, self.h_dim, self.rel_names,
                    self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                    dropout=self.dropout))
        # h2o
#         self.layers.append(RelGraphConvLayer(
#             self.h_dim, self.out_dim, self.rel_names, 
#             self.num_bases, activation=partial(F.softmax, dim=1),
#             self_loop=self.use_self_loop))
        self.classifier = nn.Linear(self.h_dim, self.out_dim)
    
    def forward(self, sg, input_nodes, blocks=None):
        H={}
        for ntype, nid in input_nodes.items():
            nid = input_nodes[ntype]
            H[ntype] = self.node_embed[ntype](sg,nid)
        if blocks is None:
            for layer in self.layers:
                H = layer(self.g, H)
        else:
            for layer, block in zip(self.layers, blocks):
                H = layer(block, H)
                
#         h=(H["usaanr"] - H["usaanr"].mean(0, keepdims=True)) / H["usaanr"].std(0, keepdims=True)
        
        output = self.classifier(H["usaanr"])
    
        return output, H