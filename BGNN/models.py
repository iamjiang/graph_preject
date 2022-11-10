import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl import edge_subgraph
import utils
import functools

# ## USAA Members Features Embedding
# class USAANR_Embedding(nn.Module):
#     def __init__(self,G,feature_size,feat_list):
#         super(USAANR_Embedding,self).__init__()
#         self.device=G.device
#         self.G=G
#         self.feature_size=feature_size
#         self.feat_list=feat_list
#         ## Embedding matrices for features of nodes.
#         self.emb = nn.ModuleDict()
        
#         for i,col in enumerate(self.feat_list):
#             self.emb[col]=nn.Embedding(G.ndata[col].max().item()+1, feature_size)
    
#     def forward(self, nid):
#         nid=nid.to(self.device)
#         extra_repr=[]
        
#         for i,col in enumerate(self.feat_list):
#             ndata=self.G.ndata[col]
#             extra_repr.append(self.emb[col](ndata[nid]).squeeze(1))
#         return th.stack(extra_repr, 0).sum(0)
    
class RelGraphConv(nn.Module):

    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer="basis",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 low_mem=False,
                 dropout=0.0,
                 layer_norm=False):
        super(RelGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases <= 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm

        if regularizer == "basis":
            # add basis weights
            self.weight = nn.Parameter(th.Tensor(self.num_bases, self.in_feat, self.out_feat))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(th.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.basis_message_func
        else:
            raise ValueError("Regularizer must be 'basis' ")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def basis_message_func(self, edges, etypes):
        """Message function for basis regularizer.
        Parameters
        ----------
        edges : dgl.EdgeBatch
            Input to DGL message UDF.
        etypes : torch.Tensor or list[int]
            Edge type data. Could be either:
                * An :math:`(|E|,)` dense tensor. Each element corresponds to the edge's type ID.
                  Preferred format if ``lowmem == False``.
                * An integer list. The i^th element is the number of edges of the i^th type.
                  This requires the input graph to store edges sorted by their type IDs.
                  Preferred format if ``lowmem == True``.
        """
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = th.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        h = edges.src['h']
        device = h.device

        if "e_weights" in edges.data.keys():
            h=h*edges.data['e_weights']
            
        if self.low_mem:
            # A more memory-friendly implementation.
            # Calculate msg @ W_r before put msg into edge.
            assert isinstance(etypes, list)
            h_t = th.split(h, etypes)
            msg = []
            for etype in range(self.num_rels):
                if h_t[etype].shape[0] == 0:
                    continue
                msg.append(th.matmul(h_t[etype], weight[etype]))
            msg = th.cat(msg)
        else:
            # Use batched matmult
            weight = weight.index_select(0, etypes)
            msg = th.bmm(h.unsqueeze(1), weight).squeeze(1)

        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}
    
    def forward(self, g, feat, etypes, norm=None):
        """Forward computation.
        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            Input node features. Could be either
                * :math:`(|V|, D)` dense tensor
                * :math:`(|V|,)` int64 vector, representing the categorical values of each
                  node. It then treat the input feature as an one-hot encoding feature.
        etypes : torch.Tensor or list[int]
            Edge type data. Could be either
                * An :math:`(|E|,)` dense tensor. Each element corresponds to the edge's type ID.
                  Preferred format if ``lowmem == False``.
                * An integer list. The i^th element is the number of edges of the i^th type.
                  This requires the input graph to store edges sorted by their type IDs.
                  Preferred format if ``lowmem == True``.
        norm : torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`.
        Returns
        -------
        torch.Tensor
            New node features.
        Notes
        -----
        Under the ``low_mem`` mode, DGL will sort the graph based on the edge types
        and compute message passing one type at a time. DGL recommends sorts the
        graph beforehand (and cache it if possible) and provides the integer list
        format to the ``etypes`` argument. Use DGL's :func:`~dgl.to_homogeneous` API
        to get a sorted homogeneous graph from a heterogeneous graph. Pass ``return_count=True``
        to it to get the ``etypes`` in integer list.
        """
        if isinstance(etypes, th.Tensor):
            if len(etypes) != g.num_edges():
                raise DGLError('"etypes" tensor must have length equal to the number of edges'
                               ' in the graph. But got {} and {}.'.format(
                                   len(etypes), g.num_edges()))
            if self.low_mem and not (feat.dtype == th.int64 and feat.ndim == 1):
                # Low-mem optimization is not enabled for node ID input. When enabled,
                # it first sorts the graph based on the edge types (the sorting will not
                # change the node IDs). It then converts the etypes tensor to an integer
                # list, where each element is the number of edges of the type.
                # Sort the graph based on the etypes
                sorted_etypes, index = th.sort(etypes)
                g = edge_subgraph(g, index, preserve_nodes=True)
                # Create a new etypes to be an integer list of number of edges.
                pos = th.searchsorted(sorted_etypes, th.arange(self.num_rels, device=g.device))
                num = th.tensor([len(etypes)], device=g.device)
                etypes = (th.cat([pos[1:], num]) - pos).tolist()
                if norm is not None:
                    norm = norm[index]

        with g.local_scope():
            g.srcdata['h'] = feat
            if norm is not None:
                g.edata['norm'] = norm
            if self.self_loop:
                loop_message = utils.matmul_maybe_select(feat[:g.number_of_dst_nodes()],
                                                         self.loop_weight)
            # message passing
            g.update_all(functools.partial(self.message_func, etypes=etypes),
                             fn.sum(msg='msg', out='h'))
                
            # apply bias and activation
            node_repr = g.dstdata['h']
            if self.layer_norm:
                node_repr = self.layer_norm_weight(node_repr)
            if self.bias:
                node_repr = node_repr + self.h_bias
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self.dropout(node_repr)
            
            return node_repr
        
class EntityClassify(nn.Module):

    def __init__(self,
                 g,
                 h_dim,
                 out_dim,
                 num_rels,
                 num_bases=None,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 low_mem=True,
                 layer_norm=False):
        super(EntityClassify, self).__init__()
        self.g=g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm
        
#         self.node_embed = USAANR_Embedding(self.g,self.h_dim,self.feat_list)
        
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(
            self.h_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            low_mem=self.low_mem, dropout=self.dropout, layer_norm = layer_norm))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConv(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                low_mem=self.low_mem, dropout=self.dropout, layer_norm = layer_norm))
        # h2o
#         self.layers.append(RelGraphConv(
#             self.h_dim, self.out_dim, self.num_rels, "basis",
#             self.num_bases, activation=None,
#             self_loop=self.use_self_loop,
#             low_mem=self.low_mem, layer_norm = layer_norm))

        self.classifier = nn.Linear(self.h_dim, self.out_dim)

    def forward(self, blocks, input_nodes):
        
#         H=self.node_embed(input_nodes)
#         H=blocks[0].srcnodes[blocks.ntypes[0]].data['node_features']
        H=blocks[0].srcdata['node_features']
    
        if not isinstance(blocks,list):
            # full graph training
            for layer in self.layers:
                H = layer(blocks, H, blocks.edata['etype'], blocks.edata['norm'])
                    
                
        else:
            for layer, block in zip(self.layers, blocks):
                block = block
                H = layer(block, H, block.edata['etype'], block.edata['norm'])
        
        logits = self.classifier(H)
        
        return logits, H

