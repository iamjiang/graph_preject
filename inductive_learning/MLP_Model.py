import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl import edge_subgraph
from dgl.nn.functional import edge_softmax
import dgl.nn as dglnn
import dgl.function as fn
from sklearn.preprocessing import LabelEncoder

class Batch_Dataset(Dataset):
    def __init__(self,df,label):
        self.x=df.values
        # ensure input data is floats
        self.x = self.x.astype('int')
        self.y=label
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
        
    # number of rows in the dataset
    def __len__(self):
        return len(self.x)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]

##### USAA Members Features Embedding
class USAANR_Embedding_MLP(nn.Module):
    def __init__(self,DF,feature_size,feat_list,device):
        super(USAANR_Embedding_MLP,self).__init__()
        self.DF=DF
        self.feature_size=feature_size
        self.feat_list=feat_list
        self.device=device
        
        ## Embedding matrices for features of nodes.
        self.emb = nn.ModuleDict()
        
        for i,col in enumerate(self.feat_list):
            self.emb[col]=nn.Embedding(len(self.DF[col].unique())+1, self.feature_size)
        
    def forward(self,X):
        extra_repr=[]
        for i,col in enumerate(self.feat_list):
            ndata=X[:,i].squeeze().to(self.device)
            extra_repr.append(self.emb[col].to(self.device)(ndata).squeeze())
        return th.stack(extra_repr, 0).sum(0)
    
class MLP(nn.Module):
    def __init__(self,
                 DF,
                 h_dim,
                 out_dim,
                 n_layers,
                 device,
                 feat_list,
                 dropout=0):
        super(MLP, self).__init__()
        self.df=DF
        self.h_dim=h_dim
        self.out_dim=out_dim
        self.n_layers=n_layers
        self.device=device
        self.feat_list=feat_list
        self.dropout=nn.Dropout(dropout)
        self.usaanr_node_embed=USAANR_Embedding_MLP(self.df,self.h_dim,self.feat_list,self.device)

        self.layers = nn.ModuleList()
        
        if self.n_layers>1:
            self.hidden1=nn.Sequential(nn.Linear(self.h_dim,self.h_dim),nn.ReLU())
            self.hidden1.apply(self.init_weights)
            self.layers.append(self.hidden1)
            for i in range(1,self.n_layers-1):
                self.hidden_i=nn.Sequential(nn.Linear(self.h_dim,self.h_dim),nn.ReLU())
                self.hidden_i.apply(self.init_weights)
                self.layers.append(self.hidden_i)
            
            self.hidden_f=nn.Linear(self.h_dim,self.out_dim)
            nn.init.xavier_uniform_(self.hidden_f.weight, gain=nn.init.calculate_gain('relu'))
            self.layers.append(self.hidden_f)
        else:
            self.hidden_f=nn.Linear(self.h_dim,self.out_dim)
            nn.init.xavier_uniform_(self.hidden_f.weight, gain=nn.init.calculate_gain('relu'))
            self.layers.append(self.hidden_f)
            
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            th.nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.fill_(0.01)
        
    def forward(self, X):
        H=self.usaanr_node_embed(X)
        for l,layer in enumerate(self.layers):
            H=layer(H)
            if l !=len(self.layers)-1:
                H=self.dropout(H)
#         logits = (H - H.mean(0, keepdims=True)) / H.std(0, keepdims=True)
        
        return H
    
