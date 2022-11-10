import math
import itertools
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl import edge_subgraph
import dgl.nn as dglnn
import dgl.function as fn

from catboost import Pool, CatBoostClassifier, CatBoostRegressor, sum_models
from lightgbm import LGBMClassifier, LGBMRegressor
from tqdm import tqdm
from collections import defaultdict as ddict
import pandas as pd
from sklearn import preprocessing
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder, label_binarize

class BGNNPredictor:
    '''
    Description
    -----------
    Boost GNN predictor for semi-supervised node classification or regression problems.
    Publication: https://arxiv.org/abs/2101.08543
    Parameters
    ----------
    gnn_model : nn.Module
        DGL implementation of GNN model.
    task: str, optional
        Regression or classification task.
    loss_fn : callable, optional
        Function that takes torch tensors, pred and true, and returns a scalar.
    trees_per_epoch : int, optional
        Number of GBDT trees to build each epoch.
    backprop_per_epoch : int, optional
        Number of backpropagation steps to make each epoch.
    lr : float, optional
        Learning rate of gradient descent optimizer.
    append_gbdt_pred : bool, optional
        Append GBDT predictions or replace original input node features.
    train_input_features : bool, optional
        Train original input node features.
    gbdt_depth : int, optional
        Depth of each tree in GBDT model.
    gbdt_lr : float, optional
        Learning rate of GBDT model.
    gbdt_alpha : int, optional
        Weight to combine previous and new GBDT trees.
    random_seed : int, optional
        random seed for GNN and GBDT models.
    Examples
    ----------
    gnn_model = GAT(10, 20, num_heads=5),
    bgnn = BGNNPredictor(gnn_model)
    metrics = bgnn.fit(graph, X, y, train_mask, val_mask, test_mask, cat_features)
    '''
    def __init__(self,
                 gnn_model,
                 device,
                 task = 'classification',
                 loss_fn = None,
                 trees_per_epoch = 10,
                 backprop_per_epoch = 10,
                 lr=0.01,
                 append_gbdt_pred = True,
                 train_input_features = True,
                 gbdt_depth=6,
                 gbdt_lr=0.1,
                 gbdt_alpha = 1,
                 random_seed = 0
                 ):
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.model = gnn_model.to(self.device)
        self.task = task
        self.loss_fn = loss_fn
        self.trees_per_epoch = trees_per_epoch
        self.backprop_per_epoch = backprop_per_epoch
        self.lr = lr
        self.append_gbdt_pred = append_gbdt_pred
        self.train_input_features = train_input_features
        self.gbdt_depth = gbdt_depth
        self.gbdt_lr = gbdt_lr
        self.gbdt_alpha = gbdt_alpha
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def init_gbdt_model(self, num_epochs, epoch):
        if self.task == 'regression':
            lightgbm_model_obj = LGBMRegressor
            METRIC = {'rmse'}
        else:
            if epoch == 0: # we predict multiclass probs at first epoch
                lightgbm_model_obj = LGBMClassifier
                METRIC = {'multiclass'}
            else: # we predict the gradients for each class at epochs > 0
                lightgbm_model_obj = LGBMRegressor
                METRIC = {'rmse'}

                
        return lightgbm_model_obj(boosting="gbdt",
                                  num_iterations=num_epochs,
                                  max_bin=255,
                                  max_depth=self.gbdt_depth,
                                  learning_rate=self.gbdt_lr,
                                  random_state=self.random_seed,
                                  device_type="gpu",
                                  metric=METRIC)

    def fit_gbdt(self, gbdt_X_train, gbdt_y_train, trees_per_epoch, epoch):
        gbdt_model = self.init_gbdt_model(trees_per_epoch, epoch)
        gbdt_model.fit(gbdt_X_train, gbdt_y_train)
        return gbdt_model

    def append_gbdt_model(self, new_gbdt_model, weights):
        if self.gbdt_model is None:
            return new_gbdt_model
#         return sum_models([self.gbdt_model, new_gbdt_model], weights=weights)
        return self.gbdt_model*weights[0]+new_gbdt_model*weights[1]

    def train_gbdt(self, gbdt_X_train, gbdt_y_train, cat_features, epoch,
                   gbdt_trees_per_epoch, gbdt_alpha):
        
        epoch_gbdt_model = self.fit_gbdt(gbdt_X_train, gbdt_y_train, gbdt_trees_per_epoch, epoch)
        if epoch == 0 and self.task=='classification':
            self.base_gbdt = epoch_gbdt_model
        else:
            self.gbdt_model = self.append_gbdt_model(epoch_gbdt_model, weights=[1, gbdt_alpha])

    def update_node_features(self, node_features, X, original_X):
        # get predictions from gbdt model
        if self.task == 'regression':
            predictions = np.expand_dims(self.gbdt_model.predict(original_X), axis=1)
        else:
            predictions = self.base_gbdt.predict_proba(original_X)
            if self.gbdt_model is not None:
                predictions_after_one = self.gbdt_model.predict(original_X)
                predictions += predictions_after_one

        # update node features with predictions
        if self.append_gbdt_pred:
            if self.train_input_features:
                predictions = np.append(node_features.weight.detach().cpu().data[:, :-self.out_dim],
                                        predictions,
                                        axis=1)  # replace old predictions with new predictions
            else:
                predictions = np.append(X, predictions, axis=1)  # append original features with new predictions

        predictions = torch.from_numpy(predictions).to(self.device)

        node_features.data = predictions.float().data

    def update_gbdt_targets(self, node_features, node_features_before, train_mask):
        return (node_features.weight.data - node_features_before).detach().cpu().numpy()[train_mask, -self.out_dim:]

    def init_node_features(self, X):
#         node_features = torch.empty(X.shape[0], self.in_dim, requires_grad=True, device=self.device)
        node_features = nn.Embedding(X.shape[0], self.in_dim, sparse=True)

        if self.append_gbdt_pred:
            node_features.weight.data[:, :-self.out_dim] = torch.from_numpy(X.to_numpy(copy=True))
        return node_features

    def init_optimizer(self, node_features, optimize_node_features, learning_rate):

        params = [self.model.parameters()]
        if optimize_node_features:
            params.append([node_features])
#         optimizer = torch.optim.Adam(itertools.chain(*params), lr=learning_rate)
        optimizer_1 = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        optimizer_2 = torch.optim.SparseAdam(node_features.parameters(), lr=learning_rate)
        return optimizer_1, optimizer_2

    def train_model(self, input_nodes, seeds, blocks, labels, device, optimizer,node_features):
        optimizer_1, optimizer_2= optimizer
        
        for blk in blocks:
            blk.srcdata['node_features']=node_features(blk.srcdata[dgl.NID])
            
        self.model.train()
        blocks = [blk.to(device) for blk in blocks]
        seeds = seeds.to(device)
        input_nodes=input_nodes.to(device)
#         input_nodes={}
#         input_nodes["usaanr"]=input_nodes_raw
#         input_nodes={k : e.to(device) for k, e in input_nodes.items()}
        
        logits,h = self.model(blocks,input_nodes)
        
        pred = logits.squeeze()
        
        labels_train=labels[seeds].to(device)  

        if self.loss_fn is not None:
            loss = self.loss_fn(pred, labels_train)
        else:
            loss = F.cross_entropy(pred, labels_train.long())

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        loss.backward()
        optimizer_1.step()
        optimizer_2.step()
        
#         print([p.grad.norm() for p in optimizer_2.param_groups[0]['params']])
        
        return loss
    
    def train_and_evaluate(self, loader,labels, device, optimizer, gnn_passes_per_epoch, node_features):
        
        loss = None

        for _ in range(gnn_passes_per_epoch):
            for step, (input_nodes, seeds, blocks) in enumerate(loader):
                loss = self.train_model(input_nodes, seeds, blocks, labels, device, optimizer,node_features)


        self.model.eval()
        
        train_metrics = self.evaluate_model(loader, labels, device, node_features)

        return train_metrics
    
    
    def get_class_count_weight(self,y,n_classes):
        classes_count=[]
        weight=[]
        for i in range(n_classes):
            count=np.sum(y.squeeze()==i)
            classes_count.append(count)
            weight.append(len(y)/(n_classes*count))
        return classes_count,weight    
    
    def lift_gain_eval(self,logit,label,topk):
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

    
    def evaluate_model(self, loader, labels, device, node_features):
        self.model.eval()
        fin_targets=[]
        fin_outputs=[]
        losses=[]
        for input_nodes, seeds, blocks in tqdm(loader, position=0, leave=True):
            for blk in blocks:
                blk.srcdata['node_features']=node_features(blk.srcdata[dgl.NID])
                
            blocks = [blk.to(device) for blk in blocks]
            seeds = seeds.to(device)
            input_nodes=input_nodes.to(device)

#             input_nodes={}
#             input_nodes["usaanr"]=input_nodes_raw
#             input_nodes={k : e.to(device) for k, e in input_nodes.items()}

            lbl = labels[seeds].squeeze().long().to(device)

            with torch.no_grad():
                logits,h = self.model(blocks,input_nodes)

                loss = F.cross_entropy(logits, lbl.to(device))
     
                losses.append(loss.item())
        
            fin_targets.append(lbl.cpu().detach().numpy())
            fin_outputs.append(logits.cpu().detach().numpy())
            
        predicted=np.concatenate(fin_outputs).squeeze() 
        target=np.concatenate(fin_targets).squeeze()
        
        with torch.no_grad():
            true_label_mask=[1 if (np.argmax(x)-target[i])==0 else 0 for i,x in enumerate(predicted)]
            nb_prediction=len(true_label_mask)
            true_prediction=sum(true_label_mask)
            false_prediction=nb_prediction-true_prediction
            accuracy=true_prediction/nb_prediction

            precision, recall, fscore, support = precision_recall_fscore_support(target, predicted.argmax(axis=1))
            auc = roc_auc_score(target.ravel(), torch.sigmoid(torch.from_numpy(predicted))[:,1].numpy().ravel())

            prec,rec,_ = precision_recall_curve(target.ravel(), torch.sigmoid(torch.from_numpy(predicted))[:,1].numpy().ravel())

            pr_auc=auc_score(rec,prec)

            arg1=predicted[:,1]
            arg2=target
            gain = self.lift_gain_eval(arg1,arg2,topk=[0.01,0.05,0.10])
            
            metrics={}
            metrics["loss"]=np.mean(losses[-10:])
            metrics["nb_example"]=len(target)
            metrics["true_prediction"]=true_prediction
            metrics["false_prediction"]=false_prediction
            metrics["accuracy"]=accuracy
            metrics["precision"]=precision[1]
            metrics["recall"]=recall[1]
            metrics["recall"]=fscore[1]
            metrics["AUC"]=auc
            metrics["pr_auc"]=pr_auc
            metrics["GAIN"]=gain
            
            return metrics


    def update_early_stopping(self, val_metric, epoch, best_metric, epochs_since_last_best_metric, metric_name,
                              lower_better=False):
        
        if (lower_better and val_metric[metric_name] < best_metric) or \
        (not lower_better and val_metric[metric_name] > best_metric):
            best_metrics = val_metric[metric_name]
            best_val_epoch = epoch
            epochs_since_last_best_metric = 0
        else:
            epochs_since_last_best_metric += 1
        return best_metrics, best_val_epoch, epochs_since_last_best_metric

 
    def fit(self, 
            graph, 
            X, 
            y,
            train_mask, 
            val_mask, 
            test_mask,
            original_X = None,
            cat_features = None,
            num_epochs=100,
            patience=10,
            metric_name='loss',
            fanout=None,
            num_layers=1,
            batch_size=1024
            ):
        '''
        :param graph : dgl.DGLGraph
            Input graph
        :param X : pd.DataFrame
            Input node features. Each column represents one input feature. Each row is a node.
            Values in dataframe are numerical, after preprocessing.
        :param y : pd.DataFrame
            Input node targets. Each column represents one target. Each row is a node
            (order of nodes should be the same as in X).
        :param train_mask : list[int]
            Node indexes (rows) that belong to train set.
        :param val_mask : list[int]
            Node indexes (rows) that belong to validation set.
        :param test_mask : list[int]
            Node indexes (rows) that belong to test set.
        :param original_X : pd.DataFrame, optional
            Input node features before preprocessing. Each column represents one input feature. Each row is a node.
            Values in dataframe can be of any type, including categorical (e.g. string, bool) or
            missing values (None). This is useful if you want to preprocess X with GBDT model.
        :param cat_features: list[int]
            Feature indexes (columns) which are categorical features.
        :param num_epochs : int
            Number of epochs to run.
        :param patience : int
            Number of epochs to wait until early stopping.
        :param metric_name : str
            Metric to use for early stopping.
        :param normalize_features : bool
            If to normalize original input features X (column wise).
        :param replace_na: bool
            If to replace missing values (None) in X.
        :return: metrics evaluated during training
        '''

        # initialize for early stopping and metrics
        if metric_name in ['r2', 'accuracy']:
            best_metric = np.float('-inf') # for train/val/test
        else:
            best_metric = np.float('inf')  # for train/val/test

        best_val_epoch = 0
        epochs_since_last_best_metric = 0
#         metrics = ddict(list)
        if cat_features is None:
            cat_features = []

        self.out_dim = np.unique(y.values).shape[0]
        self.in_dim = self.out_dim + X.shape[1] if self.append_gbdt_pred else self.out_dim

        if original_X is None:
            original_X = X.copy()
            cat_features = []

        gbdt_X_train = original_X.iloc[train_mask]
        gbdt_y_train = y.iloc[train_mask]
        gbdt_alpha = self.gbdt_alpha
        self.gbdt_model = None

        node_features = self.init_node_features(X)
        optimizer= self.init_optimizer(node_features, optimize_node_features=True, learning_rate=self.lr)
        
        y = torch.from_numpy(y.to_numpy(copy=True)).float().squeeze()

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start2epoch = time.time()

            # gbdt part
            self.train_gbdt(gbdt_X_train, gbdt_y_train, cat_features, epoch,
                            self.trees_per_epoch, gbdt_alpha)

            self.update_node_features(node_features, X, original_X)
            node_features_before = node_features.weight.data.clone()
            
#             graph.ndata['node_features']=node_features.weight.data
            sampler=dgl.dataloading.MultiLayerNeighborSampler([fanout]*num_layers)
            train_loader=dgl.dataloading.NodeDataLoader(graph.to("cpu"),torch.tensor(train_mask),\
                                                        sampler,batch_size=batch_size, shuffle=True)
            
            train_metric = self.train_and_evaluate(train_loader, y, self.device, optimizer, self.backprop_per_epoch,node_features)
            gbdt_y_train = self.update_gbdt_targets(node_features, node_features_before, train_mask)

            val_loader=dgl.dataloading.NodeDataLoader(graph.to("cpu"),torch.tensor(val_mask),\
                                                        sampler,batch_size=batch_size, shuffle=True)        
            val_metric = self.train_and_evaluate(train_loader, y, self.device, optimizer, self.backprop_per_epoch,node_features)
            
            test_loader=dgl.dataloading.NodeDataLoader(graph.to("cpu"),torch.tensor(test_mask),\
                                                        sampler,batch_size=batch_size, shuffle=True)        
            test_metric = self.train_and_evaluate(test_loader, y, self.device, optimizer, self.backprop_per_epoch,node_features)
            
#             self.log_epoch(pbar, metrics, epoch, loss, time.time() - start2epoch, logging_epochs,
#                            metric_name=metric_name)

            # check early stopping
            best_metrics, best_val_epoch, epochs_since_last_best_metric = \
                self.update_early_stopping(val_metric, epoch, best_metric, epochs_since_last_best_metric, metric_name, \
                              lower_better=True)
   
            if patience and epochs_since_last_best_metric > patience:
                break

            if np.isclose(gbdt_y_train.sum(), 0.):
                print('Node embeddings do not change anymore. Stopping...')
                break

        print('Best {} at iteration {}: {:.3f}'.format(metric_name, best_val_epoch, best_metrics))
        return train_metric, val_metric, test_metric


    def predict(self, graph, X, test_mask,fanout,num_layers,batch_size):
        graph = graph.to(self.device)
        node_features = torch.empty(X.shape[0], self.in_dim).to(self.device)
        self.update_node_features(node_features, X, X)
        graph.ndata['node_features']=node_features
        
        sampler=dgl.dataloading.MultiLayerNeighborSampler([fanout]*num_layers)
        test_loader=dgl.dataloading.NodeDataLoader(graph.to("cpu"),torch.tensor(test_mask),\
                                                    sampler,batch_size=batch_size, shuffle=True)  

        self.model.eval()
        fin_outputs=[]
        
        for input_nodes, seeds, blocks in tqdm(test_loader, position=0, leave=True):
            blocks = [blk.to(self.device) for blk in blocks]
            seeds = seeds.to(self.device)
            input_nodes=input_nodes.to(self.device)
 
            with torch.no_grad():
                logits,h = self.model(blocks,input_nodes)
 
            fin_outputs.append(logits.cpu().detach().numpy())
            
        predicted=np.concatenate(fin_outputs).squeeze()   
        
        return predicted

    def plot_interactive(self, metrics, legend, title, logx=False, logy=False, metric_name='loss', start_from=0):
        import plotly.graph_objects as go
        metric_results = metrics[metric_name]
        xs = [list(range(len(metric_results)))] * len(metric_results[0])
        ys = list(zip(*metric_results))

        fig = go.Figure()
        for i in range(len(ys)):
            fig.add_trace(go.Scatter(x=xs[i][start_from:], y=ys[i][start_from:],
                                     mode='lines+markers',
                                     name=legend[i]))

        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis_title='Epoch',
            yaxis_title=metric_name,
            font=dict(
                size=40,
            ),
            height=600,
        )

        if logx:
            fig.update_layout(xaxis_type="log")
        if logy:
            fig.update_layout(yaxis_type="log")

        fig.show()
