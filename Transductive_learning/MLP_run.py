import math 
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import edge_subgraph
from dgl.nn.functional import edge_softmax
import dgl.nn as dglnn
import dgl.function as fn

from GraphSage_Model import *
from evaluation import *
from mask_function import *
from MLP_Model import *
from print_func import *

import time
import utils

def MLP_run(args,usaanr_feat,device,data):
    
    G, train_mask, test_mask=data
    graph_df_class=transductive_graph(G,args.train_test_split,args.seed)
    DF, LABEL, USAANR, train_idx, test_idx, train_classes_weight, categorical_index = graph_df_class.graph_2_df()
    
    train_y=LABEL[train_idx].squeeze()
    test_y=LABEL[test_idx].squeeze()
    
    df_train=DF.loc[train_idx]
    df_test=DF.loc[test_idx]
    
    usaanr_train=USAANR[train_idx]
    usaanr_test=USAANR[test_idx]
    
    num_classes=th.unique(th.from_numpy(train_y)).shape[0]
    if args.loss_weight:
        train_classes_num, train_classes_weight = get_class_count_weight(train_y,num_classes)
        loss_weight=th.tensor(train_classes_weight).to(device)
    else:
        loss_weight=None
        
    train=Batch_Dataset(df_train, train_y)
    test=Batch_Dataset(df_test,test_y)

    train_dl = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=args.batch_size, shuffle=False)    
    
    df_all=df_train.append(df_test)
    
    mlp_model= MLP(df_all,h_dim=args.h_dim,out_dim=num_classes,n_layers=2,device=device,feat_list=usaanr_feat,dropout=0.2)
    if device !="cpu":
        mlp_model.cuda()
    
    optimizer = th.optim.Adam(mlp_model.parameters(), lr=args.lr, weight_decay=args.l2norm)
    
    #### Training Loop
    print()
    print("************************************************* ")
    print("========= Training Loop For MLP Model ========= ")
    print("************************************************* ")
    print()
    
    LOSS_EPOCH=[]
    LOGIT_train=[]
    LABEL_train=[]

    total_loss=0
    losses=[]

    for epoch in tqdm(range(0,args.n_epochs)):

        mlp_model.train()

        #====================================#
        #            Traning                 #
        #====================================#
        print("")
        print("========= Epoch {:} /{:}".format(epoch+1,args.n_epochs))
        print("Training...")
        t0 = time.time()
        for step, (df_batch, labels_batch) in enumerate(train_dl):

            df_batch=df_batch.to(device)
            labels_batch=labels_batch.to(device)

            logits = mlp_model(df_batch)
            optimizer.zero_grad()

            if args.loss_weight :
                loss = F.cross_entropy(logits.view(-1, num_classes), 
                                       labels_batch.squeeze().long().to(device),weight=loss_weight.float().to(device))
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes), labels_batch.squeeze().long().to(device))

            total_loss+=loss.item()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            arg1=logits[:,1].detach().cpu().numpy()
            arg2=labels_batch.squeeze().cpu().numpy()

            train_gain = lift_gain_eval(arg1,arg2,topk=[0.01,0.05,0.10])

            train_acc = th.sum(logits.argmax(dim=1) == labels_batch.squeeze()).item() / len(labels_batch)
            precision, recall, fscore, support = precision_recall_fscore_support(labels_batch.squeeze().cpu().numpy(), 
                                                                                 logits.argmax(dim=1).cpu().numpy())

            try:
                train_auc = roc_auc_score(labels_batch.squeeze().detach().cpu().numpy().ravel(), th.sigmoid(logits)\
                                          [:,1].detach().cpu().numpy().ravel())
            except ValueError:
                pass

            prec,rec,_ = precision_recall_curve(labels_batch.squeeze().detach().cpu().numpy().ravel(), th.sigmoid(logits)\
                                                [:,1].detach().cpu().numpy().ravel())
            if math.isnan(rec[0])==False:
                train_pr_auc=auc_score(rec,prec)

            LOGIT_train.extend(logits.detach().cpu().numpy().tolist())
            LABEL_train.extend(labels_batch.detach().cpu().squeeze().numpy().tolist())

            if step%(len(train_dl)//10)==0 and not step==0:

                t1 = time.time()
                elapsed=utils.format_time(t1-t0)
                print("Batch {:} of {:} | Loss {:.3f}  | Elapsed: {:}".\
                      format(step,len(train_dl),np.mean(losses[-10:]),elapsed)) 

        LOSS_EPOCH.append(loss)

        mlp_model.eval()
        print()
        print("")
        print("Running Validation on training set")
        print("")

        fin_outputs, fin_targets, losses_tmp=eval_loop_MLP(mlp_model, train_dl, device, loss_weight, num_classes)
        
        avg_loss_train=np.mean(losses_tmp)
        mlp_train=evaluate(fin_targets.reshape(-1),fin_outputs)

        t2=time.time()

        print("avg_loss: {:.2f} | True_Prediction: {:,} | False_Prediction: {:,} | accuracy: {:.2%} |  precision: {:.2%} | recall: \
            {:.2%} | F1_score: {:.2%} | Gain_top-10%: {:.1f} | ROC_AUC: {:.1%} | PR_AUC: {:.1%} | Elapsed: {:}".format(avg_loss_train, 
              mlp_train["true_prediction"], mlp_train["false_prediction"], mlp_train["accuracy"], \
              mlp_train["precision"], mlp_train["recall"],mlp_train["f1_score"], \
              mlp_train["GAIN"]['10%'], mlp_train["AUC"],mlp_train["pr_auc"],utils.format_time(t2-t1)))

        #====================================#
        #            Test-set                #
        #====================================#
        mlp_model.eval()
        print()
        print("")
        print("Running Validation on test set")
        print("")

        fin_outputs, fin_targets, losses_tmp=eval_loop_MLP(mlp_model, test_dl, device, loss_weight, num_classes)
        
        avg_loss_test=np.mean(losses_tmp)
        mlp_test=evaluate(fin_targets.reshape(-1),fin_outputs)
        
        t3=time.time()

        print("avg_loss: {:.2f} | True_Prediction: {:,} | False_Prediction: {:,} | accuracy: {:.2%} |  precision: {:.2%} | recall: \
            {:.2%} | F1_score: {:.2%} | Gain_top-10%: {:.1f} | ROC_AUC: {:.1%} | PR_AUC: {:.1%} | Elapsed: {:}".format(avg_loss_test, 
              mlp_test["true_prediction"], mlp_test["false_prediction"], mlp_test["accuracy"], \
              mlp_test["precision"], mlp_test["recall"],mlp_test["f1_score"], \
              mlp_test["GAIN"]['10%'], mlp_test["AUC"],mlp_test["pr_auc"],utils.format_time(t3-t2)))
        
    train_mlp=OrderedDict()
    train_mlp['model']="MLP"
    train_mlp['% test']=str(args.train_test_split*100)+"%"
    train_mlp['nb_example']=mlp_train['nb_example']
    train_mlp['true_prediction']=mlp_train['true_prediction']
    train_mlp['false_prediction']=mlp_train['false_prediction']
    train_mlp['accuracy']=mlp_train['accuracy']
    train_mlp['precision']=mlp_train['precision']
    train_mlp[ 'recall']=mlp_train[ 'recall']
    train_mlp[ 'f1_score']=mlp_train[ 'f1_score']
    train_mlp[ 'AUC']=mlp_train[ 'AUC']
    train_mlp[ 'pr_auc']=mlp_train[ 'pr_auc']
    
    test_mlp=OrderedDict()
    test_mlp['model']="MLP"
    test_mlp['% test']=str(args.train_test_split*100)+"%"
    test_mlp['nb_example']=mlp_test['nb_example']
    test_mlp['true_prediction']=mlp_test['true_prediction']
    test_mlp['false_prediction']=mlp_test['false_prediction']
    test_mlp['accuracy']=mlp_test['accuracy']
    test_mlp['precision']=mlp_test['precision']
    test_mlp[ 'recall']=mlp_test[ 'recall']
    test_mlp[ 'f1_score']=mlp_test[ 'f1_score']
    test_mlp[ 'AUC']=mlp_test[ 'AUC']
    test_mlp[ 'pr_auc']=mlp_test[ 'pr_auc']

    return train_mlp, test_mlp
