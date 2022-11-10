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
from MLP_Model import *
from print_func import *

import time
import utils


def graph_run_featureless(args,usaanr_feat,device,data):
    
    G, train_g, test_g, test_idx=data
#     G=G.to(device)
    
    train_idx=th.arange(train_g.num_nodes()).squeeze()       
    test_idx=th.from_numpy(test_idx).squeeze()

    train_label=train_g.nodes['usaanr'].data['label']
    test_label=test_g.nodes['usaanr'].data['label']

    label_train=train_label.squeeze().numpy()
    label_test=test_label.squeeze().numpy()
    
    print('{:<15} {:<10,}'.format("Training set",train_idx.shape[0]))
    print('{:<15} {:<10,}'.format("test set",test_idx.shape[0]))
    print()
    
    num_classes=th.unique(th.from_numpy(label_train)).shape[0]
    if args.loss_weight:
        train_classes_num, train_classes_weight = get_class_count_weight(label_train,num_classes)
        loss_weight=th.tensor(train_classes_weight).to(device)
    else:
        loss_weight=None

    model = Entity_Classify_FeatureLess(G,
                                        device,
                                        args.h_dim,
                                        num_classes,
                                        num_bases=args.num_bases,
                                        num_hidden_layers=args.num_layers,
                                        dropout=args.dropout,
                                        use_self_loop=args.use_self_loop)
    if device !="cpu":
        model.cuda()

    optimizer = th.optim.Adam(model.parameters(), lr=args.featureless_lr, weight_decay=args.l2norm)
    
    # train sampler
    train_sampler = dgl.dataloading.MultiLayerNeighborSampler([args.fanout] * args.num_layers)
    train_loader = dgl.dataloading.NodeDataLoader(
        train_g, {'usaanr': train_idx}, train_sampler,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

    test_sampler = dgl.dataloading.MultiLayerNeighborSampler([args.fanout] * args.num_layers)
    test_loader = dgl.dataloading.NodeDataLoader(
        test_g, {'usaanr': test_idx}, test_sampler,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
        
    print("The number of minibatch in training set is {:,}".format(len(train_loader)))
    print("The number of minibatch in test set is {:,}".format(len(test_loader)))
    print()

    
    #### Training Loop
    print()
    print("***************************************************************** ")
    print("========= Training Loop For Graph Model without feature ========= ")
    print("***************************************************************** ")
    print()
    
    LOSS_EPOCH=[]
    LABEL_TRAIN=[]
    total_loss=0
    losses=[]
    LOGIT_train=[]
    LABEL_train=[]
    
    for epoch in tqdm(range(0,args.n_epochs)):
    
        model.train()
        IDX=[]
        H=[]

        #====================================#
        #            Traning                 #
        #====================================#
        print("")
        print("========= Epoch {:} /{:}".format(epoch+1,args.n_epochs))
        print("Training...")
        t0 = time.time()
        for step, (input_nodes_raw, seeds_raw, blocks) in enumerate(train_loader):
            blocks = [blk.to(device) for blk in blocks]

            seeds=seeds_raw.to(device)

            labels_train=train_label[seeds].to(device)       

            input_nodes={}
            input_nodes["usaanr"]=input_nodes_raw
            input_nodes={k : e.to(device) for k, e in input_nodes.items()}

            logits,h = model(train_g,input_nodes,blocks)
            optimizer.zero_grad()

            if args.loss_weight :
                loss = F.cross_entropy(logits.view(-1, num_classes), 
                                       labels_train.squeeze().to(device),weight=loss_weight.float().to(device))
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes), labels_train.squeeze().to(device))

            total_loss+=loss.item()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            arg1=logits[:,1].detach().cpu().numpy()
            arg2=labels_train.cpu().numpy()

            train_gain = lift_gain_eval(arg1,arg2,topk=[0.01,0.05,0.10])

            train_acc = th.sum(logits.argmax(dim=1) == labels_train).item() / len(seeds)
            precision, recall, fscore, support = precision_recall_fscore_support(labels_train.cpu().numpy(), 
                                                                                 logits.argmax(dim=1).cpu().numpy())

            try:
                train_auc = roc_auc_score(labels_train.detach().cpu().numpy().ravel(), th.sigmoid(logits)\
                                          [:,1].detach().cpu().numpy().ravel())
            except ValueError:
                pass

            prec,rec,_ = precision_recall_curve(labels_train.detach().cpu().numpy().ravel(), th.sigmoid(logits)\
                                                [:,1].detach().cpu().numpy().ravel())
            if math.isnan(rec[0])==False:
                train_pr_auc=auc_score(rec,prec)

            IDX.extend(seeds.detach().cpu().numpy().tolist())
            H.extend(h["usaanr"].detach().cpu().numpy().tolist())
            LOGIT_train.extend(logits.detach().cpu().numpy().tolist())
            LABEL_train.extend(train_label[blocks[-1].dstnodes['usaanr'].data[dgl.NID].cpu().numpy()].tolist())

            if step%(len(train_loader)//10)==0 and not step==0:

                t1 = time.time()
                elapsed=utils.format_time(t1-t0)
                print("Batch {:} of {:} | Loss {:.3f}  | Elapsed: {:}".\
                      format(step,len(train_loader),np.mean(losses[-10:]),elapsed)) 

        LOSS_EPOCH.append(loss)

        LABEL_TRAIN.append(train_label[blocks[-1].nodes['usaanr'].data[dgl.NID].cpu().numpy()])


        model.eval()
        print()
        print("")
        print("Running Validation on training set")
        print("")
        fin_outputs, fin_targets, losses_tmp=eval_loop_func(train_g, model, train_loader, train_label,  device, loss_weight, num_classes)

        avg_loss_train=np.mean(losses_tmp)

        tmp_mean_pool_train=evaluate(fin_targets.reshape(-1),fin_outputs)

        t2=time.time()

        print("avg_loss: {:.2f} | True_Prediction: {:,} | False_Prediction: {:,} | accuracy: {:.2%} |  precision: {:.2%} | recall: \
            {:.2%} | F1_score: {:.2%} | Gain_top-10%: {:.1f} | ROC_AUC: {:.1%} | PR_AUC: {:.1%} | Elapsed: {:}".format(avg_loss_train, 
              tmp_mean_pool_train["true_prediction"], tmp_mean_pool_train["false_prediction"], tmp_mean_pool_train["accuracy"], \
              tmp_mean_pool_train["precision"], tmp_mean_pool_train["recall"],tmp_mean_pool_train["f1_score"], \
              tmp_mean_pool_train["GAIN"]['10%'], tmp_mean_pool_train["AUC"],tmp_mean_pool_train["pr_auc"],utils.format_time(t2-t1)))

        #====================================#
        #            test  set          #
        #====================================#

        model.eval()
        print()
        print("")
        print("Running Validation on test set")
        print("")

        fin_outputs, fin_targets, losses_tmp=eval_loop_func(test_g, model, test_loader, test_label,  device, loss_weight, num_classes)
        
        avg_loss_test=np.mean(losses_tmp)
    
        tmp_mean_pool_test=evaluate(fin_targets.reshape(-1),fin_outputs)
    
        t3=time.time()

        print("avg_loss: {:.2f} | True_Prediction: {:,} | False_Prediction: {:,} | accuracy: {:.2%} |  precision: {:.2%} | recall: \
        {:.2%} | F1_score: {:.2%} | Gain_top-10%: {:.1f} | ROC_AUC: {:.1%} | PR_AUC: {:.1%} | Elapsed: {:}".format(avg_loss_test,\
          tmp_mean_pool_test["true_prediction"], tmp_mean_pool_test["false_prediction"], tmp_mean_pool_test["accuracy"], \
          tmp_mean_pool_test["precision"], tmp_mean_pool_test["recall"], tmp_mean_pool_test["f1_score"], \
          tmp_mean_pool_test["GAIN"]['10%'], tmp_mean_pool_test["AUC"], tmp_mean_pool_test["pr_auc"], utils.format_time(t3-t2)))

    train_graph=OrderedDict()
    train_graph['model']="Graph"
    train_graph['% test']=str(args.train_test_split*100)+"%"
    train_graph['nb_example']=tmp_mean_pool_train['nb_example']
    train_graph['true_prediction']=tmp_mean_pool_train['true_prediction']
    train_graph['false_prediction']=tmp_mean_pool_train['false_prediction']
    train_graph['accuracy']=tmp_mean_pool_train['accuracy']
    train_graph['precision']=tmp_mean_pool_train['precision']
    train_graph[ 'recall']=tmp_mean_pool_train[ 'recall']
    train_graph[ 'f1_score']=tmp_mean_pool_train[ 'f1_score']
    train_graph[ 'AUC']=tmp_mean_pool_train[ 'AUC']
    train_graph[ 'pr_auc']=tmp_mean_pool_train[ 'pr_auc']
    
    test_graph=OrderedDict()
    test_graph['model']="Graph"
    test_graph['% test']=str(args.train_test_split*100)+"%"
    test_graph['nb_example']=tmp_mean_pool_test['nb_example']
    test_graph['true_prediction']=tmp_mean_pool_test['true_prediction']
    test_graph['false_prediction']=tmp_mean_pool_test['false_prediction']
    test_graph['accuracy']=tmp_mean_pool_test['accuracy']
    test_graph['precision']=tmp_mean_pool_test['precision']
    test_graph[ 'recall']=tmp_mean_pool_test[ 'recall']
    test_graph[ 'f1_score']=tmp_mean_pool_test[ 'f1_score']
    test_graph[ 'AUC']=tmp_mean_pool_test[ 'AUC']
    test_graph[ 'pr_auc']=tmp_mean_pool_test[ 'pr_auc']
    
    return train_graph, test_graph

