import scipy.io
import urllib.request
import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn as dglnn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import projectUtils
import graph_models
import json

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

def compute_roc_curve(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_curve(labels, scores, pos_label=1)

class Model(nn.Module):
    def __init__(self, G, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = graph_models.RGCN(in_features, hidden_features, out_features, G.etypes)
        self.pred = graph_models.HeteroDotProductPredictor()

        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_features))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)

    def forward(self, g, neg_g, etype):
        h = self.sage(g, self.embed)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)

    def compute_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)])
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        return F.binary_cross_entropy_with_logits(scores, labels)

    def compute_auc(self, pos_score, neg_score):
        scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)]).numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
        return roc_auc_score(labels, scores)

    def fit(self,opt,graph,test_graph,k=1,num_epochs=5):
        test_losses = []
        train_losses = []
        test_aucs = []
        train_aucs = []

        for epoch in range(num_epochs):
            negative_graph = projectUtils.construct_negative_graph(graph, k, ('drug', 'treats', 'disease'))
            pos_score, neg_score = model(graph, negative_graph, ('drug', 'treats', 'disease'))
            loss = self.compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            train_loss = loss.item()
            opt.step()
            with torch.no_grad():
                #predit on test graph
                neg_test_graph = projectUtils.construct_negative_graph(test_graph, k, ('drug', 'treats', 'disease'))
                test_pos, test_neg = self(test_graph, neg_test_graph, ('drug', 'treats', 'disease'))
                test_loss = self.compute_loss(test_pos, test_neg)
                auc = self.compute_auc(test_pos, test_neg)
                train_auc = self.compute_auc(pos_score, neg_score)
                #get validation/test scores
                if epoch % 5 ==0:
                    print('Epoch %d/%d -- Training Loss: %0.3f, Test Loss: %0.3f,AUC: %0.3f' % (epoch,num_epochs,train_loss, test_loss.item(), auc))

            writer.add_scalar('Loss/train', train_loss, epoch)
            train_losses.append(train_loss)
            writer.add_scalar('AUC/train', train_auc, epoch)
            train_aucs.append(train_auc)
            writer.add_scalar('Loss/test', test_loss.item(), epoch)
            test_losses.append(test_loss.item())
            writer.add_scalar('AUC/test', auc, epoch)
            test_aucs.append(auc)

        return train_losses, test_losses, train_aucs, test_aucs

class ModelMLP(Model):
    def __init__(self, G, in_features, hidden_features, out_features):
        super().__init__(G, in_features, hidden_features, out_features)
        self.sage = graph_models.RGCN3conv(in_features, hidden_features, out_features, G.etypes)
        self.pred = graph_models.MLPPredictor(out_features)

        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_features))
                        for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)

class Model3Conv(Model):
    def __init__(self, G, in_features, hidden_features, out_features):
        super().__init__(G, in_features, hidden_features, out_features)
        self.sage = graph_models.RGCN3conv(in_features, hidden_features, out_features, G.etypes)
        self.pred = graph_models.MLPPredictor(out_features)

        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_features))
                        for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)

class Model5Conv(Model):
    def __init__(self, G, in_features, hidden_features, out_features):
        super().__init__(G, in_features, hidden_features, out_features)
        self.sage = graph_models.RGCN5conv(in_features, hidden_features, out_features, G.etypes)
        self.pred = graph_models.MLPPredictor(out_features)

        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_features))
                        for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)

class ModelSAGE(Model):
    def __init__(self, G, in_features, hidden_features, out_features):
        super().__init__(G, in_features, hidden_features, out_features)
        self.sage = graph_models.RGCNSAGE(in_features, hidden_features, out_features, G.etypes)
        self.pred = graph_models.MLPPredictor(out_features)

        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_features))
                        for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)

class ModelMix(Model):
    def __init__(self, G, in_features, hidden_features, out_features):
        super().__init__(G, in_features, hidden_features, out_features)
        self.sage = graph_models.RGCNMix(in_features, hidden_features, out_features, G.etypes)
        self.pred = graph_models.MLPPredictor(out_features)

        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_features))
                        for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)


writer = SummaryWriter()

n_epochs = 400

diseaseDrug, diseasePathway, diseaseGene, maps = projectUtils.buildGraph()

tmp = diseaseDrug.astype('int64')
diseasePathway = diseasePathway.astype('int64')
diseaseGene = diseaseGene.astype('int64')

## train test split here
train, test = train_test_split(tmp, test_size = 0.2, random_state = 12)
pathTrain, pathTest = train_test_split(diseasePathway, test_size = 0.2, random_state = 7)
geneTrain, geneTest = train_test_split(diseaseGene, test_size = 0.2, random_state = 3)

#make two graphs, one for test one for training (and more for comparison)
full_graph = dgl.heterograph({
    ('drug', 'treats', 'disease') : (torch.tensor(train[:,1]),torch.tensor(train[:,0])),
    ('disease', 'treated-by', 'drug') : (torch.tensor(train[:,0]),torch.tensor(train[:,1])),
    ('pathway', 'used-by', 'disease') : (torch.tensor(pathTrain[:,1]),torch.tensor(pathTrain[:,0])),
    ('disease', 'uses', 'pathway') : (torch.tensor(pathTrain[:,0]),torch.tensor(pathTrain[:,1])),
    ('gene', 'related', 'disease') : (torch.tensor(geneTrain[:,1]),torch.tensor(geneTrain[:,0])),
    ('disease', 'relates', 'gene') : (torch.tensor(geneTrain[:,0]),torch.tensor(geneTrain[:,1]))
})

full_test_graph = dgl.heterograph({
    ('drug', 'treats', 'disease') : (torch.tensor(test[:,1]),torch.tensor(test[:,0])),
    ('disease', 'treated-by', 'drug') : (torch.tensor(test[:,0]),torch.tensor(test[:,1])),
    ('pathway', 'used-by', 'disease') : (torch.tensor(pathTest[:,1]),torch.tensor(pathTest[:,0])),
    ('disease', 'uses', 'pathway') : (torch.tensor(pathTest[:,0]),torch.tensor(pathTest[:,1])),
    ('gene', 'related', 'disease') : (torch.tensor(geneTest[:,1]),torch.tensor(geneTest[:,0])),
    ('disease', 'relates', 'gene') : (torch.tensor(geneTest[:,0]),torch.tensor(geneTest[:,1]))
})

DCh_graph = full_graph.edge_type_subgraph(["treats","treated-by"])
DCh_test_graph = full_test_graph.edge_type_subgraph(["treats","treated-by"])

DChG_graph = full_graph.edge_type_subgraph(["treats","treated-by","related","relates"])
DChG_test_graph = full_test_graph.edge_type_subgraph(["treats","treated-by","related","relates"])


'''
## Begin number of convolutions comparison ##

#2 Convolutions w/ MLP predictor on Disease-Drug Graph
model = ModelMLP(DCh_graph,15, 15, 20)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
Conv2train_losses, Conv2test_losses, Conv2train_aucs, Conv2test_aucs = model.fit(optimizer,DCh_graph,DCh_test_graph,num_epochs=n_epochs)
with torch.no_grad():
    neg_test_graph = projectUtils.construct_negative_graph(DCh_test_graph, 1, ('drug', 'treats', 'disease'))
    test_pos, test_neg = model(DCh_test_graph, neg_test_graph, ('drug', 'treats', 'disease'))

    Conv2fpr, Conv2tpr, Conv2thresholds = compute_roc_curve(test_pos, test_neg)
    Conv2auc = model.compute_auc(test_pos, test_neg)

#3 Convolutions w/ MLP predictor on Disease-Drug Graph
model = Model3Conv(DCh_graph,15, 15, 20)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
Conv3train_losses, Conv3test_losses, Conv3train_aucs, Conv3test_aucs = model.fit(optimizer,DCh_graph,DCh_test_graph,num_epochs=n_epochs)
with torch.no_grad():
    neg_test_graph = projectUtils.construct_negative_graph(DCh_test_graph, 1, ('drug', 'treats', 'disease'))
    test_pos, test_neg = model(DCh_test_graph, neg_test_graph, ('drug', 'treats', 'disease'))

    Conv3fpr, Conv3tpr, Conv3thresholds = compute_roc_curve(test_pos, test_neg)
    Conv3auc = model.compute_auc(test_pos, test_neg)

#5 Convolutions w/ MLP predictor on Disease-Drug Graph
model = Model5Conv(DCh_graph,15, 15, 20)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
Conv5train_losses, Conv5test_losses, Conv5train_aucs, Conv5test_aucs = model.fit(optimizer,DCh_graph,DCh_test_graph,num_epochs=n_epochs)
with torch.no_grad():
    neg_test_graph = projectUtils.construct_negative_graph(DCh_test_graph, 1, ('drug', 'treats', 'disease'))
    test_pos, test_neg = model(DCh_test_graph, neg_test_graph, ('drug', 'treats', 'disease'))

    Conv5fpr, Conv5tpr, Conv5thresholds = compute_roc_curve(test_pos, test_neg)
    Conv5auc = model.compute_auc(test_pos, test_neg)

fig = plt.figure()
lw = 2
plt.plot(Conv2fpr, Conv2tpr, color='darkorange',
         lw=lw, label='2 Conv ROC curve (area = %0.3f)' % Conv2auc)
plt.plot(Conv3fpr, Conv3tpr, color='olive',
         lw=lw, label='3 Conv ROC curve (area = %0.3f)' % Conv3auc)
plt.plot(Conv5fpr, Conv5tpr, color='gray',
         lw=lw, label='5 Conv ROC curve (area = %0.3f)' % Conv5auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show() 

fig.savefig('ROC_Convolutions.png',dpi=400,format='png')

fig = plt.figure()
lw = 2
plt.plot(Conv2train_losses, color='darkorange',
         lw=lw, label='2 Conv Training Loss')
plt.plot(Conv2test_losses, color='darkorange', lw=lw,label='2 Conv Validation Loss',linestyle='--')
plt.plot(Conv3train_losses, color='olive',
         lw=lw, label='3 Conv Training Loss')
plt.plot(Conv3test_losses, color='olive', lw=lw,label='3 Conv Validation Loss',linestyle='--')
plt.plot(Conv5train_losses, color='gray',
         lw=lw, label='5 Conv Training Loss')
plt.plot(Conv5test_losses, color='gray', lw=lw,label='5 Conv Validation Loss',linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross Entropy Loss')
plt.legend(loc="center right")
plt.show()

fig.savefig('loss_Convolutions.png',dpi=400,format='png')


## Begin GNN Model Comparison ##

#2 Convolutions w/ MLP predictor on Disease-Drug Graph
model = ModelSAGE(DCh_graph,15, 15, 20)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
SAGEtrain_losses, SAGEtest_losses, SAGEtrain_aucs, SAGEtest_aucs = model.fit(optimizer,DCh_graph,DCh_test_graph,num_epochs=n_epochs)
with torch.no_grad():
    neg_test_graph = projectUtils.construct_negative_graph(DCh_test_graph, 1, ('drug', 'treats', 'disease'))
    test_pos, test_neg = model(DCh_test_graph, neg_test_graph, ('drug', 'treats', 'disease'))

    SAGEfpr, SAGEtpr, SAGEthresholds = compute_roc_curve(test_pos, test_neg)
    SAGEauc = model.compute_auc(test_pos, test_neg)

model = ModelMix(DCh_graph,15, 15, 20)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
mixtrain_losses, mixtest_losses, mixtrain_aucs, mixtest_aucs = model.fit(optimizer,DCh_graph,DCh_test_graph,num_epochs=n_epochs)
with torch.no_grad():
    neg_test_graph = projectUtils.construct_negative_graph(DCh_test_graph, 1, ('drug', 'treats', 'disease'))
    test_pos, test_neg = model(DCh_test_graph, neg_test_graph, ('drug', 'treats', 'disease'))

    mixfpr, mixtpr, mixthresholds = compute_roc_curve(test_pos, test_neg)
    mixauc = model.compute_auc(test_pos, test_neg)

fig = plt.figure()
lw = 2
plt.plot(Conv3fpr, Conv3tpr, color='darkorange',
         lw=lw, label='GCN ROC curve (area = %0.3f)' % Conv3auc)
plt.plot(SAGEfpr, SAGEtpr, color='olive',
         lw=lw, label='GraphSAGE ROC curve (area = %0.3f)' % SAGEauc)
plt.plot(mixfpr, mixtpr, color='gray',
         lw=lw, label='Mix ROC curve (area = %0.3f)' % mixauc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show() 

fig.savefig('ROC_Models.png',dpi=400,format='png')

fig = plt.figure()
lw = 2
plt.plot(Conv3train_losses, color='darkorange',
         lw=lw, label='GCN Training Loss')
plt.plot(Conv3test_losses, color='darkorange', lw=lw,label='GCN Validation Loss',linestyle='--')
plt.plot(SAGEtrain_losses, color='olive',
         lw=lw, label='GraphSAGE Training Loss')
plt.plot(SAGEtest_losses, color='olive', lw=lw,label='GraphSAGE Validation Loss',linestyle='--')
plt.plot(mixtrain_losses, color='gray',
         lw=lw, label='Mix Training Loss')
plt.plot(mixtest_losses, color='gray', lw=lw,label='Mix Validation Loss',linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross Entropy Loss')
plt.legend(loc="center right")
plt.show()

fig.savefig('loss_Models.png',dpi=400,format='png')

## Begin Graph Size Comparison ## 

#All Available Edges and Nodes
model = ModelMLP(full_graph,15, 15, 20)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
train_losses, test_losses, train_aucs, test_aucs = model.fit(optimizer,full_graph,full_test_graph,num_epochs=n_epochs)
with torch.no_grad():
    neg_test_graph = projectUtils.construct_negative_graph(full_test_graph, 1, ('drug', 'treats', 'disease'))
    test_pos, test_neg = model(full_test_graph, neg_test_graph, ('drug', 'treats', 'disease'))

    fpr, tpr, thresholds = compute_roc_curve(test_pos, test_neg)
    auc = model.compute_auc(test_pos, test_neg)

#Only Disease-Drug Edges and Nodes
model = ModelMLP(DCh_graph,15, 15, 20)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
DChtrain_losses, DChtest_losses, DChtrain_aucs, DChtest_aucs = model.fit(optimizer,DCh_graph,DCh_test_graph,num_epochs=n_epochs)
with torch.no_grad():
    neg_test_graph = projectUtils.construct_negative_graph(DCh_test_graph, 1, ('drug', 'treats', 'disease'))
    test_pos, test_neg = model(DCh_test_graph, neg_test_graph, ('drug', 'treats', 'disease'))

    DChfpr, DChtpr, DChthresholds = compute_roc_curve(test_pos, test_neg)
    DChauc = model.compute_auc(test_pos, test_neg)

#Only Disease-Drug-Gene Edges and Nodes
model = ModelMLP(DChG_graph,15, 15, 20)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
DChGtrain_losses, DChGtest_losses, DChGtrain_aucs, DChGtest_aucs = model.fit(optimizer,DChG_graph,DChG_test_graph,num_epochs=n_epochs)
with torch.no_grad():
    neg_test_graph = projectUtils.construct_negative_graph(DChG_test_graph, 1, ('drug', 'treats', 'disease'))
    test_pos, test_neg = model(DChG_test_graph, neg_test_graph, ('drug', 'treats', 'disease'))

    DChGfpr, DChGtpr, DChGthresholds = compute_roc_curve(test_pos, test_neg)
    DChGauc = model.compute_auc(test_pos, test_neg)


fig = plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='Full Dataset ROC curve (area = %0.3f)' % auc)
plt.plot(DChfpr, DChtpr, color='olive',
         lw=lw, label='Drug-Disease ROC curve (area = %0.3f)' % DChauc)
plt.plot(DChGfpr, DChGtpr, color='gray',
         lw=lw, label='Drug-Disease-Gene ROC curve (area = %0.3f)' % DChGauc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show() 

fig.savefig('ROC_GraphSize.png',dpi=400,format='png')

fig = plt.figure()
lw = 2
plt.plot(train_losses, color='darkorange',
         lw=lw, label='Full Dataset Training Loss')
plt.plot(test_losses, color='darkorange', lw=lw,label='Full Dataset Validation Loss',linestyle='--')
plt.plot(DChtrain_losses, color='olive',
         lw=lw, label='Disease-Drug Training Loss')
plt.plot(DChtest_losses, color='olive', lw=lw,label='Disease-Drug Validation Loss',linestyle='--')
plt.plot(DChGtrain_losses, color='gray',
         lw=lw, label='Disease-Drug-Gene Training Loss')
plt.plot(DChGtest_losses, color='gray', lw=lw,label='Disease-Drug-Gene Validation Loss',linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross Entropy Loss')
plt.legend(loc="center right")
plt.show()

fig.savefig('loss_GraphSize.png',dpi=400,format='png')


## Begin MLP Consine Similarity Comparison ## 

model = Model(DCh_graph,15, 15, 20)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
DOTtrain_losses, DOTtest_losses, DOTtrain_aucs, DOTtest_aucs = model.fit(optimizer,DCh_graph,DCh_test_graph,num_epochs=n_epochs)
with torch.no_grad():
    neg_test_graph = projectUtils.construct_negative_graph(DCh_test_graph, 1, ('drug', 'treats', 'disease'))
    test_pos, test_neg = model(DCh_test_graph, neg_test_graph, ('drug', 'treats', 'disease'))

    DOTfpr, DOTtpr, DOTthresholds = compute_roc_curve(test_pos, test_neg)
    DOTauc = model.compute_auc(test_pos, test_neg)

model = ModelMLP(DCh_graph,15, 15, 20)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
MLPtrain_losses, MLPtest_losses, MLPtrain_aucs, MLPtest_aucs = model.fit(optimizer,DCh_graph,DCh_test_graph,num_epochs=n_epochs)
with torch.no_grad():
    neg_test_graph = projectUtils.construct_negative_graph(DCh_test_graph, 1, ('drug', 'treats', 'disease'))
    test_pos, test_neg = model(DCh_test_graph, neg_test_graph, ('drug', 'treats', 'disease'))

    MLPfpr, MLPtpr, MLP50thresholds = compute_roc_curve(test_pos, test_neg)
    MLPauc = model.compute_auc(test_pos, test_neg)


fig = plt.figure()
lw = 2
plt.plot(DOTfpr, DOTtpr, color='darkorange',
         lw=lw, label='Cosine Similarity ROC curve (area = %0.3f)' % DOTauc)
plt.plot(Conv2fpr, Conv2tpr, color='olive',
         lw=lw, label='MLP ROC curve (area = %0.3f)' % Conv2auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show() 

fig.savefig('ROC_MLP_DOT.png',dpi=400,format='png')

fig = plt.figure()
lw = 2
plt.plot(DOTtrain_losses, color='darkorange',
         lw=lw, label='Cosine Similarity Training Loss')
plt.plot(DOTtest_losses, color='darkorange', lw=lw,label='Cosine Similarity Validation Loss',linestyle='--')
plt.plot(Conv2train_losses, color='olive',
         lw=lw, label='MLP Training Loss')
plt.plot(Conv2test_losses, color='olive', lw=lw,label='MLP Validation Loss',linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross Entropy Loss')
plt.legend(loc="center right")
plt.show()

fig.savefig('loss_MLP_DOT.png',dpi=400,format='png')
'''


## Begin Model Grid Search ##

embedded_space = [15,30,100]
hidden_layers = [5,50,100]
latent_space = [5,20,50,100]
prediction = ['DOT','MLP']
results = {}

n_epochs = 300

i=0
for types in prediction:
    for spc in embedded_space:
        for layers in hidden_layers:
            for latent in latent_space:
                results[i]= {'parameters': {
                    'prediction': types,
                    'embedded space size': spc,
                    'hidden layer size': layers
                }}

                if types == 'MLP':
                    model = ModelMLP(DCh_graph,spc, layers, latent)
                else:
                    model = Model(DCh_graph,spc, layers, latent)

                optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
                train_losses, test_losses, train_aucs, test_aucs = model.fit(optimizer,DCh_graph,DCh_test_graph,num_epochs=n_epochs)
                with torch.no_grad():
                    neg_test_graph = projectUtils.construct_negative_graph(DCh_test_graph, 1, ('drug', 'treats', 'disease'))
                    test_pos, test_neg = model(DCh_test_graph, neg_test_graph, ('drug', 'treats', 'disease'))

                    fpr, tpr, thresholds = compute_roc_curve(test_pos, test_neg)
                    auc = model.compute_auc(test_pos, test_neg)

                results[i]['results'] = {
                    'training_loss': train_losses,
                    'validation_loss': test_losses,
                    'train_score': train_aucs,
                    'validation_Score': test_aucs,
                    'test_pos': test_pos.tolist(),
                    'test_neg': test_neg.tolist(),
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                    'auc': auc.tolist()
                }
                i+=1

with open("results.json", "w") as outfile: 
    json.dump(results, outfile)


writer.flush()
writer.close()
