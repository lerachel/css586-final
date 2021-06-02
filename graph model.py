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
import buildGraph

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


class Model(nn.Module):
    def __init__(self, G, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, G.etypes)
        self.pred = HeteroDotProductPredictor()

        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_features))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)

    def forward(self, g, neg_g, etype):
        h = self.sage(g, self.embed)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)
    #return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def compute_roc_curve(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_curve(labels, scores, pos_label=1)


writer = SummaryWriter()

diseaseDrug, diseasePathway, diseaseGene, maps = buildGraph.buildGraph()

tmp = diseaseDrug.astype('int64')
diseasePathway = diseasePathway.astype('int64')

## train test split here
train, test = train_test_split(tmp, test_size = 0.1, random_state = 12)
pathTrain, pathTest = train_test_split(diseasePathway, test_size = 0.1, random_state = 12)

#make two graphs, one for test one for training
hetero_graph = dgl.heterograph({
    ('drug', 'treats', 'disease') : (torch.tensor(train[:,1]),torch.tensor(train[:,0])),
    ('disease', 'treated-by', 'drug') : (torch.tensor(train[:,0]),torch.tensor(train[:,1])),
    ('pathway', 'used-by', 'disease') : (torch.tensor(pathTrain[:,1]),torch.tensor(pathTrain[:,0])),
    ('disease', 'uses', 'pathway') : (torch.tensor(pathTrain[:,0]),torch.tensor(pathTrain[:,1]))
})

test_graph = dgl.heterograph({
    ('drug', 'treats', 'disease') : (torch.tensor(test[:,1]),torch.tensor(test[:,0])),
    ('disease', 'treated-by', 'drug') : (torch.tensor(test[:,0]),torch.tensor(test[:,1])),
    ('pathway', 'used-by', 'disease') : (torch.tensor(pathTest[:,1]),torch.tensor(pathTest[:,0])),
    ('disease', 'uses', 'pathway') : (torch.tensor(pathTest[:,0]),torch.tensor(pathTest[:,1]))
})


test_losses = []
train_losses = []
test_aucs = []
train_aucs = []

k = 5
model = Model(hetero_graph,15, 15, 5)
opt = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)
num_epochs = 3
for epoch in range(num_epochs):
    negative_graph = construct_negative_graph(hetero_graph, k, ('drug', 'treats', 'disease'))
    pos_score, neg_score = model(hetero_graph, negative_graph, ('drug', 'treats', 'disease'))
    loss = compute_loss(pos_score, neg_score)
    opt.zero_grad()
    loss.backward()
    train_loss = loss.item()
    opt.step()
    with torch.no_grad():
        #predit on test graph
        neg_test_graph = construct_negative_graph(test_graph, k, ('drug', 'treats', 'disease'))
        test_pos, test_neg = model(test_graph, neg_test_graph, ('drug', 'treats', 'disease'))
        test_loss = compute_loss(test_pos, test_neg)
        auc = compute_auc(test_pos, test_neg)
        train_auc = compute_auc(pos_score, neg_score)
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

with torch.no_grad():

    neg_test_graph = construct_negative_graph(test_graph, k, ('drug', 'treats', 'disease'))
    test_pos, test_neg = model(test_graph, neg_test_graph, ('drug', 'treats', 'disease'))

    fpr, tpr, thresholds = compute_roc_curve(test_pos, test_neg)

fig = plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show() 

fig.savefig('ROC.png',dpi=400,format='png')
writer.add_figure('ROC', fig)

fig = plt.figure()
lw = 2
plt.plot(train_losses, color='darkorange',
         lw=lw, label='Training Loss')
plt.plot(test_losses, color='navy', lw=lw,label='Validation Loss')
plt.plot(test_aucs, color='black', lw=lw,label='Validation AUC Score')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend(loc="center right")
plt.show()

fig.savefig('loss.png',dpi=400,format='png')

writer.flush()
writer.close()
