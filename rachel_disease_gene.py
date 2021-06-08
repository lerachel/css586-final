#!/usr/bin/env python
# coding: utf-8

### CSS 586 - Deep Learning
### Professor Dong Si
### Student: Rachel (Thu) Le
### This model is built based on Relation Graph Convolution Networks in DGL library
### Reference: https://docs.dgl.ai/en/0.6.x/guide/training-link.html
### The base model + training loss + evaluation metrics (AUC, ROC) is shared code
### Everything else is individual work such as preprocessing and how data is split 
# and used for training/testing

import dgl
import dgl.nn as dglnn
import dgl.function as fn
from dgl.nn.pytorch import HeteroGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import os
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Location of 2 datasets
dis_gene_filepath = 'DG-Miner_miner-disease-gene.tsv'
dis_drug_filepath = 'DCh-Miner_miner-disease-chemical.tsv'

# Function to remove edges with non-shared diseases between 2 datasets
# Not in used because we decide to combine dataset later on, no need to filter unshared disease nodes
def get_modified_disease_gene_dataset(disease_gene_filepath, disease_drug_filepath):
    
    modified_dataset_name = 'Modified_' + disease_gene_filepath
    if os.path.isfile(modified_dataset_name):
        return pd.read_csv(modified_dataset_name, sep='\t')
    else:
    
        # Disease-gene dataset
        dis_gene = pd.read_csv(disease_gene_filepath, sep='\t')

        # Disease-drug dataset
        dis_drug = pd.read_csv(disease_drug_filepath, sep='\t') 

        # View head of disease-gene dataset
        print(dis_gene.columns.values.tolist())

        # View head of disease-drug dataset
        print(dis_drug.columns.values.tolist())

        # Extract list of diseases from disease-gene and disease-drug 
        # (can contain repeated ones)
        disease_dg = np.array(dis_gene.iloc[:, 0])
        disease_dd = np.array(dis_drug.iloc[:, 0])

        # Unique number of diseases in disease-gene
        unique_disease_gene = len(np.unique(disease_dg))
        unique_disease_drug = len(np.unique(disease_dd))
        # Unique number of diseases in disease-drug
        print("Number of Diseases in disease-gene: {}".format(unique_disease_gene))
        print('Number of Diseases in disease-drug: {}'.format(unique_disease_drug))    

        # Find common diseases in 2 datasets, removing the first one 
        # as it's the header name of first column in DataFrame
        print('Finding common diseases in 2 datasets...')
        common_diseases = np.intersect1d(disease_dg, disease_dd)[1:]

        # List of diseases that disease-drug and disease-gene dataset don't share
        removed_diseases = list(set(disease_dg) - set(common_diseases))
        print('Number of non-overlapping diseases: {}'.format(len(removed_diseases)))

        # Remove all non-overlapping diseases from disease-gene set

        print('Removing non-overlapping diseases from disease-gene dataset...')
        modified_dis_gene = dis_gene[~dis_gene['# Disease(MESH)'].isin(removed_diseases)]
        print('Number of total disease nodes in original disease-gene: {}'.format(len(dis_gene)))
        print('Number of total disease nodes in modified disease-gene: {}'.format(len(modified_dis_gene)))
        print('Diff in number between 2 lists: {}'.format(len(dis_gene)-len(modified_dis_gene)))

        # Save modified dataset to disk

        print('Saving modified disease-gene dataset to disk...')
        modified_dis_gene.to_csv('Modified_DG-Miner_miner-disease-gene.tsv', sep='\t', index=False)
        print('Done!')

        return modified_dis_gene

# Create modified dataset of disease-gene after remove non-shared diseases and its edges from disease-gene dataset
modified_dis_gene = get_modified_disease_gene_dataset(dis_gene_filepath, dis_drug_filepath)


data = pd.read_csv('DG-Miner_miner-disease-gene.tsv', sep='\t')

# Convert dataset in pandas to numpy format

np_dataset = pd.DataFrame.to_numpy(data)

# shape of dataset should be 2 columns
# first is disease, second is gene, representing edge list
print("\nFirst 10 edges in dataset:")
print(np_dataset[:10,:])

# Convert categorical data to int using LabelEncoder
print('\nConverting categorical data to integer...')
disease_encoder = LabelEncoder()
gene_encoder = LabelEncoder()

encoded_disease = disease_encoder.fit_transform(np_dataset[:,0])
encoded_gene = gene_encoder.fit_transform(np_dataset[:,1])

new_dataset = np.hstack((encoded_disease.reshape(-1,1), encoded_gene.reshape(-1,1)))
print("\nFirst 10 edges in encoded dataset: ")
print(new_dataset[:10,:])

# Split into train and test set
print('Applying train_test_split with test_size=0.1, random_state=101...')

train_set, test_set = train_test_split(new_dataset, test_size = 0.2, random_state = 21)

# ------ 1. Load heterogeneous graph ------ #

train_g = dgl.heterograph({
    ('disease', 'caused by', 'gene'): (train_set[:,0], train_set[:,1]),
    ('gene', 'causes', 'disease'): (train_set[:,1], train_set[:,0])})

test_g = dgl.heterograph({
    ('disease', 'caused by', 'gene'): (test_set[:,0], test_set[:,1]),
    ('gene', 'causes', 'disease'): (test_set[:,1], test_set[:,0])})

# Remove disease node from train_set[0]
print("Disease node to be removed: {}".format(train_set[0][0]))

train_g.remove_nodes(torch.tensor(train_set[0][0]), ntype='disease')


print("#### train graph's meta data ####\n{}\n".format(train_g))
print("#### test graph's meta data ####\n{}\n".format(test_g))

# Based on DGL library, compute edge embedding using source and destination node embedding
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            #graph.nodes(etype) = graph.nodes(etype)[:-1]
            #graph.nodes(etype).data['w'][:-1] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

# Based on DGL library, Relation Graph Convolution Neural Network
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


# The actual model, based on DGL, consist of RGCN and HeteroDotProductPredictor module
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

# Create negative edges for training and testing
def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


# Compute loss during training
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

# Compute AUC for eval metrics
def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


# Compute ROC for eval metrics
def compute_roc_curve(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_curve(labels, scores, pos_label=1)

# Model training
test_losses = []
train_losses = []
test_aucs = []
train_aucs = []

k = 5
model = Model(train_g,15, 15, 5)
opt = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)
num_epochs = 300
for epoch in range(1, num_epochs + 1):
    neg_g = construct_negative_graph(train_g, k, ('disease', 'caused by', 'gene'))
    pos_score, neg_score = model(train_g, neg_g, ('disease', 'caused by', 'gene'))
    loss = compute_loss(pos_score, neg_score)
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if epoch % 1 ==0:
        with torch.no_grad():
            neg_test_graph = construct_negative_graph(test_g, k, ('disease', 'caused by', 'gene'))
            test_pos, test_neg = model(test_g, neg_test_graph, ('disease', 'caused by', 'gene'))
            test_loss = compute_loss(test_pos, test_neg)
            auc = compute_auc(test_pos, test_neg)
            train_auc = compute_auc(pos_score, neg_score)

            test_neg_g = construct_negative_graph(test_g, k, ('disease', 'caused by', 'gene'))
            test_pos, test_neg = model(test_g, test_neg_g, ('disease', 'caused by', 'gene'))
            test_loss = compute_loss(test_pos, test_neg)

            train_auc = compute_auc(pos_score, neg_score)
            test_auc = compute_auc(test_pos, test_neg)

            fpr, tpr, thresholds = compute_roc_curve(test_pos, test_neg)
            
        print('Epoch %d/%d -- Training Loss: %0.3f, Test Loss: %0.3f, train AUC: %0.3f, test AUC: %0.3f' %               (epoch,num_epochs,loss.item(), test_loss.item(), train_auc, test_auc))
            
        train_losses.append(loss.item())
        train_aucs.append(train_auc)
        test_losses.append(test_loss.item())
        test_aucs.append(auc)


fig = plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % test_auc)
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





