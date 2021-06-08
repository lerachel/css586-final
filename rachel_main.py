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
import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

import argparse
import os
import sys
import time

'''
List of hyperparameters
--link: 1 or 2 edge type link prediction
--in_feat = 15
--hid_feat = 15
--out_feat = 5
--epoch = 300
'''

# Create the parser to get hyperparameters from command line input
my_parser = argparse.ArgumentParser(description='hyperparameters')
my_parser.add_argument('--link', action='store', type=int, default=1)
my_parser.add_argument('--layer', action='store', type=int, default=2)
my_parser.add_argument('--in_feat', action='store', type=int, default=15)
my_parser.add_argument('--hid_feat', action='store', type=int, default=15)
my_parser.add_argument('--out_feat', action='store', type=int, default=5)
my_parser.add_argument('--epoch', action='store', type=int, default=300)

args = my_parser.parse_args()

for arg in vars(args):
     print('{}: {}'.format(arg, getattr(args, arg)))


link_prediction, layer, in_feat, hidden_feat, out_feat, num_epochs = \
args.link, args.layer, args.in_feat, args.hid_feat, args.out_feat, args.epoch

# Reference: https://docs.dgl.ai/en/0.6.x/api/python/dgl.function.html#api-built-in
# Can create many different Predictor for link scoring based on builtin function or my own for apply_edges

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

# Original RGCN from DGL
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

# RGCN with 3 layers of HeteroGraphConv
class RGCN2(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
 
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
         
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
         
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        return h

# Create negative edges for link prediction
def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


# The actual model, consists of RGCN for graph embedding and HeteroDotProductPredictor for 
# scoring edge loss

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


# Model2 is similar to Model but with 3 layers of HeteroGraphConv in RGCN2
class Model2(nn.Module):
    def __init__(self, G, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = RGCN2(in_features, hidden_features, out_features, G.etypes)
        self.pred = HeteroDotProductPredictor()

        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_features))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)

    def forward(self, g, neg_g, etype):
        h = self.sage(g, self.embed)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)


# Calculate loss for training
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


# Evaluation metrics AUC
def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

# Evaluation metrics ROC
def compute_roc_curve(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_curve(labels, scores, pos_label=1)

# Open two tsv files, extract edge lists, combine datasets

print('Opening disease-gene & disease-chemical datasets from .tsv files...')

file1 = pd.read_csv('DG-Miner_miner-disease-gene.tsv','\t')
file2 = pd.read_csv('DCh-Miner_miner-disease-chemical.tsv', sep='\t')
file = np.concatenate( (file1.to_numpy(), file2.to_numpy()), axis=0)

# Extract disease, gene, chemicals

print('Extracting disease, gene, chemical data...')

disease1 = list(file1.iloc[:, 0])
disease2 = list(file2.iloc[:, 0])
gene = list(file1.iloc[:, 1])
chem = list(file2.iloc[:, 1])

# Combine list of diseases
disease = disease1 + disease2
disease = np.array(disease)

# Get unique disease, chem, gene
unique_disease = list(set(disease1 + disease2))
unique_chem = list(set(chem))
unique_gene = list(set(gene))


# Create numerical labels for disease, gene, chemical
print('Applying Label Encoding for disease, gene and chemicals...')

disease_encoder = LabelEncoder()
gene_encoder = LabelEncoder()
chem_encoder = LabelEncoder()

encoded_disease = disease_encoder.fit_transform(disease)
encoded_gene = gene_encoder.fit_transform(gene)
encoded_chem = chem_encoder.fit_transform(chem)

# Create new datasets with encoded labels

print('Creating new datasets with encoded labels...')

encoded_gene = encoded_gene.reshape(-1,1)
encoded_chem = encoded_chem.reshape(-1,1)
encoded_disease = encoded_disease.reshape(-1,1)
dataset1 = np.hstack((encoded_disease[:len(file1)], encoded_gene))
dataset2 = np.hstack((encoded_disease[len(file1):], encoded_chem))

print('Size of disease-gene dataset: {}'.format(len(file1)))
print('Size of disease-chem dataset: {}'.format(len(file2)))

# Create train and test set based number of link prediction

def create_train_test_graphs(dataset1, dataset2, link_prediction=1): 
    if link_prediction == 2:
        # Need equal amount of training and testing samples for both types of links
        # Allocate 0.5 for each types of link

        print('link_prediction = 2\nApplying 70/30 train_test_split on disease-gene, disease-chem datasets...')
        # train1, test1 are disease-gene; train2, test2 are disease-chem
        train1, test1 = train_test_split(dataset1, test_size = 0.3, random_state = 101)
        train2, test2 = train_test_split(dataset2, test_size = 0.3, random_state = 101)
    
    
        # Create 2 heterographs: 1 for training, 1 for testing

        print('Creating train and test graphs...')

        # Use train1 and train2
        train_graph = dgl.heterograph({
        ('disease', 'relate', 'gene') : (torch.tensor(train1[:,0]), torch.tensor(train1[:,1])),
        ('gene', 'related', 'disease') : (torch.tensor(train1[:,1]), torch.tensor(train1[:,0])),
        ('disease', 'treated-by', 'drug') : (torch.tensor(train2[:,0]),torch.tensor(train2[:,1])),
        ('drug', 'treats', 'disease') : (torch.tensor(train2[:,1]),torch.tensor(train2[:,0])),    
        })


        # Use test1 and test2
        test_graph = dgl.heterograph({
        ('disease', 'relate', 'gene') : (torch.tensor(test1[:,0]), torch.tensor(test1[:,1])),
        ('gene', 'related', 'disease') : (torch.tensor(test1[:,1]), torch.tensor(test1[:,0])),    
        ('drug', 'treats', 'disease') : (torch.tensor(test2[:,1]),torch.tensor(test2[:,0])),
        ('disease', 'treated-by', 'drug') : (torch.tensor(test2[:,0]),torch.tensor(test2[:,1]))
        })
        
        return train_graph, test_graph
    
    print('link_prediction = 1\nApplying train_test_split to disease-gene dataset...')
    # link_prediction = 1, only prediction disease-gene link
    train, test = train_test_split(dataset1, test_size = 0.3, random_state = 101)
        
    train_graph = dgl.heterograph({
        ('disease', 'relate', 'gene') : (torch.tensor(train[:,0]), torch.tensor(train[:,1])),
        ('gene', 'related', 'disease') : (torch.tensor(train[:,1]), torch.tensor(train[:,0])),
        ('disease', 'treated-by', 'drug') : (torch.tensor(dataset2[:,0]),torch.tensor(dataset2[:,1])),
        ('drug', 'treats', 'disease') : (torch.tensor(dataset2[:,1]),torch.tensor(dataset2[:,0])),    
        })

    test_graph = dgl.heterograph({
    ('disease', 'relate', 'gene') : (torch.tensor(test[:,0]), torch.tensor(test[:,1])),
    ('gene', 'related', 'disease') : (torch.tensor(test[:,1]), torch.tensor(test[:,0]))
    })   
    
    return train_graph, test_graph

# train test split here
train_graph, test_graph = create_train_test_graphs(dataset1, dataset2, link_prediction)


# Function to remove nodes in train graph so train and test graph have equal numbers of edges
def preprocess_graphs(train_graph, test_graph):
    print('Num of disease nodes in train graph: {}'.format(len(train_graph.nodes('disease'))))
    print('Num of disease nodes in test_graph: {}'.format(len(test_graph.nodes('disease'))))
    
    if len(train_graph.nodes('disease')) > len(test_graph.nodes('disease')):
        diff = len(train_graph.nodes('disease')) - len(test_graph.nodes('disease'))
        # remove diff number of diseases starting from beginning of train_graph
        train_graph.remove_nodes(torch.tensor(train_graph.nodes('disease')[:diff]), ntype='disease')
        return train_graph
    
    print('train_graph and test_graph have equal num of disease. No processing needed.')
    return train_graph

train_graph = preprocess_graphs(train_graph, test_graph)

print("#### Train graph's metadata #### \n{}\n".format(train_graph))
print("#### Test graph's metadata #### \n{}\n".format(test_graph))


# In[74]:
train_losses = []
test_losses = []
train_aucs = []
test_aucs = []
epochs = []
fpr, tpr, test_auc = 0,0,0
if layer == 3:
    model = Model2(train_graph, in_feat, hidden_feat, out_feat)
else:
    model = Model(train_graph, in_feat, hidden_feat, out_feat)
        
start_time = time.time()

### 1 edge-type prediction ###
if link_prediction == 1:

    print('Starting training 1 link prediction: disease-gene...')

    k = out_feat
    opt = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)
    for epoch in range(1, num_epochs + 1):
        negative_graph = construct_negative_graph(train_graph, k, ('disease', 'relate', 'gene'))
        pos_score, neg_score = model(train_graph, negative_graph, ('disease', 'relate', 'gene'))
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                neg_test_graph = construct_negative_graph(test_graph, k, ('disease', 'relate', 'gene'))
                test_pos, test_neg = model(test_graph, neg_test_graph, ('disease', 'relate', 'gene'))
                test_loss = compute_loss(test_pos, test_neg)
                test_auc = compute_auc(test_pos, test_neg)
                train_auc = compute_auc(pos_score, neg_score)
                
                epochs.append(epoch)
                train_losses.append(loss.item())
                test_losses.append(test_loss.item())
                train_aucs.append(train_auc)
                test_aucs.append(test_auc)
            print('Epoch %d/%d -- Train Loss: %0.3f, Train AUC: %0.3f, Validation Loss: %0.3f, Test AUC: %0.3f' % \
            (epoch,num_epochs,loss.item(), train_auc, test_loss.item(), test_auc))

### 2 edge-type prediction ###
else:

    print('Starting training 2 link prediction: disease-gene, disease-chem...')
    k = out_feat
    opt = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)
    num_epochs = num_epochs

    for epoch in range(1, num_epochs + 1):
        
        # Make 2 negative graphs for 2 types of links: disease-gene, drug-disease
        dg_neg_g = construct_negative_graph(train_graph, k, ('disease', 'relate', 'gene'))
        dc_neg_g = construct_negative_graph(train_graph, k, ('drug', 'treats', 'disease'))
        
        # calculate 2 losses
        dg_pos_score, dg_neg_score = model(train_graph, dg_neg_g, ('disease', 'relate', 'gene'))
        dc_pos_score, dc_neg_score = model(train_graph, dc_neg_g, ('drug', 'treats', 'disease')) 
        pos_score = torch.cat((dg_pos_score, dc_pos_score), axis=0)
        neg_score = torch.cat((dg_neg_score, dc_neg_score), axis=0)
        
        loss_dg = compute_loss(dg_pos_score, dg_neg_score)
        loss_dc = compute_loss(dc_pos_score, dc_neg_score)
        
        loss = (loss_dg + loss_dc) / 2
        
        opt.zero_grad()
        loss.backward()
        opt.step()
                
        # get validation scores
        if epoch % 10 == 0:
            with torch.no_grad():
                dg_neg_test_graph = construct_negative_graph(test_graph, k, ('disease', 'relate', 'gene'))
                dc_neg_test_graph = construct_negative_graph(test_graph, k, ('drug', 'treats', 'disease'))

                dg_test_pos, dg_test_neg = model(test_graph, dg_neg_test_graph, ('disease', 'relate', 'gene'))
                dc_test_pos, dc_test_neg = model(test_graph, dc_neg_test_graph, ('drug', 'treats', 'disease'))

                test_loss_dg = compute_loss(dg_test_pos, dg_test_neg)
                test_loss_dc = compute_loss(dc_test_pos, dc_test_neg)
                test_loss = (test_loss_dg + test_loss_dc) / 2

                test_pos = torch.cat((dg_test_pos, dc_test_pos), axis=0)
                test_neg = torch.cat((dg_test_neg, dc_test_neg), axis=0)

                train_auc = compute_auc(pos_score, neg_score)
                test_auc = compute_auc(test_pos, test_neg)
                fpr, tpr, thresholds = compute_roc_curve(test_pos, test_neg)

                epochs.append(epoch)
                train_losses.append(loss.item())
                train_aucs.append(train_auc)
                test_losses.append(test_loss.item())
                test_aucs.append(test_auc)               

                print('Epoch %d/%d -- Train Loss: %0.3f, Train AUC: %0.3f, Test Loss: %0.3f, Test AUC: %0.3f' % \
            (epoch,num_epochs,loss.item(), train_auc, test_loss.item(), test_auc))    

print('Training time: {:.2f} seconds'.format(time.time() - start_time))

# Validation 

### 1 edge-type link validation ### 
with torch.no_grad():
    if link_prediction == 1:
        neg_test_graph = construct_negative_graph(test_graph, out_feat, ('disease', 'relate', 'gene'))
        test_pos, test_neg = model(test_graph, neg_test_graph, ('disease', 'relate', 'gene'))
        
        test_auc = compute_auc(test_pos, test_neg)
        fpr, tpr, thresholds = compute_roc_curve(test_pos, test_neg)
    else:
        dg_neg_test_graph = construct_negative_graph(test_graph, k, ('disease', 'relate', 'gene'))
        dc_neg_test_graph = construct_negative_graph(test_graph, k, ('drug', 'treats', 'disease'))      
        dg_test_pos, dg_test_neg = model(test_graph, dg_neg_test_graph, ('disease', 'relate', 'gene'))
        dc_test_pos, dc_test_neg = model(test_graph, dc_neg_test_graph, ('drug', 'treats', 'disease'))      
        test_pos = torch.cat((dg_test_pos, dc_test_pos), axis=0)
        test_neg = torch.cat((dg_test_neg, dc_test_neg), axis=0)  

        test_auc = compute_auc(test_pos, test_neg)
        fpr, tpr, thresholds = compute_roc_curve(test_pos, test_neg)          


fig = plt.figure()
lw = 2
plt.plot(epochs, train_losses, color='darkorange', lw=lw, label='Train Loss')
plt.plot(epochs, test_losses, color='navy', lw=lw,label='Validation Loss')
plt.plot(epochs, test_aucs, color='black', lw=lw,label='Validation AUC Score')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.ylim(0, 1)
plt.legend(loc="center right")
plt.show()
fig.savefig('loss.png',dpi=1024,format='png')

fig = plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % test_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('ROC (Receiver Operating Characteristic) curve')
plt.show() 

fig.savefig('ROC.png',dpi=1024,format='png')




