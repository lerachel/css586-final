#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


# In[3]:


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


# In[4]:


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


# In[5]:


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


# In[6]:


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


# In[7]:


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


# In[8]:


def compute_roc_curve(pos_score, neg_score):
    scores = torch.cat([pos_score.view(-1,), neg_score.view(-1,)]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_curve(labels, scores, pos_label=1)


# In[9]:


# Open two tsv files, extract edge lists, combine datasets

print('Opening disease-gene & disease-chemical datasets from .tsv files...')

file1 = pd.read_csv('DG-Miner_miner-disease-gene.tsv','\t')
file2 = pd.read_csv('DCh-Miner_miner-disease-chemical.tsv', sep='\t')
file = np.concatenate( (file1.to_numpy(), file2.to_numpy()), axis=0)


# In[10]:


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


# In[11]:


'''# Create lookup table
print('Create lookup tables for disease, gene and chemicals...')
disease_map = {unique_disease[i]: i for i in range(len(unique_disease))}
gene_map =       {unique_gene[i]: i for i in range(len(unique_gene))}
chem_map =       {unique_chem[i]: i for i in range(len(unique_chem))}
'''


# In[12]:


# Create numerical labels for disease, gene, chemical
print('Applying Label Encoding for disease, gene and chemicals...')

disease_encoder = LabelEncoder()
gene_encoder = LabelEncoder()
chem_encoder = LabelEncoder()

encoded_disease = disease_encoder.fit_transform(disease)
encoded_gene = gene_encoder.fit_transform(gene)
encoded_chem = chem_encoder.fit_transform(chem)


# In[13]:


# Create new datasets with encoded labels

print('Creating new datasets with encoded labels...')

encoded_gene = encoded_gene.reshape(-1,1)
encoded_chem = encoded_chem.reshape(-1,1)
encoded_disease = encoded_disease.reshape(-1,1)
dataset1 = np.hstack((encoded_disease[:len(file1)], encoded_gene))
dataset2 = np.hstack((encoded_disease[len(file1):], encoded_chem))


# In[14]:


# train test split here
# Need equal amount of training and testing samples for both types of links
# Allocate 0.5 for each types of link

print('Applying train_test_split, taking 0.5 from disease-gene and 0.5 from disease-chem for testing...')
# train1, test1 are disease-gene; train2, test2 are disease-chem
train1, test1 = train_test_split(dataset1, test_size = 0.5, random_state = 101)
train2, test2 = train_test_split(dataset2, test_size = 0.5, random_state = 101)


# In[15]:


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


# In[24]:


print("#### Train graph's metadata #### \n{}\n".format(train_graph))
print("#### Test graph's metadata #### \n{}\n".format(test_graph))


# In[ ]:


print('Starting training...')

test_losses = []
train_losses = []
test_aucs = []
train_aucs = []

in_features, hidden_features, out_features = 15, 5, 5
print('in_features = {}'.format(in_features))
print('hidden_features = {}'.format(hidden_features))
print('out_features = {}'.format(out_features))

k = 5
model = Model(train_graph, 15, 15, 5)
opt = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)
num_epochs = 10 #300
for epoch in range(num_epochs):
    
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
    loss = loss_dg + loss_dc
    
    opt.zero_grad()
    loss.backward()
    train_loss = loss.item()
    opt.step()
    
    print(epoch)
    
    with torch.no_grad():
        
        # Testing
        
        dg_neg_test_graph = construct_negative_graph(test_graph, k, ('disease', 'relate', 'gene'))
        dc_neg_test_graph = construct_negative_graph(test_graph, k, ('drug', 'treats', 'disease'))
        
        dg_test_pos, dg_test_neg = model(test_graph, dg_neg_test_graph, ('disease', 'relate', 'gene'))
        dc_test_pos, dc_test_neg = model(test_graph, dc_neg_test_graph, ('drug', 'treats', 'disease'))
        
        test_loss_dg = compute_loss(dg_test_pos, dg_test_neg)
        test_loss_dc = compute_loss(dc_test_pos, dc_test_neg)
        test_loss = test_loss_dg + test_loss_dc
        
        test_pos = torch.cat((dg_test_pos, dc_test_pos), axis=0)
        test_neg = torch.cat((dg_test_neg, dc_test_neg), axis=0)
        
        auc = compute_auc(test_pos, test_neg)
        train_auc = compute_auc(pos_score, neg_score)
        
        # get validation/test scores
        if epoch % 5 ==0:
            print('Epoch %d/%d -- Training Loss: %0.3f, Test Loss: %0.3f,AUC: %0.3f' % (epoch,num_epochs,train_loss, test_loss.item(), auc))
    


# In[ ]: