'''
graph_models.py
1/4 Python files submitted for CSS 586 final
Chip Kirchner
6/8/2021

Base models were developed jointly by partners Chip Kirchner and Rachel Le unless otherwise noted in the comments below
'''

import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

#Cosine Similarity or dot product prediction model adapted from Deep Graph Library
#Reference: https://docs.dgl.ai/en/0.6.x/guide/training-link.html
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

#Prediction model adapted from Deep Graph Library
#Code implementation and sequential model written by Chip Kirchner
class MLPPredictor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.W = nn.Sequential(
            nn.Linear(in_features * 2,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def apply_edges(self, edges):
        u = edges.src['h']
        v = edges.dst['h']
        score = self.W(torch.cat([u, v], 1))
        return {'score': score}

    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges,etype=etype)
            return graph.edges[etype].data['score']

#Embedding model adapted from Deep Graph Library
#Reference: https://docs.dgl.ai/en/0.6.x/guide/training-link.html
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
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

#3 Convolution GCN Model adapted from above by Chip Kirchner
class RGCN3conv(RGCN):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__(in_feats, hid_feats, out_feats, rel_names)

        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats*2)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats*2, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        return h

#5 Convolution GCN Model adapted from above by Chip Kirchner
class RGCN5conv(RGCN):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__(in_feats, hid_feats, out_feats, rel_names)

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats*2)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats*2, hid_feats*4)
            for rel in rel_names}, aggregate='sum')
        self.conv4 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats*4, hid_feats*2)
            for rel in rel_names}, aggregate='sum')
        self.conv5 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats*2, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv4(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv5(graph, h)
        return h

#GraphSAGE Model application adapted from above by Chip Kirchner
class RGCNSAGE(RGCN):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__(in_feats, hid_feats, out_feats, rel_names)

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats, hid_feats,'pool')
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats, hid_feats*2,'pool')
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats*2, out_feats,'pool')
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        return h

#GraphSAGE mixed with GCN Model application adapted from above by Chip Kirchner
class RGCNMix(RGCN):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__(in_feats, hid_feats, out_feats, rel_names)

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats*2)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats*2, out_feats,'pool')
            for rel in rel_names}, aggregate='sum')
    
    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        return h
