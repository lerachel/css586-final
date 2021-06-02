import pandas as pd
import torch
import numpy as np
import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

def buildGraph():
    A = pd.read_csv("data/DCh-Miner_miner-disease-chemical.tsv", sep="\t")

    B = pd.read_csv("data/CTD_diseases_pathways.tsv", sep="\t")

    C = pd.read_csv("data/DG-Miner_miner-disease-gene.tsv", sep="\t")

    #convert diseases to order set for ID'ing
    diseaseA = A['# Disease(MESH)'].tolist()

    diseaseA = list(set(diseaseA))

    diseaseB = B['DiseaseID'].tolist()

    diseaseB = list(set(diseaseB))

    diseaseC = C['# Disease(MESH)'].tolist()

    diseaseC = list(set(diseaseC))

    disease = list(set(diseaseA + diseaseB + diseaseC))

    #convert chemicals to ordered set for ID'ing
    chem = A['Chemical'].tolist()

    chem = list(set(chem))

    #convert pathways to ordered set for ID'ing
    pathway = B['PathwayID'].tolist()
    pathway = list(set(pathway))

    #convert genes to ordered set for ID'ing
    gene = C['Gene'].tolist()
    gene = list(set(gene))

    #create map for edge list generation and future lookup
    diseaseMap = {disease[i]: i for i in range(len(disease))}
    chemMap = {chem[i]: i for i in range(len(chem))}
    pathMap = {pathway[i]: i for i in range(len(pathway))}
    geneMap = {gene[i]: i for i in range(len(gene))}

    maps = {'disease': diseaseMap,
                'chemical': chemMap,
                'pathway': pathMap,
                'gene': geneMap}
    
    DCh = A.to_numpy()

    for i in range(np.shape(DCh)[0]):
        DCh[i,0] = diseaseMap[DCh[i,0]]
        DCh[i,1] = chemMap[DCh[i,1]]

    DPt = B.drop(columns=['# DiseaseName','PathwayName','InferenceGeneSymbol']).to_numpy()

    for i in range(np.shape(DPt)[0]):
        DPt[i,0] = diseaseMap[DPt[i,0]]
        DPt[i,1] = pathMap[DPt[i,1]]

    DGn = C.to_numpy()

    for i in range(np.shape(DGn)[0]):
        DGn[i,0] = diseaseMap[DGn[i,0]]
        DGn[i,1] = geneMap[DGn[i,1]]
    
    return DCh, DPt, DGn, maps

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})
