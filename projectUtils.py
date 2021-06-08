'''
projectUtils.py
2/4 Python files submitted for CSS 586 final
Chip Kirchner
6/8/2021

Base models and utility functions were developed jointly with partners Chip Kirchner and Rachel Le unless otherwise noted in the comments below
'''
import pandas as pd
import torch
import numpy as np
import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

#function returns the three data sets used to build the graph for main code
def buildGraph():
    print("Pre-processing Graph Edge Lists...\n")

    A = pd.read_csv("data/DCh-Miner_miner-disease-chemical.tsv", sep="\t")

    B = pd.read_csv("data/CTD_diseases_pathways.tsv", sep="\t")

    C = pd.read_csv("data/DG-Miner_miner-disease-gene.tsv", sep="\t")

    #convert diseases to order set for ID'ing
    diseaseA = A['# Disease(MESH)'].tolist()
    #convert chemicals to ordered set for ID'ing
    chem = A['Chemical'].tolist()
    #remove duplicates by converting to set
    chem = list(set(chem))
    #get the raw number of edges in list
    num_edges = len(diseaseA)
    #remove duplicates by converting to set
    diseaseA = list(set(diseaseA))

    print("\t Loading disease chemical edge list...")
    print("\t %d Edges, %d Diseases, %d Drugs " % (num_edges,len(diseaseA),len(chem)))    


    diseaseB = B['DiseaseID'].tolist()
    num_edges = len(diseaseB)
    diseaseB = list(set(diseaseB))
    #convert pathways to ordered set for ID'ing
    pathway = B['PathwayID'].tolist()
    pathway = list(set(pathway))

    print("\t Loading disease-pathway edge list...")
    print("\t %d Edges, %d Diseases, %d Pathways " % (num_edges,len(diseaseB),len(pathway))) 


    diseaseC = C['# Disease(MESH)'].tolist()
    num_edges = len(diseaseC)
    diseaseC = list(set(diseaseC))
    #convert genes to ordered set for ID'ing
    gene = C['Gene'].tolist()
    gene = list(set(gene))
    
    print("\t Loading disease-gene edge list...")
    print("\t %d Edges, %d Diseases, %d Genes \n" % (num_edges,len(diseaseC),len(gene))) 

    #outer join all three lists
    disease = list(set(diseaseA + diseaseB + diseaseC))
    
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

#function to build inverse graph adapted from Deep Graph Library
#Reference: https://docs.dgl.ai/en/0.6.x/guide/training-link.html
#build a graph of 'negative' link samples for link prediction classifier
def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})
