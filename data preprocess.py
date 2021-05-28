import pandas as pd
import torch
import numpy as np

#take TSV
#create set for Disease nodes
#assigned ID as Key and loc (1-N) as value
    #loc
    #features
    #???
#create set for Drug nodes
#assign ID as key and loc (1-N) as value

#take sets, create edge list using 1-N and feed into DGL dataset?? or just create hetero graph with set. 

A = pd.read_csv("DCh-Miner_miner-disease-chemical.tsv", sep="\t")

print(list(A))

#convert diseases to order set for ID'ing
disease = A['# Disease(MESH)'].tolist()
print("# of entries disease column of edge list: %d" % len(disease))
disease = list(set(disease))
print("Convert to Set...")
print("# distinct diseases: %d" % len(disease))

#convert chemicals to ordered set for ID'ing
chem = A['Chemical'].tolist()
print("# Chemical entries in edge list: %d" % len(chem))
chem = list(set(chem))
print("Convert to set...")
print("# of distinct chemicals: %d" % len(chem))

#create map for edge list generation and future lookup
diseaseMap = {disease[i]: i for i in range(len(disease))}
chemMap = {chem[i]: i for i in range(len(chem))}

tmp = A.to_numpy()
print(tmp[0])

print(np.shape(tmp[:,0]))
for i in range(np.shape(tmp)[0]):
    tmp[i,0] = diseaseMap[tmp[i,0]]
    tmp[i,1] = chemMap[tmp[i,1]]

print(tmp[:,0])
