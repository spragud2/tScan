import numpy as np
import itertools
from itertools import product
from seekr.kmer_counts import BasicCounter
from seekr.fasta_reader import Reader
from tqdm import tqdm_notebook as tqdm
from seekr.fasta_reader import Reader
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby
from operator import itemgetter
from collections import defaultdict

import pickle

def classify(seq, k, lrTab):
    """ Classify seq using given log-ratio table.  We're ignoring the
        initial probability for simplicity. """
    seq = seq.upper()
    bits = 0
    nucmap = { 'A':0, 'T':1, 'C':2, 'G':3 }
    rowmap = dict(zip([''.join(p) for p in product(['A','T','C','G'],repeat=k-1)],range(4**(k-1))))
    for kmer in [seq[i:i+k] for i in range(len(seq)-k+1)]:
        if ('N' not in kmer) and ('$' not in kmer):
            i, j = rowmap[kmer[:k-1]], nucmap[kmer[-1]]
            #print(f'P({kmer[-1]}|{kmer[:-1]})',lrTab[i,j])
            bits += lrTab[i, j]
    return bits

def markov_chain(kmers,k,bases):
    
    conds = np.zeros((4**(int(k)-1), 4), dtype=np.float64)
    
    margs = np.zeros(4, dtype=np.float64)
    
    for i, ci in enumerate([''.join(p) for p in itertools.product(bases,repeat=k-1)]):
        
        tot = 0
        
        for j, cj in enumerate('ATCG'):
            
            count = kmers[ci+cj]
            tot += count
        
        if tot > 0:
            
            for j, cj in enumerate('ATCG'):
                
                conds[i, j] = kmers[ci+cj] / float(tot)

    return conds

def setCoords(hits):
    coords = [np.arange(start=i*s,stop=i*s+w,step=1) for i in hits]
    if isinstance(coords[0],list):
        mergedCoords = np.concatenate(coords,axis=None)
    else:
        mergedCoords = coords
    flatCoords = np.unique(mergedCoords)
    flatCoords = list(flatCoords)
    finalCoords = []
    for k, g in groupby(enumerate(flatCoords),lambda kv:kv[0]-kv[1]):
        finalCoords.append(list(map(itemgetter(1), g)))
    return finalCoords

def nsmall(a, n):
    return np.partition(a, n)[n]


k = 4

BCD = np.load('./mamBCD4.mkv.npy')
AE = np.load('./mamAE4.mkv.npy')

mm10Genes = Reader('gencodeGenesMm10.fa')
mm10Seq = mm10Genes.get_seqs()
mm10Head = mm10Genes.get_headers()
w=200
s=20
seqMap = defaultdict(list)

initModelMap = {'BCD':BCD,'AE':AE}
for head,seq in tqdm(zip(mm10Head,mm10Seq),total=len(mm10Seq)):
    tiles = [seq[i:i+w] for i in range(0,len(seq)-w+1,s)]
    mapMm10BCD = np.array([classify(tile,k,initModelMap['BCD']) for tile in tiles])
    mapMm10AE = np.array([classify(tile,k,initModelMap['AE']) for tile in tiles])
    whereHitBCD = np.array(np.nonzero(mapMm10BCD < 0))
    whereHitAE = np.array(np.nonzero(mapMm10AE < 0))
    whereHit = np.unique(np.concatenate((whereHitBCD,whereHitAE),axis=None))
    if whereHit.size > 0:
        coords = setCoords(whereHit)
        for consecCoords in coords:
            seqMap[head].append(''.join([seq[c] for c in consecCoords]))
pickle.dump(seqMap,open('./seqMapID.p','wb'))

