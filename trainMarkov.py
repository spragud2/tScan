
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

# Null Model
genomeRawCounts = BasicCounter('/Users/danielsprague/Downloads/gencode.vM21.transcripts.fa',k=k,mean=False,std=False,log2=False,alphabet='ATCG')
genomeFa = Reader('/Users/danielsprague/Downloads/gencode.vM21.transcripts.fa')
genomeSeqs = genomeFa.get_seqs()
genomeSeqLens = [len(i) for i in genomeSeqs]
genomeRawCounts.get_counts()
unNormGenomeCounts = genomeRawCounts.counts.T*genomeSeqLens/1000
genomeCounts = np.rint(unNormGenomeCounts.T)
weightedAvgGenomeCounts = np.average(genomeCounts,weights=genomeSeqLens,axis=0)

kmers = [''.join(p) for p in itertools.product('ATCG',repeat=k)]
curr_kmers = dict(zip(kmers,weightedAvgGenomeCounts))
genome_avg = markov_chain(curr_kmers,k,'ATCG')

np.save(f'./genome{k}.mkv.npy',genome_avg)

# initModel
initModelMap = {}
for modelName in ['BCD','AE']:
    q = BasicCounter(k=k,mean=False,std=False,log2=False,alphabet='ATCG')
    q.seqs = Reader(f'./{modelName}.fa').get_seqs()
    qSeqRelLen = np.array([len(i) for i in q.seqs])/sum([len(i) for i in q.seqs])
    q.get_counts()
    unNorm = q.counts.T*[len(s) for s in q.seqs]/1000
    queryCounts = np.rint(unNorm.T)
    queryCounts += 1
    weightedAvgCounts = np.average(queryCounts,axis=0)
    kmers = [''.join(p) for p in itertools.product('ATCG',repeat=k)]
    curr_kmers = dict(zip(kmers,weightedAvgCounts))
    qTransitionMats = markov_chain(curr_kmers,k,'ATCG')
    print('Row Sums: ',np.sum(qTransitionMats,axis=1))
    lgTbl = np.log2(qTransitionMats) - np.log2(genomeInit)
    initModelMap[modelName] = lgTbl
# curr_kmers = dict(zip(kmers,queryCounts[0]))
# qConcat = markov_chain(curr_kmers,k,'ATCG')

np.save(f'./BCD{k}.mkv.npy',initModelMap['BCD'])
np.save(f'./AE{k}.mkv.npy',initModelMap['AE'])
