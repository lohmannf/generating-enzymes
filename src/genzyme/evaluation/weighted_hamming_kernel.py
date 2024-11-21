import numpy as np
import torch
import blosum as bl

def weighted_hamming_kernel(seqs1, seqs2):
    '''Returns a matrix of pairwise hamming distance weighted by a modified blosum matrix'''

    # modify blosum matrix:
    # scale all entries between 0 and 1
    # assign 0 to retention of same amino acid
    # assign highest weight to smallest entry in blosum matrix
    wmat = bl.BLOSUM(90)
    for aa in wmat.keys():
        for col in wmat.keys():
            if aa == col:
                wmat[aa][col] = 0
            else:
                wmat[aa][col] = (-1*wmat[aa][col]+6)/12

    K = np.zeros((len(seqs1), len(seqs2)))

    for i, seq_x in enumerate(seqs1):
            for j, seq_y in enumerate(seqs2):
                # Vectorized computation of scores for each pair of amino acids in the sequences
                K[i, j] = sum(wmat[aa_x][aa_y] for aa_x, aa_y in zip(seq_x, seq_y))

        
    return torch.from_numpy(1/(K+1))
