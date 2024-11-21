import numpy as np

class SpecialTokens():

    def __init__(self,
                 start: str = None,
                 end: str = None,
                 pad: str = None,
                 eot: str = None,
                 sep: str = None,
                 space: str = None,
                 unk: str = None
                 ):
        
        self.start = start
        self.end = end
        self.pad = pad
        self.eot = eot
        self.sep = sep
        self.space = space
        self.unk = unk


AA_DICT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 
           'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 
           'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

def aa2int_single(sequence: str, vocab_map: dict = AA_DICT):
    return [vocab_map[aa] for aa in list(sequence)]

def aa2int(sequences: list, vocab_map: dict = AA_DICT):
    return np.array([aa2int_single(seq, vocab_map) for seq in sequences])

def int2aa_single(seq: str, reverse_map: dict):
    return ''.join(reverse_map[aa] for aa in seq)

def onehot2aa(sequences: list, vocab_map: dict = AA_DICT):
    aas = np.array(list(vocab_map.keys()))[np.argsort(list(vocab_map.values()))]
    return [''.join(aas[pos][0] for pos in seq.bool().numpy()) for seq in sequences]
