import torch
import numpy as np
import warnings

from genzyme.data.utils import AA_DICT, aa2int_single, int2aa_single

class ProteinTokenizer:

    def __init__(self, vocab_map: dict = AA_DICT, group: bool = False, has_sep: bool = True):
        self.vocab_map = vocab_map
        self.group = group
        self.has_sep = has_sep

        if group and not has_sep:
            raise ValueError(f"Cannot group data without separator token")


    def remove_unknown(self, seq: str):
        return ''.join([aa for aa in seq if aa in self.vocab_map.keys()])
    

    def encode(self, seq: str):
        '''Encode a sequence and add eos token'''

        seq_enc = aa2int_single(self.remove_unknown(seq), self.vocab_map)
        return torch.Tensor(seq_enc + [len(self.vocab_map)])
    
    
    def batch_encode(self, batches):
        enc_batches = []
        for batch in batches:
            enc_batches.append([self.encode(seq) for seq in batch])
            
        return torch.Tensor(enc_batches)
        
    
    
    def batch_decode(self, batches):

        seqs = []
        if isinstance(batches, torch.Tensor):
            batches = batches.numpy()

        flatten = len(batches.shape) == 2
        if flatten:
            batches = np.expand_dims(batches, axis=0)

        reverse_map = {v: k for k,v in self.vocab_map.items()}

        if self.has_sep:
            SEP = "<sep>"
            reverse_map[len(reverse_map)] = SEP
        else:
            SEP = ""


        for batch in batches:
            if self.group:
                dec_batch = []
                for sample in batch:
                    tmp = int2aa_single(sample, reverse_map)
                    # drop first and last part (can be truncated)
                    tmp_split = tmp.split(SEP)[1:]
                    if tmp[-1] != SEP and len(tmp_split) > 1:
                        tmp_split = tmp_split[:-1]
                    
                    dec_batch += tmp_split
            
            else:
                # remove padding tokens
                dec_batch = [int2aa_single(sample, reverse_map).replace(SEP, "") for sample in batch]
            
            seqs.append(dec_batch)

        return seqs[0] if flatten else seqs
        

        


        



    