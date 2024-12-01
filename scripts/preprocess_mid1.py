from Bio import SeqIO
from Bio.Seq import Seq
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from mutedpy.experiments.od1.loader import get_wiltype_seq
from mutedpy.experiments.od1.loader import filter_seq # , load_data_pickle
import torch 

def load_sequences(filepath):
    """Load all sequences from a FASTA file."""
    with open(filepath, "r") as file:
        return [record.seq for record in SeqIO.parse(file, "fasta")]

def compare_sequences(wildtype, mutant):
    """Compare two sequences and return a list of mutated positions."""
    return [i for i, (wt, mt) in enumerate(zip(wildtype, mutant)) if wt != mt]

def convert_dna_to_protein(dna):
    my_seq = Seq(dna)
    prot_seq = str(my_seq.translate())
    prot_seq = prot_seq[0:-1]
    prot_seq = prot_seq.replace('*', '<mask>')
    return prot_seq

def convert_to_protein(seqs):
    r = []
    for s in seqs:
        r.append(convert_dna_to_protein(s))
    return r

def get_percentages(sequences,wt_seq, dna = True, unique = False, normalized = True):
    counts = np.zeros(len(wt_seq))
    wt_arr = np.array(list(wt_seq))
    total_count = 0
    # Perform element-wise comparison and convert boolean results to integers (1s and 0s)

    if unique: 
        # remove duplicates 
        print ("No seq:",len(sequences))
        sequences = list(set(sequences))
        print ("No unique seqs:", len(sequences))
    else:
        print ("No seqs:", len(sequences))
        pass
    
    for dna_seq in sequences:
        if dna:
            seq = convert_dna_to_protein(dna_seq)
        else:
            seq = dna_seq
        arr = np.array(list(seq))
        
        if len(arr)==97:
            comparison_array = (arr != wt_arr).astype(int)
            counts += comparison_array
            total_count += 1
            
    # Calculate the percentages
    if normalized:
        percentages = 100 * counts / total_count
    else:
        percentages = counts
    return percentages


if __name__ == "__main__":

    raw_data_path = "../mid/"
    
    max_sort = 9
    unique = False
    sorts_evo_1 = [2,3,4]
    sorts_evo_2 = [6,7,8]

    # load wiltype 

    y_vals = []
    seqs = []

    sort_1_values = [1,1.2,3]
    sort_2_values = [4.,5.,6.]

    print ("Evo1")
    for index, i in enumerate(sorts_evo_1):
        sequences = convert_to_protein(load_sequences(raw_data_path+str(i)+".fasta")) #path to dir with fasta files
        print ("sort:"+str(index+1), len(sequences), "unique:", len(set(sequences)))
        seqs = seqs + sequences
        y_vals = y_vals + [sort_1_values[index]]*len(sequences) 

    y_vals2 = []
    seqs2 = []
    print ("Evo2")
    for index, i in enumerate(sorts_evo_2):
        sequences = convert_to_protein(load_sequences(raw_data_path+str(i)+".fasta")) #path to dir with fasta files
        print ("sort:"+str(index+1), len(sequences), "unique:", len(set(sequences)))
        seqs2 = seqs2 + sequences
        y_vals2 = y_vals2 + [sort_2_values[index]]*len(sequences) 

        #percentages = get_percentages(sequences,wt_seq)
        #results_unique.append(get_percentages(sequences,wt_seq, unique = True))
        #results.append(percentages)
        #y_vals = 

    evolution = [1]*len(seqs) + [2]*len(seqs2)

    dts = pd.DataFrame({'seq':seqs+ seqs2,'val':y_vals+y_vals2,'evo':evolution}) 
    if unique:
        dts = dts.drop_duplicates()
    else:
        pass 

    evolution_1_data = dts[dts['evo']==1]
    evolution_2_data = dts[dts['evo']==2]

    #filter_seq(evolution_1_data['seq'],y)
    y1 = evolution_1_data['val']>=1.
    y2 = evolution_1_data['val']>=1.2
    y3 = evolution_1_data['val']>=3.
    seq = evolution_1_data['seq']
    y = np.concatenate([y1.values.reshape(-1,1),y2.values.reshape(-1,1),y3.values.reshape(-1,1)], axis = 1)


    #filter_seq(evolution_1_data['seq'],y)
    y1 = evolution_2_data['val']>=4.
    y2 = evolution_2_data['val']>=5
    y3 = evolution_2_data['val']>=6.
    seq2 = evolution_2_data['seq']
    y2 = np.concatenate([y1.values.reshape(-1,1),y2.values.reshape(-1,1),y3.values.reshape(-1,1)], axis = 1)

    new_seq = []
    new_y = []
    no_sorts = 3
    new_seqs = pd.DataFrame(seq.values, columns=['variant'])
    for i in range(3):
        new_seqs[str(i)] = y[:, i]
    new_seqs['occ'] = 1.

    agg_dict = {}
    agg_list = [{str(i) : 'sum'} for i in range(no_sorts)] + [{'occ':'sum'}]
    for d in agg_list:
        agg_dict.update(d)

    #if unique:
    new_seqs = new_seqs.groupby('variant').agg(agg_dict).reset_index()
    #else:
    #new_seqs.agg(agg_dict).reset_index()

    def is_sorted_descending(lst):
        return all(lst[i] > lst[i + 1] or (lst[i+1]==0 and lst[i]==0) for i in range(len(lst) - 1))

    for index, row in new_seqs.iterrows():

        if is_sorted_descending([row[str(i)] for i in range(no_sorts)]):
        # print ("good.", row)

            yn = torch.zeros(size=(1, no_sorts)).bool()
            new_seq.append(row['variant'])
            ac = max([i if row[str(i)]>0 else 0 for i in range(no_sorts)])
            yn[0,0:ac+1] = 1.
            new_y.append(yn)
    #
    # # unique sequences
    # unique_seqs = list(set(new_seqs.values.tolist()))
    #
    # for index, seq in enumerate(unique_seqs):
    #     print (index, len(unique_seqs))
    #     mask = new_seqs == seq
    #     entries = y[mask, :]
    #
    #     occurences = entries.size()[0]
    #     #print (entries)
    #     #print (entries.sum(dim = 0))
    #     sums = entries.sum(dim = 0)
    #     if all([sums[i] == max(0,occurences-i) for i in range(no_sorts)]):
    #         # print ('------------------')
    #         # print("true entry.")
    #         # print('------------------')
    #         for i in range(occurences):
    #             new_seq.append(seq)
    #         new_y.append(entries)
    #

    new_y = torch.vstack(new_y)
    print ("Filtering finished:")
    for i in range(3):
        print (torch.sum(new_y[:, i]))


    seq, y = filter_seq(seq,torch.from_numpy(y))
    seq2, y2 = filter_seq(seq2,torch.from_numpy(y2), no_mutants = 10)

    data1 = pd.DataFrame(data = y)
    data2 = pd.DataFrame(data = y2)

    data1["sequences"] = seq
    data2["sequences"] = seq2
    data1["round"] = 1
    data2["round"] = 2

    data = pd.concat([data1, data2], axis=0, ignore_index=True)
    data.to_csv("../data/MID1/mid_filtered.csv")
