import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from genzyme.models.basemodel import BaseModel
from genzyme.data.utils import AA_DICT

class UniformRandomModel(BaseModel):
    ''' 
    A model that samples each position uniformly at random from the marginals of the training data.
    Without training, defaults to a completely random model.
    Can be used as a baseline for more complex models.
    '''

    def __init__(self, seed: int = 31, length: int = 290):
        
        self.seed = seed
        np.random.seed(seed)

        self.rng = np.random.default_rng()
        self.frequency = np.ones((length, len(AA_DICT)))/len(AA_DICT)
        self.freq_idx = np.array(list(AA_DICT.keys()))
        self.l = length

    def pad_data(self, data: np.ndarray):
        '''Append padding tokens so that all sequences in data have the same length'''

        lens = [len(seq) for seq in data]
        max_len = max(lens)
        
        return np.array([seq + (max_len - lens[i])*'$' for i, seq in enumerate(data)])


    def run_training(self,
        train_dataset: str | np.ndarray):
        '''
        Learn the empirical marginal distribution of each position in the sequence

        Parameters
        ----------
        train_dataset: str | np.ndarray
            The training sequences or the path to a file that stores them

        Returns
        -------
        None
        '''

        if isinstance(train_dataset, str):
            pass
        #TODO: read data from file

        freqs = pd.DataFrame()
        train_dataset = self.pad_data(train_dataset)

        for pos in tqdm(range(self.l), desc = 'Learning positions'):
            aas = [seq[pos] for seq in train_dataset if len(seq) > pos]
            aa, cts = np.unique(aas, return_counts = True)
            freqs = pd.concat([freqs, pd.DataFrame([cts / sum(cts)], columns = aa)], ignore_index = True)
        
        self.freq_idx = freqs.columns.values
        self.frequency = freqs.fillna(0).to_numpy()
        

    def generate(self, 
                 n_samples: int,
                 output_file: str = None,
                 keep_in_memory: bool = True):
        '''
        Generate new sequences with the trained model

        Parameters
        ----------
        n_samples: int
            How many samples to generate

        ouput_file: str
            Location to save the outputs to. Will not
            write results to file if None, default = None

        keep_in_memory: bool
            Whether to keep the generated sequences in memory and
            return them, set to False if n_samples is large.
            Default = True

        Returns
        -------
        seqs: list
            Will be empty if keep_in_memory = False
        '''

        if self.frequency is None:
            raise ValueError('Train model first')

        seqs = []

        for i in tqdm(range(n_samples), desc = "Generating sequences"):
            seq = ''
            avg_lh = 0

            for freqs in self.frequency:
                
                sample_msk = self.rng.multinomial(1, freqs).astype(bool)
                seq += str(self.freq_idx[sample_msk][0])
                avg_lh += freqs[sample_msk][0]

                # stop generation if padding token is sampled
                if seq[-1] == '$':
                    seq = seq[:-1]
                    break
            
            if keep_in_memory:
                seqs.append(seq)
            avg_lh /= len(seq)

            if output_file is not None:
                with open(output_file, "a") as file:
                    file.write(f'>{avg_lh}\n{seq}\n')

        return seqs
