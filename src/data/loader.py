import os
import pandas as pd
import numpy as np
import mavenn

class BaseLoader():
    '''
    Base class for data loaders, outlines the basic functionality
    
    Extend with new class for every dataset
    '''

    def __init__(self):
        self.data = None

    #TODO: Function to tokenize sequence strings for language models
    #TODO: Function to perform train/test splitting and transform to torch tensors



class GB1Loader(BaseLoader):
    '''
    Data loader for GB1 dataset from Olson et al., 2014
    Essentially a wrapper for mavenn loader
    '''

    def __init__(self):
        super().__init__()

    def load(self):
        self.data =  mavenn.load_example_dataset("gb1").x.to_numpy()


class IRedLoader(BaseLoader):
    ''' 
    Data loader for the Imine Reductase dataset from Gantz et al., 2024
    Uses only active sequences
    '''

    def __init__(self):
        super().__init__()

    def load(self, rel_path: str = "/../../data/IRed/srired_active_data.csv"):
        ''' 
        Parameters
        -----------
        rel_path: str
            The path to the csv file containing the data relative to loader.py
        
        Returns
        -------
        None
        '''
        
        path = os.path.dirname(__file__) + rel_path
        self.data = pd.read_csv(path).aa_seq.to_numpy()
        self._strip()

    
    def _strip(self):
        '''Remove any asterisk characters from the sequences in self.data'''

        if self.data is None:
            raise ValueError("Read data first by calling self.load()")

        self.data = np.array([seq.replace("*", "") for seq in self.data])
        