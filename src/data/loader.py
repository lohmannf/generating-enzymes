import os
import pandas as pd
import numpy as np
import mavenn
import random
import math
from datasets import Dataset, DatasetDict
from typing import Callable
import torch
from torch.utils.data import DataLoader as TorchDL, DistributedSampler, TensorDataset

from src.data.utils import SpecialTokens, aa2int_single, aa2int

class DataLoader():
    '''
    Base class for data loaders
    Implements load methods for different datasets
    Extend with new class for every model type
    '''

    DATASETS = ['gb1', 'ired', 'fasta', "entropy"]

    def __init__(self, seed: int = 31):
        self.data = None
        self.label = None
        self.step = None
        self.len_strategy = "median"
        self.block_sz = 0
        random.seed(seed)


    def get_data(self):
        return self.data
    
    @property
    def length(self):
        return 0 if self.data is None else len(self.data)
    

    def set_data(self, data: np.ndarray):
        '''
        Set the data in the dataloader manually from an array

        Parameters
        ----------
        data: np.ndarray
            An array of amino acid sequences

        Returns
        -------
        None
        '''

        self.data = data

    def set_label(self, label: np.ndarray):
        '''Set the some label of the data manually from an array
        
        Parameters
        ----------
        label: np.ndarray
            The labels, has to have the same length as self.data

        Returns
        ------
        None
        '''

        if len(label) != len(self.data):
            raise ValueError('label and self.data have different lengths')

        self.label = label


    def load(self, dataset: str, **kwargs):
        '''Loads the dataset into the DataLoader
        
        Parameters
        ----------
        dataset: str
            Name of the dataset

        **kwargs: Any
            Arguments passed to the loading function of the respective dataset

        Returns
        -------
        None
        '''
        
        if dataset == "gb1":
            self._load_gb1()
        
        elif dataset == "ired":
            self._load_ired(**kwargs)

        elif dataset == "fasta":
            self._load_fasta(**kwargs)
        
        elif dataset == "entropy":
            self._load_entropy(**kwargs)

        else:
            raise ValueError(f"Unknown dataset, must be one of {DataLoader.DATASETS}")

    
    def _load_gb1(self):
        '''Loads the GB1 dataset from Olson et al., 2014
        Essentially a wrapper for mavenn loader'''

        self.data =  mavenn.load_example_dataset("gb1").x.to_numpy()


    def _load_ired(self, rel_path: str = "/../../data/IRed/srired_active_data.csv", remove_ast: bool = False):
        ''' 
        Loads the Imine Reductase dataset from Gantz et al., 2024
        Uses only active sequences
  
        Parameters
        -----------
        rel_path: str
            The path to the csv file containing the data relative to loader.py

        remove_ast: bool
            Whether to remove asterisks in the sequences, default = False
        
        Returns
        -------
        None
        '''
        
        path = os.path.dirname(__file__) + rel_path
        raw_data = pd.read_csv(path)
        self.data = raw_data.aa_seq.to_numpy()
        self.len_strategy = "median"
        self.reference = 'MRDTDVTVLGLGLMGQALAGAFLKDGHATTVWNRSEGKAGQLAEQGAVLASSARDAAEASPLVVVCVSD'\
        'HAAVRAVLDPLGDVLAGRVLVNLTSGTSEQARATAEWAAERGITYLDGAIMAIPQVVGTADAFLLYSGPEAAYEAHEPTLRSLGAGTTY'\
        'LGADHGLSSLYDVALLGIMWGTLNSFLHGAALLGTAKVEATTFAPFANRWIEAVTGFVSAYAGQVDQGAYPALDATIDTHVATVDHLIH'\
        'ESEAAGVNTELPRLVRTLADRALAGGQGGLGYAAMIEQFRSPS*'

        self.fitness = raw_data.fitness.to_numpy()

        if remove_ast:
            self._replace("*")

    
    def _load_fasta(self, path: str, 
                    replace: dict = {},
                    append: bool = False, 
                    has_step: bool = False, 
                    extract_header: Callable = None):
        ''' 
        Loads the data from a fasta file

        Parameters
        ----------
        path: str
            Path to the fasta file

        replace: dict
            A dictionary with tokens and their replacements,
            default = {}

        append: bool
            Whether to append to self.data or replace it

        has_step: bool
            Whether the headers include the training step, default = False.
            Training step is assumed to be last token in header

        extract_header: Callable
            An optional function to extract a label from the fasta header
            of each sequence, default = None.
            Subclasses overlead this function by using a model-specific function

        Returns
        -------
        None
        '''
        
        if extract_header is not None:
            label = []

        if has_step:
            step = []

        data = []


        with open(path, "r") as file:
            lines = file.readlines()

        for line in lines:
            if not line.strip().startswith('>'):            
                data.append(line.strip())
            
            elif line.strip().startswith('>'):
                if extract_header is not None:
                    # extract a label from the fasta header
                    label.append(extract_header(line.strip()))

                if has_step:
                    # extract the training step from the header
                    step.append(line.strip().split()[-1])
        
        concat = append and self.data is not None
        self.data = np.concatenate((self.data, data)) if concat else np.array(data)
        if extract_header is not None:
            self.label = np.concatenate((self.label, label)) if concat else np.array(label)
        if has_step:
            self.step = np.concatenate((self.step, step)) if concat else np.array(step)

        # replace any characters
        for tok, repl in replace.items():
            self._replace(tok, repl)

    def _load_entropy(self, rel_path: str = "../../gen_data/scheduled_entropy/random.fasta"):

        path = os.path.join(os.path.dirname(__file__), rel_path)
        DataLoader._load_fasta(self, path = path)
        


    def _replace(self, character: str, replacement: str = ""):
        '''Replace all occurences of character from the sequences in self.data
            
        Parameters
        ----------
        character: str
        The character or token to replace

        replacement: str
        Replacement used for character, default = "" (removes it)

        Returns
        -------
        None
        '''

        if self.data is None:
            raise ValueError("Read data first by calling self.load()")

        self.data = np.array([seq.replace(character, replacement) for seq in self.data])


    def unify_seq_len(self, l: int = None):

        lens = np.array([len(seq) for seq in self.data])

        if l is None:
            if self.len_strategy=='median':
                l = int(np.median(lens))

            elif self.len_strategy=='min':
                l = min(lens)

            else:
                raise NotImplementedError(f'Unknown strategy {self.len_strategy}')


        self.data = self.data[lens >= l]
        self.data = np.unique([seq[:l] for seq in self.data])

    
    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop,
        # you can customize this part to your needs.
        if total_length >= self.block_sz:
            total_length = (total_length // self.block_sz) * self.block_sz
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + self.block_sz] for i in range(0, total_length, self.block_sz)]
                for k, t in concatenated_examples.items()
        }
        return result
        


class CTRLLoader(DataLoader):
    '''
    Data loader for CTRL-based models
    Based on code from https://huggingface.co/AI4PD/ZymCTRL
    '''

    def __init__(self, block_sz: int = 1024):
        
        super().__init__()
        self.tags = None
        self.tokenizer = None
        self.tags = None
        self.block_sz = block_sz


    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


    def assign_control_tags(self, tag: str | np.ndarray):
        '''Assign a control tag to each sequence
        
        Parameters
        ----------
        tag: str | np.ndarray
            Global control tag or control tag for each sequence

        Returns
        -------
        None
        
        '''

        if self.data is None:
            raise ValueError("Read data first by calling self.load()")
        
        if isinstance(tag, np.ndarray): 
            if self.data.shape != tag.shape:
                raise ValueError("tag must have same shape as self.data")
        
            self.tags = tag

        else:
            self.tags = np.repeat(tag, len(self.data))


    def insert_special(self, special: SpecialTokens):
        '''
        Concatenate control tags and sequences with inserted special tokens
        '''

        if self.data is None or self.tags is None:
            raise ValueError("Load data and assign control tags first")
        
        concat = []
        for tag, seq in zip(self.tags, self.data):
            # 4 for sep, start, end, eot
            ctrl_len = len(self.tokenizer(tag)['input_ids']) + 1
            space = self.block_sz - ctrl_len - 3

            if len(seq) > space:

                concat.append((ctrl_len + len(self.tokenizer(seq[:space])['input_ids']) + 1, tag + special.sep + seq[:space] + special.eot))

            else:

                concat.append((ctrl_len + len(self.tokenizer(seq)['input_ids']) + 3, tag + special.sep + special.start + seq + special.end + special.eot))

        return concat
    

    def _group(self, sequences):
        '''Group smaller sequences into blocks
        '''
        prev = None
        group = ''
        group_sz = 0

        for sz, seq in sequences:
            if prev is None or sz + group_sz <= self.block_sz:
                group += seq
                group_sz += sz

            else:
                group_sz = sz
                yield group
                group = seq

            prev = seq

        if group:
            group_sz = 0
            yield group


    def preprocess(self, 
                   special: SpecialTokens, 
                   test_frac: float, 
                   val_frac: float,
                   save: bool = False,
                   data_dir: str = '.'):
        
        '''
        Transform the data in the loader into huggingface dataset instances ready for training
        
        Parameters
        ---------
        special: SpecialTokens
            Holds the special tokens for the specific CTRL model

        test_frac: float
            Fraction of the data used for testing

        val_frac: float
            Fraction of the data used for validation

        save: bool
            Whether to save the resulting datasets to disk, default = False

        data_dir: str
            Directory to save datasets into, default = '.'


        Returns
        -------
        splits: DatasetDict
            Contains train, test and validation dataset
        '''

        if self.tokenizer is None:
            raise ValueError("Set tokenizer first")

        blocks = self.insert_special(special)
        random.shuffle(blocks)
        blocks = self._group(blocks)

        dataset = []
        # add padding to each block
        for bl in blocks:
            
            padding = special.pad * (self.block_sz - len(self.tokenizer(bl)['input_ids']))
            dataset.append(bl + padding)

        train_ds = Dataset.from_dict(
            {'text': dataset[:math.ceil(len(dataset)*(1-test_frac-val_frac))]})
        test_ds = Dataset.from_dict(
            {'text': dataset[math.ceil(len(dataset)*(1-test_frac-val_frac)):math.ceil(len(dataset)*(1-val_frac))]})
        
        splits = {'train': train_ds, 'test': test_ds}
        
        if val_frac > 0:
            val_ds = Dataset.from_dict(
                {'text': dataset[math.ceil(len(dataset)*(1-val_frac)):]})
            splits["validation"] = val_ds
        
        splits = DatasetDict(splits)
        splits = splits.map(lambda x: self.tokenizer(x["text"]), batched=True, num_proc=32, 
                            remove_columns=['text'], load_from_cache_file=False, desc="Tokenizing dataset")
        
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop,
            # you can customize this part to your needs.
            if total_length >= self.block_sz:
                total_length = (total_length // self.block_sz) * self.block_sz
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + self.block_sz] for i in range(0, total_length, self.block_sz)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        splits = splits.map(group_texts, batched=True, num_proc=124, load_from_cache_file=False, desc="Grouping dataset")

        if save:
            splits["train"].save_to_disk(data_dir+"/train")
            splits["test"].save_to_disk(data_dir+"/test")
            if val_frac > 0:
                splits["validation"].save_to_disk(data_dir+"/validation")

        return splits


    def extract_CTRL_header(header: str):
        '''
        Extract perplexity from the fasta header generated by CTRL models

        Parameters
        ----------
        header: str
            A fasta header of shape '>X.X.X.X <perplexity>'

        Returns
        -------
        ppl : float
            The perplexity value
        '''
        
        header = header.strip().strip('>').split()
        ppl = header[1]

        return float(ppl)
    

    def _load_fasta(self, path: str, replace: dict = {}, append: bool = False, 
                    has_step: bool = False):
        ''' 
        Loads the data from a fasta file generated by a CTRL model

        Parameters
        ----------
        path: str
            Path to the fasta file with headers generated by a CTRL model

        replace: dict
            A dictionary with tokens and their replacements,
            default = {}

        Returns
        -------
        None
        '''

        super()._load_fasta(path, replace, append, has_step, extract_header=CTRLLoader.extract_CTRL_header)


class UniformRandomLoader(DataLoader):

    def __init__(self):
        super().__init__()

    def extract_header(header: str):
        return float(header.strip().strip('>'))
    
    def _load_fasta(self, path: str, replace: dict = {}, append: bool = False,
                    has_step: bool = False):
        
        super()._load_fasta(path, replace, append, has_step, extract_header = UniformRandomLoader.extract_header)



class PottsLoader(DataLoader):

    def __init__(self):
        super().__init__()

    def preprocess(self,
                   test_frac: float,
                   d: int,
                   train_batch_size: int = 100,
                   test_batch_size: int = 100,
                   shuffle: bool = True):
        
        if self.data is None:
            raise ValueError("Load data first")

        data = self.data.copy()
        
        if shuffle:
            random.shuffle(data)

        data = torch.from_numpy(aa2int(data))
        data = torch.nn.functional.one_hot(data, num_classes = d).double()

        idx = np.ceil((1-test_frac) * len(self.data)).astype(int)
        
        train_data = data[:idx]
        test_data = data[idx:]

        train_loader = TorchDL(train_data, shuffle = shuffle, batch_size = train_batch_size, pin_memory=True, drop_last=True)
        test_loader = TorchDL(test_data, shuffle = shuffle, batch_size = test_batch_size, pin_memory=True, drop_last=True)

        return train_loader, test_loader


class SEDDLoader(DataLoader):

    def __init__(self, block_sz: int = 1024):
        super().__init__()

        self.block_sz = block_sz

    
    def _load_fasta(self, path: str, replace: dict = {}, append: bool = False,
                    has_step: bool = False):
        
        super()._load_fasta(path, replace, append, has_step, extract_header = None)
    
    
    def preprocess(self, 
                   test_frac: float,
                   group: bool = False,
                   save: bool = False,
                   shuffle: bool = True,
                   data_dir: str = '.'):
        
        '''
        Transform the data in the loader into huggingface dataset instances ready for training
        
        Parameters
        ---------
        test_frac: float
            Fraction of the data used for testing

        val_frac: float
            Fraction of the data used for validation

        save: bool
            Whether to save the resulting datasets to disk, default = False

        shuffle: bool
            Whether to shuffle the sequences before constructing the datasets

        data_dir: str
            Directory to save datasets into, default = '.'


        Returns
        -------
        splits: DatasetDict
            Contains train, test and validation dataset
        '''

        if self.data is None:
            raise ValueError("Load data first")
        
        data = self.data.copy()

        if shuffle:
            random.shuffle(data)

        #TODO: use ProteinTokenizer
        if group:
            tokenized_seqs = list(map(lambda x: aa2int_single(x) + [20], data))
        else:
            data = np.array([x[:self.block_sz-1] for x in data if len(x) >= self.block_sz-1])
            #max_len = max(lens)
            #tokenized_seqs = [aa2int_single(x) + (self.block_sz - lens[i])*[20] for i, x in enumerate(self.data)]
            tokenized_seqs = [aa2int_single(x) + [20] for x in data]

        n = len(tokenized_seqs)

        train_ds = Dataset.from_dict(
            {'input_ids': tokenized_seqs[:math.ceil(n*(1-test_frac))]})
        test_ds = Dataset.from_dict(
            {'input_ids': tokenized_seqs[math.ceil(n*(1-test_frac)):]})
        
        splits = {'train': train_ds, 'test': test_ds}
        splits = DatasetDict(splits)
        
        if group:
            splits = splits.map(self.group_texts, batched=True, num_proc=124, load_from_cache_file=False, desc="Grouping dataset")
        splits = splits.with_format("torch")

        if save:
            splits["train"].save_to_disk(data_dir+"/train")
            splits["test"].save_to_disk(data_dir+"/test")

        return splits
    

    def get_torch_loaders(self, 
                          train_set, 
                          test_set,
                          train_batch_sz: int,
                          test_batch_sz: int,
                          train_accum: int,
                          distributed: bool = False):

        if distributed:
            train_sampler = DistributedSampler(train_set) 
            test_sampler = DistributedSampler(test_set)
        else:
            train_sampler = None
            test_sampler = None

        ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

        def cycle_loader(dataloader, sampler=None):
            while 1:
                if sampler is not None:
                    sampler.set_epoch(np.random.randint(0, 100000))
                for data in dataloader:
                    yield data

        train_loader = cycle_loader(TorchDL(
            train_set,
            batch_size=train_batch_sz // (ngpus * train_accum),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=(train_sampler is None),
            persistent_workers=True,
            drop_last = True,
        ))
        test_loader = cycle_loader(TorchDL(
            test_set,
            batch_size=test_batch_sz // (ngpus * train_accum),
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=(test_sampler is None),
            drop_last = True,
        ))

        return train_loader, test_loader
    

class DeepEBMLoader(DataLoader):
    
        def __init__(self):
            super().__init__()
        
        def preprocess(self,
                       test_frac: float,
                       train_batch_size: int,
                       test_batch_size: int,
                       tokenizer = None,
                       shuffle: bool = True):

            if self.data is None:
                raise ValueError("Load data first")

            data = self.data.copy()
                
            if shuffle:
                random.shuffle(data)

            data_p = torch.from_numpy(aa2int(data))
            data_em = torch.concat([tokenizer.encode(x, return_tensors="pt") for x in data], dim=0)

            idx = np.ceil((1-test_frac) * len(self.data)).astype(int)
            
            train_data = TensorDataset(data_em[:idx], data_p[:idx])
            test_data = TensorDataset(data_em[idx:], data_p[idx:])

            nwork = 4 * (torch.cuda.device_count() if torch.cuda.is_available() else 1 )
            train_loader = TorchDL(train_data, shuffle = shuffle, batch_size = train_batch_size, pin_memory=True, drop_last=True, num_workers=nwork)
            test_loader = TorchDL(test_data, shuffle = shuffle, batch_size = test_batch_size, pin_memory=True, drop_last=True, num_workers=nwork)

            return train_loader, test_loader




def loaderFactory(model_type: str = "base", **kwargs):

    models = {
        "base" : DataLoader,
        "ctrl" : CTRLLoader,
        "random" : UniformRandomLoader,
        "potts" : PottsLoader,
        "sedd" : SEDDLoader,
        "debm" : DeepEBMLoader,
    }

    return models[model_type](**kwargs)



    

        