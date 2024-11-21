import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os
import itertools
import torch

from genzyme.data import DataLoader
from genzyme.evaluation.weighted_hamming_kernel import weighted_hamming_kernel

from mutedpy.protein_learning.kernels.hamming_kernel import hamming_kernel
from mutedpy.protein_learning.kernels.onehot_kernel import onehot_kernel
from mutedpy.protein_learning.kernels.blosum_kernel import blosum_kernel

class SimilarityStats():

    def __init__(self, train_dat: DataLoader | str, gen_dat: DataLoader):
        
        self.single_ref = isinstance(train_dat, str)

        if self.single_ref:
            train_dat = np.array([train_dat])

        self.train = train_dat
        self.gen = gen_dat

        self.metrics = {"hamming": 0, "one-hot": 1, "blosum": 2, "weighted hamming": 3}
        
    def ppl_sim_heatmap(self,
                        sim_metric: np.ndarray,
                        idx : np.ndarray,
                        sort_by_sz: bool = True,
                        q: float = 0.99,
                        **kwargs):
        '''
        Create a 2D-heatmap of similarity metric vs perplexity

        Parameters
        ----------
        sim_metric: np.ndarray
            Similarity of each generated sequence to the training data

        idx: np.ndarray
            Index to subsample the generated sequences

        sort_by_sz: bool
            Whether to sort the generated sequences by size to match
            the order of sim_metric

        q: float
            Centered fraction of the distribution to cover.
            The bottom q quantile and top 1-q quantile will be removed.
            Set to 1 to prevent truncation, default = 0.99

        **kwargs
            Passed to np.hist2d

        Returns
        --------
        ax: plt.Axes
            The ax object containing the plot
        '''
        
        lens = [len(seq) for seq in self.gen.get_data()[idx]]

        if sort_by_sz:
            idx = np.argsort(lens, stable = True)
        else:
            idx = range(len(sim_metric))
        
        alpha = (1-q)/2

        fig, ax = plt.subplots()
        hh = ax.hist2d(self.gen.label[idx], sim_metric, bins=70, density=True)
        ax.set_xlim(np.quantile(self.gen.label[idx], alpha), np.quantile(self.gen.label[idx], q+alpha))
        ax.set_ylim(np.quantile(sim_metric, alpha), np.quantile(sim_metric, q+alpha))
        ax.set_ylabel('similarity')
        ax.set_xlabel('perplexity')
        fig.colorbar(hh[3], ax=ax)

        return ax

    
    def map_property_to_gen(self,
                    property: str,
                    sim_metric: str):
        '''
        Map a property of the training data to the generated data
        based on the closest sequence in the training data.
        Modifies gen_dat in-place.

        Parameters
        ----------
        property: str
            Property to map from training data to generated data.
            Must be an attribute of train

        sim_metric: str
            Similarity metric to use to determine the closest sequence.

        Returns
        -------
        prop_gen: np.ndarray
            Property map for each sequence in gen

        similarity: np.ndarray
            Similarity to closest sequence for each sequence in gen
        '''

        if sim_metric not in self.metrics.keys():
            raise NotImplementedError(f"Unknown similarity metric {sim_metric}")
        
        pool = Pool(os.cpu_count())
        argmax = -np.ones(self.gen.length, dtype=int)
        similarity = -np.ones(self.gen.length)

        train_lens = np.array([len(seq) for seq in self.train.get_data()])
        gen_lens = np.array([len(seq) for seq in self.gen.get_data()])
        
        for l in np.unique(train_lens):
            lib = self.train.get_data()[train_lens == l]
            query = self.gen.get_data()[gen_lens == l]
            
            args = zip(query, itertools.repeat(lib, len(query)))
            indices = []
            sims = []

            for res in pool.starmap(single_seq_similarity, tqdm(args, total = len(query)), chunksize = 1):
                sim, idx = res[self.metrics[sim_metric]]
                indices.append(idx)
                sims.append(sim)

            argmax[gen_lens == l] = indices
            similarity[gen_lens == l] = sims

        pool.close()
        pool.join()

        prop_gen = np.empty((self.gen.length,))
        prop_gen.fill(np.nan)
        prop_gen[argmax != -1] = self.train.__dict__[property][argmax[argmax != -1]]
        self.gen.__dict__[property] = prop_gen

        return prop_gen, similarity

        
    def compute_similarity(self, 
                           subsample_gen: bool = True, 
                           subsample_train: bool = False, 
                           n_samples: int = 1000):
        '''
        Get the similarity of each generated sequnece ot the closest real sequence.
        Computes 1/hamming distance and normalized one-hot similarity.

        Parameters
        ----------
        subsample_gen: bool
            Whether to use a random subset of the generated sequences for evaluation,
            default = True

        subsample_train: bool
            Whether to use a random subset of the real sequences for evaluation,
            default = False

        n_samples: int
            Number of samples to choose during subsampling

        Returns
        --------
        hdist, ohdist, bldist, whdist: list
            The hamming, one-hot and blosum similarity to the closest real sequence for each generated sequence

        train_idx, gen_idx: np.ndarray | None
            Index used for subsampling the respective dataset
        '''

        train_dat = self.train if self.single_ref else self.train.get_data()
        gen_dat = self.gen.get_data()
        
        train_idx = np.arange(len(train_dat), dtype=int)
        gen_idx = np.arange(len(gen_dat), dtype=int)
        if subsample_train:
            train_idx = np.random.choice(len(train_dat), min(n_samples, len(train_dat)), replace = False)
            train_dat = train_dat[train_idx]

        if subsample_gen:
            gen_idx =  np.random.choice(len(gen_dat), min(n_samples, len(gen_dat)), replace = False)
            gen_dat = gen_dat[gen_idx]

        train_lens = np.array([len(seq) for seq in train_dat])
        gen_lens = np.array([len(seq) for seq in gen_dat])

        gen_idx = gen_idx[np.isin(gen_lens, np.unique(train_lens))]

        pool = Pool(os.cpu_count())
        hsim = []
        ohsim = []
        blsim = []
        whsim = []
        
        for l in np.unique(train_lens):
            lib = train_dat[train_lens == l]
            query = gen_dat[gen_lens == l]
            
            args = zip(query, itertools.repeat(lib, len(query)))

            for (hamm, _), (oneh, __), (blos, ___), (whamm, ____) in pool.starmap(single_seq_similarity, tqdm(args, total = len(query)), chunksize = 1):
                hsim.append(hamm)
                ohsim.append(oneh/l)
                blsim.append(blos/l)
                whsim.append(whamm)

        pool.close()
        pool.join()

        return hsim, ohsim, blsim, whsim, train_idx, gen_idx


def single_seq_similarity(seq, lib):
    '''
    Get the similarity of seq to the closest sequence in library.

    Parameters
    ----------
    seq: str
        The query sequence

    lib: list
        The sequence library

    Returns
    --------
    hamm, oneh, blos: float
        The hamming, one-hot and blosum similarity to the closest sequence
    '''

    hamm, hidx = torch.max(hamming_kernel([seq], lib)[0], dim=0)
    oneh, ohidx = torch.max(onehot_kernel([seq], lib)[0], dim=0)
    blos, bidx = torch.max(blosum_kernel([seq], lib)[0], dim=0)
    whamm, whidx = torch.max(weighted_hamming_kernel([seq], lib)[0], dim=0)

    return (hamm.item(), hidx.item()), (oneh.item(), ohidx.item()), (blos.item(), bidx.item()), (whamm.item(), whidx.item())