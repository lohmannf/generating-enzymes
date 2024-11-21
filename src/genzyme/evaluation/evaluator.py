import logging
import matplotlib.pyplot as plt
from typing import Callable
import numpy as np
from tqdm import tqdm

from genzyme.evaluation.dstatistics import DatasetStatistics
from genzyme.evaluation.similarity import SimilarityStats
from genzyme.data import DataLoader
from genzyme.models import BaseModel

logger = logging.Logger(__name__)

class Evaluator:

    def __init__(self, 
                train_data: str | DataLoader, 
                gen_data: str | DataLoader, 
                run_name: str,
                loader_class: Callable = DataLoader,
                model_class: Callable = BaseModel,
                seed: int = 31):
        ''' 
        Parameters
        ----------
        train_data: str | DataLoader
            Path to a .fasta file or DataLoader that contains the real data used for training

        gen_data: str | DataLoader
            Path to a .fasta file or DataLoader that contains the generated data

        run_name: str
            Name to use for naming plots

        loader_class: Callable
            DataLoader subclass to use for constructing the datasets from fasta files,
            default = DataLoader

        model_class: Callable
            The model class, should match loader_class, default = BaseModel

        seed: int
            Random seed, default = 31

        Returns
        -------
        Evaluator instance
        '''
        np.random.seed(seed)

        if isinstance(train_data, str):
            self.train = loader_class()
            self.train.load('fasta', file = train_data)
        
        else:
            self.train = train_data

        if isinstance(gen_data, str):
            self.gen = loader_class()
            self.gen.load('fasta', file = gen_data)
        
        else:
            self.gen = gen_data

        self.name = run_name
        self.model_class = model_class
        self.loader_class = loader_class


    def calculate_ppl(self, 
                data: np.ndarray, 
                model_dir: str,
                **eval_kwargs):
        '''
        Calculates the perplexity of the sequences in data using
        the model stored at model_dir.

        Parameters
        ----------
        data: np.ndarray
            An array of sequences

        model_dir: str
            Path to the model state from which model_class is instantiated

        Returns
        -------
        ppls: list
            The perplexities in the same order as data
        '''

        model = self.model_class(model_dir = model_dir)
        ppls = model.eval(data, **eval_kwargs)

        return ppls

    
    def ppl_comparison(self):
        ''' 
        Generate a violin plot comparing the perplexities of the real and generated data

        Returns
        --------
        ax: plt.Axes
            An ax object containing the plot
        '''

        ax = plt.subplot()
        ax.violinplot([self.train.label, self.gen.label], showmedians=True, showmeans = True)
        ax.set_xticks([1,2], ['real', 'generated'])
        ax.set_ylabel('Perplexity')
        
        return ax


    def run_evaluation(self,
                       length_dist: bool = True,
                       aa_heatmap: bool = True,
                       similarity: bool = True,
                       model_dir: str = None):

        logger.setLevel(logging.DEBUG)

        stats = DatasetStatistics(self.train, self.gen)

        if length_dist:

            mwu, bpl = stats.length_dist(plot_type = 'hist')
            
            plt.show()
            plt.savefig(f'{self.name}_len_dist.png')
            plt.close()

            print(f'Mann-Whitney-U test with statistic = {mwu.statistic} and p-value = {mwu.pvalue}')
        
        # chisq, bpl = stats.amino_acid_dist(plot=True)
        
        # plt.show()
        # plt.savefig(f'{self.name}_aa_dist.png')
        # plt.close()
        
        # print(f'Chi-squared test with statistic = {chisq.statistic} and p-value = {chisq.pvalue}')

        if aa_heatmap:
            # heatmap of amino acid vs sequence position
            aa_hms = stats.aa_position_heatmap()

            plt.show()
            plt.savefig(f'{self.name}_aa_hm.png')
            plt.close()


            var_ppos = stats.aa_variation()

            plt.show()
            plt.savefig(f'{self.name}_var_pposition.png')
            plt.close()


        # # output seqs with lowest perplexity
        # ppls, seqs = self.get_top_sequences(10)
        # for ppl,seq in zip(ppls, seqs):

        #     print(ppl)
        #     print(seq)

        if similarity:

            sim = SimilarityStats(self.train, self.gen)

            hdist, ohdist, bldist, whdist, train_idx, gen_idx = sim.compute_similarity(subsample_train = False, 
            subsample_gen = True, n_samples = 1000)
            fig, axs = plt.subplots(nrows=2, ncols = 2, figsize=([11, 9]), layout='tight')
            axs[0,0].hist(hdist, bins = 30)
            axs[0,0].set_title('1 / Hamming distance')

            axs[0,1].hist(ohdist, bins = 30)
            axs[0,1].set_title('One-hot similarity')

            axs[1,0].hist(bldist, bins = 30)
            axs[1,0].set_title('Blosum similarity')

            axs[1,1].hist(whdist, bins = 30)
            axs[1,1].set_title('1 / Blosum-weighted Hamming distance')

            plt.show()
            plt.savefig(f'{self.name}_similarities.png')
            plt.close()


            simhm = sim.ppl_sim_heatmap(bldist, gen_idx)
            simhm.set_title('Blosum')

            plt.show()
            plt.savefig(f'{self.name}_blosum_psim_hm.png')
            plt.close()


            simhm = sim.ppl_sim_heatmap(hdist, gen_idx)
            simhm.set_title('Hamming')

            plt.show()
            plt.savefig(f'{self.name}_hamming_psim_hm.png')
            plt.close()


            simhm = sim.ppl_sim_heatmap(ohdist, gen_idx)
            simhm.set_title('One-hot')

            plt.show()
            plt.savefig(f'{self.name}_oneh_psim_hm.png')
            plt.close()


            simhm = sim.ppl_sim_heatmap(whdist, gen_idx)
            simhm.set_title('Weighted Hamming')

            plt.show()
            plt.savefig(f'{self.name}_whamming_psim_hm.png')
            plt.close()

        # if model_dir is not None:

        #     # get perplexities for training data
        #     self.train.set_label(self.calculate_ppl(self.train.get_data(), model_dir))
        #     vpl = self.ppl_comparison()

        #     plt.show()
        #     plt.savefig(f'{self.name}_perplexity.png')
        #     plt.close()
    
        
    def get_top_sequences(self, n_seq: int, descending: bool = False):
        '''
        Get the n_seq generated sequences according to value of self.gen.label

        Parameters
        -----------
        n_seq: int
            Number of sequences to return

        descending: bool
            Whether to evaluate rank of sequences in descending order

        Returns
        --------
        label, data: np.ndarray
            The ordered label values and corresponding sequences
        '''

        
        idx = np.argsort(self.gen.label)
        if descending:
            idx = idx[::-1]

        idx = idx[:n_seq].astype(int)

        return self.gen.label[idx], self.gen.get_data()[idx]