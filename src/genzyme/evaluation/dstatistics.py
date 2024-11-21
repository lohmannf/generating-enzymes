import matplotlib.pyplot as plt
import logging
from scipy.stats import mannwhitneyu, gaussian_kde
from scipy.stats.mstats import chisquare
import numpy as np
import pandas as pd
from tqdm import tqdm
from seaborn import heatmap

from genzyme.data import DataLoader

logger = logging.Logger(__name__)

class DatasetStatistics:
    '''
    Calculate and compare distribution statistics on the given datasets
    '''

    def __init__(self, train_data: DataLoader, gen_data: DataLoader):
        ''' 
        Parameters
        ----------
        loaders: list[DataLoader]
            Holds the datasets for which to compare the statistics

        Returns
        -------
        DatasetStatistics
        '''

        self.real = train_data
        self.gen = gen_data


    def length_dist(self, plot_type: str = None, **kwargs):
        '''
        Perform a Mann-Whitney-U test and create a plot of the sequence length distributions
        of the datasets

        Parameters
        ----------
        plot_type: str
            What kind of plot to create, must be 'box', 'violin' or None, default = None

        **kwargs
            Passed to the respective plot function

        Returns
        -------
        test_res: MannWhiteyUResult
            The result of the Mann-Whitney-U test

        ax: plt.Axes
            The ax object containing the plot
        '''
        
        length_real = [len(seq) for seq in self.real.get_data()]
        length_gen = [len(seq) for seq in self.gen.get_data()]

        test_res = mannwhitneyu(length_real, length_gen, alternative = 'two-sided')

        ax = None
        if plot_type == 'box':
            
            ax = plt.subplot()
            ax.boxplot(x = [length_real, length_gen], tick_labels = ['real', 'generated'], **kwargs)

        elif plot_type == 'violin':
            
            ax = plt.subplot()
            ax.violinplot(dataset = [length_real, length_gen],  **kwargs)
            ax.set_xticks([1,2],['real', 'generated'])
            ax.set_ylabel('sequence length')

        elif plot_type == 'hist':
            
            fig, ax = plt.subplots(nrows=2, ncols=1)
            # kde_real = gaussian_kde(length_real)
            # kde_gen = gaussian_kde(length_gen)
            # xx_real = np.linspace(min(length_real), max(length_real), 1000)
            # xx_gen = np.linspace(min(length_gen), max(length_gen), 1000)
            
            kwargs['bins'] = max(length_real)-min(length_real)+1
                
            min_glob = min(length_real+length_gen)
            max_glob = max(length_real+length_gen)
            
            ax[0].hist(length_real, density=True, **kwargs)
            # ax[0].plot(xx_real, kde_real(xx_real))
            ax[0].set_title('Real')
            ax[0].set_xlim(min_glob, max_glob)
            ax[0].axvline(x = np.median(length_real), ymin=0, ymax=1, label= 'Median', color='r', linestyle='--')

            kwargs['bins'] = max(length_gen)-min(length_gen)+1

            ax[1].hist(length_gen, density=True, **kwargs)
            # ax[1].plot(xx_gen, kde_gen(xx_gen))
            ax[1].set_title('Generated')
            ax[1].set_xlim(min_glob, max_glob)
            ax[1].axvline(x = np.median(length_gen), ymin=0, ymax=1, label= 'Median', color='r', linestyle='--')

            plt.tight_layout()


        return test_res, ax

    

    def amino_acid_dist(self, plot: bool = False):
        '''
        Compare the amino acid distributions of the two datasets using
        a chi-squared test and plot their histograms

        Parameters
        ----------
        plot: bool
            Whether to create histograms of the two distributions

        Returns
        -------
        test_res: ChisquareResult
            The result of the chi-squared test

        ax: plt.Axes
            The ax object containing the plot
        '''
        def aa_occurrences(seq):
            
            aas, ct = np.unique(list(seq), return_counts=True)
            aa_map = dict(zip(aas, ct/len(seq)))

            return aa_map

        aa_real = pd.DataFrame([aa_occurrences(seq) for seq in self.real.get_data()])
        aa_real.fillna(0, inplace = True)
        aa_real = aa_real.mean(axis=0)

        aa_gen = pd.DataFrame([aa_occurrences(seq) for seq in self.gen.get_data()])
        aa_gen.fillna(0, inplace = True)
        aa_gen = aa_gen.mean(axis=0)

        aa = pd.concat([aa_real, aa_gen], axis=1).fillna(0)
        aa_real = aa.to_numpy()[:,0]
        aa_gen = aa.to_numpy()[:,1]

        avg_len = np.mean([len(seq) for seq in self.gen.get_data()])

        test_res = chisquare(f_obs = aa_gen*avg_len, f_exp = aa_real*avg_len)

        ax = None
        if plot:
            ax = plt.subplot()
            xpos = np.arange(len(aa_real))
            ax.bar(x = xpos - 0.4, height = aa_real,  width = 0.4, align='edge', label = 'real')
            ax.bar(x = xpos, height = aa_gen, width = 0.4, align = 'edge', label = 'generated')
            ax.set_xticks(xpos, aa.index)
            ax.legend()

        return test_res, ax
    
    def get_frequencies(self, sequences: list):
         
        lens = [len(seq) for seq in sequences]
        n_pos = max(lens)
        n_seq = len(sequences)

        hm_data = pd.DataFrame()

        for pos in range(n_pos):
            
            data = [sequences[i][pos] for i in range(n_seq) if pos < lens[i]]
            
            aa, cts = np.unique(data, return_counts = True)
            cts = cts.astype(float)/len(data)

            hm_data = pd.concat([hm_data, pd.DataFrame([cts], columns = aa, index = [pos])], ignore_index=False)

        hm_data = hm_data.reindex(sorted(hm_data.columns), axis=1)         

        return hm_data.fillna(0)


    def aa_variation(self, plot_real: bool = True, ax: plt.Axes = None, gen_label: str = "generated"):
        '''
        Generate a plot of amino acid variation per position

        Parameters
        ----------
        plot_real: bool
            Whether to plot variation in the real (training) data aswell,
            default = True

        ax: plt.Axes
            Existing axis to add the plot to, default = None

        gen_label: str
            Label to use for generated data in the legend, default = "generated"

        Returns
        --------
        ax: plt.Axes
            The axis containing the plot
        '''

        hm_real = self.get_frequencies(self.real.get_data()).fillna(0)
        hm_gen = self.get_frequencies(self.gen.get_data()).fillna(0)
        
        entropy = lambda x: -x * np.log(x) if x!=0 else 0
        hm_real = hm_real.map(entropy)
        hm_gen = hm_gen.map(entropy)

        hm_real = hm_real.sum(axis = 1)
        hm_gen = hm_gen.sum(axis = 1)

        if ax is None:
            ax = plt.subplot()
        if plot_real:
            ax.plot(hm_real.to_numpy(), label = "real")
        ax.plot(hm_gen.to_numpy(), label = gen_label)
        ax.set_ylabel('entropy')
        ax.set_xlabel('sequence position')
        ax.legend()

        return ax, hm_real, hm_gen
    

    def n_mutations_ppos(self, normalize = True):
        '''
        Make a barplot of the number of mutations per position
        '''

        ref = np.array(list(self.real.reference))
        self.gen.unify_seq_len(len(ref))

        data = np.array([list(x) for x in self.gen.get_data()])

        mut_ppos = np.sum((data != ref), axis=0)
        if normalize:
            mut_ppos /= self.gen.length

        fig, ax = plt.subplots(layout = "constrained")
        ax.bar(range(len(mut_ppos)), mut_ppos)

        ax.set_xlabel("sequence position")

        if normalize:
            ax.set_ylabel("pct mutated")
        else:
            ax.set_ylabel("# mutated")

        return fig, ax


    def n_mutations(self):
        '''
        Make histogram of the total number of mutations per sequence
        '''

        ref = np.array(list(self.real.reference))
        self.gen.unify_seq_len(len(ref))

        data = np.array([list(x) for x in self.gen.get_data()])

        mut = np.sum((data != ref), axis=1)

        n_bins = (max(mut)+1) - min(mut)

        fig, ax = plt.subplots(layout = "constrained")
        ax.hist(mut, bins=n_bins, range=(min(mut), max(mut)+1))

        ax.set_ylabel("# sequences")
        ax.set_xlabel("# mutations")

        return fig, ax


    def aa_position_heatmap(self):
        '''
        Generate a heatmap of normalized counts of each amino acid at every sequence position
        '''

        hm_real = self.get_frequencies(self.real.get_data())
        hm_gen = self.get_frequencies(self.gen.get_data())

        max_len = max(max(hm_gen.index), max(hm_real.index))
        
        hm_real = hm_real.reindex(index = range(max_len+1)).fillna(0)
        hm_gen = hm_gen.reindex(index = range(max_len+1)).fillna(0)

        fig, axs = plt.subplots(nrows=1, ncols=3, layout='tight', figsize=[8, 12], width_ratios = [10,10,1])

        axs[0].tick_params(axis='both', which='major', labelsize=8)
        heatmap(hm_real, xticklabels= True, yticklabels = 'auto', ax=axs[0], cbar_ax = axs[2])
        axs[0].set_xlabel('Amino acid')
        axs[0].set_ylabel('Sequence position')
        axs[0].set_title(f'Real ({len(self.real.get_data())})')

        axs[1].tick_params(axis='both', which='major', labelsize=8)
        heatmap(hm_gen, xticklabels = True, yticklabels = False, ax=axs[1], cbar_ax = axs[2])
        labs = axs[0].get_yticklabels()
        pos = axs[0].get_yticks()
        axs[1].set_yticks(pos, labs)
        axs[1].set_xlabel('Amino acid')
        axs[1].set_ylabel('Sequence position')
        axs[1].set_title(f'Generated ({len(self.gen.get_data())})')

        return axs
    
    