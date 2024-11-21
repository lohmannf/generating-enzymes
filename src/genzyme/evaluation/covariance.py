import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import contingency, chi2_contingency
from omegaconf import DictConfig

from genzyme.models import modelFactory

class CovarianceStats:

    def __init__(self, L: int):
        
        self.L = L
        self.cms = {}
        self.chi_maps = {}
        self.pvalue_maps = {}
        

    def add_contact_map(self, name: str, seqs, cfg: DictConfig):
        '''
        Train a Potts model and extract a contact map from it.
        Contact map is saved internally

        Parameters
        ----------
        name: str
            Name of the model / dataset to which the sequences belong

        seqs: PottsLoader
            DataLoader that contains the sequences
        
        cfg: DictConfig
            Dict-style config that contains training and model sub-configs

        Returns
        -------
        None

        '''

        model = modelFactory("potts", model_cfg = cfg.model)
        train_set, val_set = seqs.preprocess(0.2, d = 20, test_batch_size = 2)
        # n_steps = 100000
        # n_epochs = np.ceil(n_steps / (len(seqs.get_data())*(1-0.2)//100)).astype(int)
        model.optimize_parameters(train_set, val_set, cfg.training)
        with torch.no_grad():
            self.cms[name] = model.get_coupling_info(apc=True).cpu()

    def add_chi_map(self, name: str, seqs, alpha: float = 0.05):
        '''
        Perform a chi-squared test on each pair of positions.
        Save the matrix of chi-squared statistics and significance of the corresponding
        p-values internally. Uses Bonferroni-correction to determine test significance.

        Parameters
        ----------
        name: str
            Name of the model / dataset to which the sequences belong

        seqs: PottsLoader
            DataLoader that contains the sequences

        alpha: float
            Significance level for the test.

        Returns
        --------
        None
        '''

        data = np.array([list(seq) for seq in seqs.get_data()])
        l = data.shape[1]

        data = data.T # position x sequence

        cont = contingency.crosstab(data)[1]
        S = np.zeros((l,l))
        P = np.zeros((l,l), dtype=bool)
        for i in range(l-1):
            for j in range(i+1,l):
                cont = contingency.crosstab(data[i], data[j])[1]
                # remedy sparsity of contingency table
                cont = cont[np.any(cont > 0, axis=1)]
                cont = cont[:, np.any(cont > 0, axis=0)]
                res = chi2_contingency(cont)

                S[i, j] = res[0]
                S[j, i] = S[i, j]
                P[i, j] = res[1] <= alpha/l
                P[j, i] = P[i, j]
        
        self.chi_maps[name] = S
        self.pvalue_maps[name] = P

    
    def plot_map(self, name: str, map_type: str = "contact", filter_ns: bool = True):
        '''
        Plot the internally saved map of model name and type map_type

        Parameters
        ----------
        name: str
            Name of the model / dataset whose map to plot

        map_type: str
            The kind of map to plot. Must be one of contact, chi, pvalue.
            Default = "contact"

        Returns
        -------
        fig, ax
            The figure and axes object that contain the plot
        '''
        
        fig, ax = plt.subplots(layout="constrained")
        if map_type == "contact":
            im = ax.imshow(self.cms[name], cmap="viridis")

        elif map_type == "chi":
            dat = self.chi_maps[name]
            print(self.pvalue_maps[name])
            if filter_ns:
                dat[~self.pvalue_maps[name]] = 0.
            im = ax.imshow(dat)

        elif map_type == "pvalue":
            im = ax.imshow(self.pvalue_maps[name])
        else:
            raise NotImplementedError()

        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.8, aspect=20, orientation="vertical")

        return fig, ax


    def get_distances(self, map_type: str = "contact"):
        '''
        Calculate the pairwise Frobenius distance between all available
        maps of type map_type

        Parameters
        ---------
        map_type: str
            The map type to calculate the distances from

        Returns
        -------
        fig, ax 
            Figure and axes object that contain the distance matrix
        '''

        if map_type == "contact":
            maps = self.cms
        elif map_type == "chi":
            maps = self.chi_maps
        elif map_type == "pvalue":
            maps = self.pvalue_maps
        else:
            raise NotImplementedError()
        
        if len(maps) == 0:
            raise ValueError(f'No {map_type} maps to compare')
        
        data = np.zeros((len(maps), len(maps)))
        
        for i, (name1, cm1) in enumerate(maps.items()):
            for j in range(i+1, len(maps)):

                cm2 = list(maps.values())[j]

                data[i,j] = torch.linalg.norm(cm1-cm2, ord="fro").numpy()
                data[j,i] = data[i,j]

        print(data, flush=True)

        fig, ax = plt.subplots(layout="constrained")
        im = ax.imshow(data)

        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xticks(np.arange(len(maps)), labels=maps.keys(), fontsize="small")
        ax.set_yticks(np.arange(len(maps)), labels=maps.keys(), fontsize="small")
        fig.colorbar(im, ax=ax, shrink=0.8, aspect=20, orientation="vertical")

        ax.set_title('Pairwise Frobenius distance')

        return fig, ax

        
    def pairwise_dist_hm(self, name1, name2, map_type: str = "contact"):
        '''
        Create an element-wise distance map of two particular maps
        of type map_type

        Parameters
        ---------
        name1, name2: str
            Names of the models to compare

        map_type: str
            The map type to calculate the distances from

        Returns
        -------
        fig, ax 
            Figure and axes object that contain the difference map
        '''

        if map_type == "contact":
            maps = self.cms
        elif map_type == "chi":
            maps = self.chi_maps
        elif map_type == "pvalue":
            maps = self.pvalue_maps
        else:
            raise NotImplementedError()

        dist = (maps[name1]-maps[name2]).numpy()

        edge = np.max(np.abs(dist))
        
        fig, ax = plt.subplots(layout="constrained")
        im = ax.imshow(dist, cmap="seismic", vmin=-edge, vmax=edge)
        ax.set_title(f'{name1}-{name2}')

        fig.colorbar(im, ax=ax, shrink=0.8, aspect=20, orientation="vertical")

        return fig, ax


if __name__ == "__main__":

    from genzyme.data import loaderFactory
    from genzyme.data.utils import AA_DICT


    res_dirs = {
    "frozen": "./gen_data/frozen/1.5.1.-.fasta",
    "pretrained": "./gen_data/zymCTRL_results/1.5.1.-.fasta",
    "tiny": "./gen_data/tiny_model/1.5.1.-.fasta",
    # "small": "./gen_data/small_model/1.5.1.-.fasta",
    "ft lr 8e-05": "./gen_data/zytune_untrunc/1.5.1.-.fasta",
    "ft lr 8e-06": "./gen_data/zymctrl_lr-06/1.5.1.-.fasta",
    "ft lr 8e-07": "./gen_data/zymctrl_lr-07/1.5.1.-.fasta",
    "ft lr 8e-08": "./gen_data/zymctrl_lr-08/1.5.1.-.fasta",
    "ft lr 8e-09": "./gen_data/zymctrl_lr-09/1.5.1.-.fasta",
    "random informed": "./gen_data/random_trained.fasta",
    "random": "./gen_data/random.fasta",
    #"potts": "./potts/potts.fasta"
    }

    res_dirs = {
            "sedd": "./gen_data/mid1/sedd/mid1.fasta",
            "dfm": "./gen_data/mid1/dfm/mid1.fasta",
            "frozen": "./gen_data/mid1/zymctrl/frozen.fasta",
            "pretrained": "./gen_data/mid1/zymctrl/zymctrl_pretrained.fasta",
            "ft lr 8e-05": "./gen_data/mid1/zymctrl/zymctrl_lr_8e-05.fasta",
            "random": "./gen_data/mid1/random/random_uninformed.fasta",
            "random informed": "./gen_data/mid1/random/random_informed.fasta",
            }
    
    n_seqs = 1000
    length = 97
    stats = CovarianceStats(length)

    for name, path in res_dirs.items():
        
        gen_dat = loaderFactory("potts")

        gen_dat.load('fasta', path = path, replace = {'[UNK]':''})
        gen_dat.set_data(np.array([''.join([pos for pos in seq if pos in AA_DICT.keys()]) for seq in gen_dat.get_data()]))
        gen_dat.unify_seq_len(length)

        print(name, len(gen_dat.get_data()))
        #gen_dat.set_data(np.random.choice(gen_dat.get_data(), 1000, replace=False) if n_seqs < len(gen_dat.get_data()) else gen_dat.get_data())

        #stats.add_contact_map(name, gen_dat,lambda_J=0.01, lambda_h=0.01)

        stats.add_chi_map(name, gen_dat)
        stats.plot_map(name, map_type="chi")
        plt.show()
        plt.savefig(f'am_{name}_mid1.png', dpi=500)
        plt.close()
        

    res_dirs["train data"] = ""
    train_data = loaderFactory("potts")
    # train_data.load("ired", replace_ast = True)
    # train_data._replace("*")
    # train_data.unify_seq_len(290)
    train_data.load("mid1")
    #train_data.set_data(np.random.choice(train_data.get_data(), n_seqs, replace=False))
    #train_data.set_data(np.random.choice(train_data.get_data(), 50, replace=False) if n_seqs < len(train_data.get_data()) else train_data.get_data())
    #stats.add_contact_map("train data", train_data)
    stats.add_chi_map("train data", train_data)
    stats.plot_map("train data", map_type="chi")
    plt.show()
    plt.savefig('am_train_mid1.png', dpi=500)
    plt.close()
    
    

    # stats.get_distances()
    # plt.show()
    # plt.savefig('cm_distance.png')
    # plt.close()

    # stats.plot_contact_map("train data")
    # plt.savefig(f"cm_train_data.png", dpi=500)
    # plt.close()

    # for name2 in list(res_dirs.keys())[:-1]:
    #     stats.plot_contact_map(name2)
    #     plt.savefig(f"cm_{name2}.png", dpi=500)
    #     plt.close()
    #     stats.pairwise_dist_hm("train data", name2)
    #     plt.savefig(f"dist_train_{name2}.png", dpi=500)
    #     plt.close()



    
    