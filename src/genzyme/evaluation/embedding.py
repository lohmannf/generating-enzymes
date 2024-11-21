import torch
import esm
import numpy as np
from sklearn import metrics, discriminant_analysis, linear_model, decomposition, preprocessing, pipeline
import os
from cycler import cycler
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm

from genzyme.data import loaderFactory
from genzyme.evaluation.similarity import SimilarityStats

class ESM:
    # Get ESM-2 Embeddings for protein sequences
    # adapted from https://pypi.org/project/fair-esm/

    def __init__(self):
        
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)


    def get_embeddings(self, sequences: np.ndarray, batch_sz: int = 20):
        '''
        Get ESM-2 representations and LH metrics for protein sequences

        Parameters
        ----------
        sequences: np.ndarray
            Amino acid sequences to compute embeddings for

        batch_sz: int
            Batch size to use when passing data to ESM-2.
            Decrease when using smaller GPUs. Default = 20.

        Returns
        --------
        embddings, nll, ppl
            Sequence embeddings, average NLL and perplexity.
        
        '''

        sequences = [(str(i), seq) for i, seq in enumerate(sequences)]
        converter = self.alphabet.get_batch_converter()

        nll_fn = torch.nn.CrossEntropyLoss(reduction="mean")

        seq_repr = []
        seq_nll = []

        for i in range(int(np.ceil(len(sequences)/batch_sz))):

            labs, strs, toks = converter(sequences[i*batch_sz: min(len(sequences), (i+1)*batch_sz)])
            batch_lens = (toks != self.alphabet.padding_idx).sum(1)

            # get per token representations
            with torch.no_grad():
                results = self.model(toks.to(self.device), repr_layers=[33], return_contacts=False)
                tok_repr = results["representations"][33].cpu()

                # average to per sequence representations
                for j, tokens_len in enumerate(batch_lens):
                    seq_repr.append(tok_repr[j, 1:(tokens_len-1)].mean(0).numpy())
                    logits = results["logits"][j, 1:(tokens_len-1)]
                    seq_nll.append(nll_fn(logits, toks[j, 1:(tokens_len-1)].to(self.device)).item())

        return np.array(seq_repr), np.array(seq_nll), np.exp(seq_nll)
    

class EmbeddingStats:
    '''Analyze an embedding of the data'''

    def __init__(self, 
                 labels: np.ndarray, 
                 embeddings: np.ndarray,
                 label_names: np.ndarray = None,
                 n_dims: int = 5,
                 seed: int = 31):

        if len(labels) != len(embeddings):
            raise ValueError('Number of labels and embeddings does not match')

        np.random.seed(seed)

        um = umap.UMAP(min_dist=0.5, n_neighbors=50)
        self.umap = um.fit_transform(embeddings)
        self.labels = labels
        self.embeddings, self.explained_variance = self.reduce_dim(embeddings, n_dims, True, True)
        self.label_names = label_names

    def train_test_split(self, split: float):
        '''Creates a class-balanced train-test split
        
        Parameters
        ----------
        split: float
            Fraction of the data used for testing

        Returns
        ------
        X_train, y_train, X_test, y_test: np.ndarray
            The training and test data and labels
        '''
        
        idx = np.array([])
        for l in np.unique(self.labels):
            idx_curr = np.arange(len(self.labels)).astype(int)[self.labels == l]
            idx = np.concatenate([idx, np.random.choice(idx_curr, int(np.ceil(len(idx_curr)*split)), replace=False)]).astype(int)

        X_test = self.embeddings[idx]
        y_test = self.labels[idx]
        X_train = self.embeddings[~idx]
        y_train = self.labels[~idx]

        return X_train, y_train, X_test, y_test
    

    def reduce_dim(self, X: np.ndarray, n_dims: int, plot: bool = False, scale: bool = False):
        '''
        Reduce the dimensionality of X to n_dims with PCA

        X: np.ndarray
            The input data

        n_dims: int
            Number of output dimensions

        plot: bool
            Whether to create an elbow plot and mark n_dims, default = False

        Returns
        -------
        X_pc: np.ndarray
            Reduced data (n_samples x n_dims)
        '''

        if scale:
            pca = pipeline.make_pipeline(preprocessing.StandardScaler(), decomposition.PCA())
        else:
            pca = decomposition.PCA()
        X_pc = pca.fit_transform(X)

        if scale:
            pca = pca.steps[1][1]
        
        fig, ax = plt.subplots()

        print(pca.explained_variance_ratio_, flush=True)
        
        if plot:
            ax.plot(np.arange(min(X.shape[1], X.shape[0])), pca.explained_variance_ratio_)
            ax.vlines(x=n_dims, ymin=0, ymax=1, linestyles="--", colors="red")
            plt.show()
            plt.savefig("elbow.png")
            plt.close()

        return X_pc[:,:n_dims], pca.explained_variance_ratio_[:n_dims]
    

    def plot_embeddings(self, color: np.ndarray = None, col_name: str = None, umap: bool = False):
        '''
        Plot the embedding space in 2D colored by some label

        Parameters
        ----------
        color: np.ndarray | None
            Labels according to which the data points are colored.
            If None, use model labels instead. Default = None

        col_name: str | None
            The name of the property mapped in color.
            Used for colorbar labeling.

        umap: bool
            Whether to use umap embedding instead of PCA, default = False.

        Returns
        -------
        fig, ax
        '''

        if umap:
            X = self.umap
        else:
            X = self.embeddings[:,:2]

        print(self.explained_variance)

        fig, ax = plt.subplots(layout="constrained")

        ax.set_prop_cycle(cycler(color=["red", "blue", "green", "lime", "dodgerblue", "saddlebrown",
                                         "darkorange", "blueviolet", "gray", "yellow", "darkkhaki", "lightpink","aqua", "fuchsia"]))
        
        if color is not None:

            im = ax.scatter(X[:, 0], X[:, 1], s=4, c = color, vmin=min(color), vmax = max(color))
            fig.colorbar(im, ax=ax, shrink=0.6, aspect=20, orientation="vertical", label = col_name)

        else:
            for l in np.unique(self.labels):
                im = ax.scatter(X[self.labels == l, 0], X[self.labels == l, 1], label = l if self.label_names is None else self.label_names[l], 
                        s=4, zorder = (2 if l==9 else 1))

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, ncols=int(np.ceil(len(np.unique(self.labels))/2)), loc = "outside lower center", fontsize="xx-small", frameon=False)
        
            
        if umap:
            ax.set_xlabel(f'UMAP1')
            ax.set_ylabel(f'UMAP2')
        else:
            ax.set_xlabel(f'PC1 ({np.round(self.explained_variance[0]*100, 1)}%)')
            ax.set_ylabel(f'PC2 ({np.round(self.explained_variance[1]*100, 1)}%)')

        return fig, ax


    def silhouette(self):
        '''
        Get the average silhouette score for each model.
        Score in [-1, 1] with 1 best score

        Returns
        -------
        sil_per_cluster: dict
            Silhouette score for every label
        '''

        sil_per_samp = metrics.silhouette_samples(self.embeddings, self.labels)

        sil_per_cluster = {}
        for lab in np.unique(self.labels):
            sil_per_cluster[lab if self.label_names is None else self.label_names[lab]] = np.mean(sil_per_samp[self.labels == lab])

        return sil_per_cluster


    def logistic_regression(self, test_split: float = 0.2):
        '''
        Train logistic regression with n_classes = n_models

        Parameters
        ----------
        test_split: float
            What fraction of the data to exclude from the training set.
            Default = 0.2

        Returns
        -------
        model: sklearn.linear_model.LogisticRegression
            The trained classifier
        '''
        
        X_train, y_train, X_test, y_test = self.train_test_split(test_split)

        model = linear_model.LogisticRegression(multi_class = "ovr", n_jobs = os.cpu_count())
        model = model.fit(X_train, y_train)

        cm = metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, labels = self.labels)
        print(metrics.classification_report(y_test, model.predict(X_test)))

        return model

    
    def plot_confusion_matrix(self, model, X_test, y_test):
        '''
        Plot the confusion matrix of a trained classifier
        when predicting on X_test

        Parameters
        ----------
        model
            Trained sklearn classifier

        X_test: np.ndarray
            Test data

        y_test: np.ndarray
            Test labels

        Returns
        -------
        fig, ax
            Figure and ax object that contain the confusion matrix

        cm: np.ndarray
            Raw confusion matrix data
        '''

        cm = metrics.confusion_matrix(y_test, model.predict(X_test))
        fig, ax = plt.subplots(layout="constrained")
        im = ax.imshow(cm)
        ax.tick_params(axis='x', labelrotation=90)

        if self.label_names is not None:
            ax.set_xticks(np.arange(len(cm)), labels=self.label_names, fontsize="small")
            ax.set_yticks(np.arange(len(cm)), labels=self.label_names, fontsize="small")

        for k in range(len(cm)):
            for j in range(len(cm)):
                text = ax.text(j, k, cm[k, j],
                            ha="center", va="center", color="w", fontsize="x-small")
                text = ax.text(k, j, cm[j, k],
                            ha="center", va="center", color="w", fontsize="x-small")
                
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')

        fig.colorbar(im, ax=ax, shrink=0.8, aspect=20, orientation="vertical")

        return fig, ax, cm



    def qda(self, test_split: float = 0.2):
        '''
        Run a quadratic discriminant analysis with n_classes = n_models

        Parameters
        ----------
        test_split: float
            What fraction of the data to exclude from the training set.
            Default = 0.2

        Returns
        -------
        model: sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
            The trained classifier
        '''

        X_train, y_train, X_test, y_test = self.train_test_split(test_split)

        model = discriminant_analysis.QuadraticDiscriminantAnalysis()
        model = model.fit(X_train, y_train)

        fig, ax, cm = self.plot_confusion_matrix(model, X_test, y_test)
        ax.set_title('QDA confusion matrix')

        print(cm)
        plt.show()
        plt.savefig("qda.png")
        plt.close()
        
        print(metrics.classification_report(y_test, model.predict(X_test)))

        return model



if __name__ == "__main__":

    res_dirs = {
    "sedd": "./gen_data/sedd/ired.fasta",
    #"dfm": "./gen_data/dfm/ired.fasta",
    "frozen": "./gen_data/frozen/1.5.1.-.fasta",
    "pretrained": "./gen_data/zymCTRL_results/1.5.1.-.fasta",
    "tiny": "./gen_data/tiny_model/1.5.1.-.fasta",
    #"small": "./gen_data/small_model/1.5.1.-.fasta",
    "ft lr 8e-05": "./gen_data/zytune_untrunc/1.5.1.-.fasta",
    "ft lr 8e-06": "./gen_data/zymctrl_lr-06/1.5.1.-.fasta",
    "ft lr 8e-07": "./gen_data/zymctrl_lr-07/1.5.1.-.fasta",
    "ft lr 8e-08": "./gen_data/zymctrl_lr-08/1.5.1.-.fasta",
    "ft lr 8e-09": "./gen_data/zymctrl_lr-09/1.5.1.-.fasta",
    "potts gwg": "./gen_data/potts_adam_gwg_T_0.01_lr_0.001_unified.fasta",
    "random informed": "./gen_data/random_trained.fasta",
    "random": "./gen_data/random.fasta",
    }

    res_dirs = {
            "sedd": "./gen_data/mid1/sedd/mid1.fasta",
            #"dfm": "./gen_data/mid1/dfm/mid1.fasta",
            "frozen": "./gen_data/mid1/zymctrl/frozen.fasta",
            "pretrained": "./gen_data/mid1/zymctrl/zymctrl_pretrained.fasta",
            "ft lr 8e-05": "./gen_data/mid1/zymctrl/zymctrl_lr_8e-05.fasta",
            "deep ebm T=1": "./gen_data/mid1/deep_ebm/frozen_seed_31_uniform_T_1.fasta",
            "deep ebm T=5": "./gen_data/mid1/deep_ebm/frozen_seed_31_uniform_T_5.fasta",
            "deep ebm T=10": "./gen_data/mid1/deep_ebm/frozen_seed_31_uniform_T_10.fasta",
            #"potts gwg T=0.01": "./gen_data/mid1/potts/frozen_seed_31_local_T_0.01.fasta",
            #"potts gwg T=1": "./gen_data/mid1/potts/frozen_seed_31_local_T_1.fasta",
            "random": "./gen_data/mid1/random/random_uninformed.fasta",
            #"random informed": "./gen_data/mid1/random/random_informed.fasta",
    }

    model_dirs = {
        "deep ebm": "/cluster/project/krause/flohmann/deep_ebm_f_True_lr_0.001_acc_16_mid1",
        "potts": "/cluster/home/flohmann/generating-enzymes/potts_mid1.pt"
    }

    from genzyme.models import modelFactory
    from omegaconf import OmegaConf
    from genzyme.data.utils import aa2int

    potts = torch.load(model_dirs["potts"])
    debm = modelFactory("debm", cfg = OmegaConf.load("./configs/deep_ebm/config.yaml"))
    debm.load_ckpt(model_dirs["deep ebm"])

    n_per_dataset = 600
    l = 97
    seed = 7
    sim_metric = "one-hot"

    seqs = []
    labs = []

    np.random.seed(seed)

    for i, path in enumerate(res_dirs.values()):
        print(path, flush=True)
        with open(path, "r") as file:
            lines = [s.strip().replace("[UNK]", "").replace("X", "") for s in file.readlines() if not s.startswith('>')]
            lines = np.unique([''.join([pos for pos in seq if not pos.isdigit()]) for seq in lines])
        
        lines = np.unique([x[:l] for x in lines if len(x) >= l])

        if len(lines) > n_per_dataset:
            lines = np.random.choice(lines, n_per_dataset, replace=False)

        seqs.extend(lines)
        labs.extend(len(lines)* [i])

    train_dat = loaderFactory()
    # train_dat.load('ired')
    # train_dat._replace("*")
    train_dat.load("mid1")

    lines = train_dat.get_data()[np.random.choice(len(lines), n_per_dataset, replace=False)]
    labs.extend([i+1]*len(lines))
    seqs.extend(lines)

    gen_dat = loaderFactory()
    gen_dat.set_data(np.array(seqs))

    # sim_stats = SimilarityStats(train_dat, gen_dat)
    # fitness, similarity = sim_stats.map_property_to_gen("fitness", sim_metric)
    
    embs = ESM()
    embeddings, nll, ppl = embs.get_embeddings(seqs)

    avg_nll = {}
    avg_ppl = {}
    for l in np.unique(labs):
        avg_nll[(list(res_dirs.keys())+["train data"])[l]] = np.mean(nll[np.array(labs) == l])
        avg_ppl[(list(res_dirs.keys())+["train data"])[l]] = np.mean(ppl[np.array(labs) == l])

    print(avg_nll)
    print(avg_ppl)

    seqs_oh_p = torch.nn.functional.one_hot(torch.from_numpy(aa2int(seqs)), num_classes=20).double()
    seqs_int_em = torch.concat([debm.em_tokenizer.encode(x, return_tensors="pt") for x in seqs], dim=0)

    batch_size=50
    debm_energy = np.empty((0,))
    potts_energy = np.empty((0,))
    with torch.no_grad():
        for batch_em, batch_p in tqdm(zip(torch.split(seqs_int_em, batch_size, dim=0), torch.split(seqs_oh_p, batch_size, dim=0))):
            debm_energy = np.concatenate([debm_energy, debm(batch_em.to(debm.device)).cpu().squeeze().numpy()])
            potts_energy = np.concatenate([potts_energy, potts(batch_p.to(potts.device)).cpu().numpy()])
    
    debm_energy *= -1
    potts_energy *= -1

    stats = EmbeddingStats(np.array(labs), np.array(embeddings), label_names = list(res_dirs.keys())+["train data"])
    print(stats.silhouette())

    # _, ax = stats.plot_embeddings(color = fitness, col_name = "fitness")
    # ax.set_title('ESM-2 embeddings')
    # plt.show()
    # plt.savefig("embeddings_fitness.png", dpi=500)
    # plt.close()

    # _, ax = stats.plot_embeddings(color = fitness, col_name = "fitness", umap=True)
    # ax.set_title('ESM-2 embeddings')
    # plt.show()
    # plt.savefig("embeddings_umap_fitness.png", dpi=500)
    # plt.close()

    # _, ax = stats.plot_embeddings(color = similarity, col_name = "similarity")
    # ax.set_title('ESM-2 embeddings')
    # plt.show()
    # plt.savefig("embeddings_similarity.png", dpi=500)
    # plt.close()

    # _, ax = stats.plot_embeddings(color = similarity, col_name = "similarity", umap=True)
    # ax.set_title('ESM-2 embeddings')
    # plt.show()
    # plt.savefig("embeddings_umap_similarity.png", dpi=500)
    # plt.close()

    # _, ax = stats.plot_embeddings(color = np.log(similarity), col_name = "log(similarity)")
    # ax.set_title('ESM-2 embeddings')
    # plt.show()
    # plt.savefig("embeddings_log_similarity.png", dpi=500)
    # plt.close()

    # _, ax = stats.plot_embeddings(color = np.log(similarity), col_name = "log(similarity)", umap=True)
    # ax.set_title('ESM-2 embeddings')
    # plt.show()
    # plt.savefig("embeddings_umap_log_similarity.png", dpi=500)
    # plt.close()

    _, ax = stats.plot_embeddings(color = debm_energy, col_name = "energy")
    ax.set_title('Deep EBM Energy')
    plt.show()
    plt.savefig("embeddings_debm.png", dpi=500)
    plt.close()

    _, ax = stats.plot_embeddings(color = debm_energy, col_name = "energy", umap=True)
    ax.set_title('Deep EBM Energy')
    plt.show()
    plt.savefig("embeddings_debm_umap.png", dpi=500)
    plt.close()

    _, ax = stats.plot_embeddings(color = potts_energy, col_name = "energy")
    ax.set_title('Potts Model Energy')
    plt.show()
    plt.savefig("embeddings_potts.png", dpi=500)
    plt.close()

    _, ax = stats.plot_embeddings(color = potts_energy, col_name = "energy", umap=True)
    ax.set_title('Potts Energy')
    plt.show()
    plt.savefig("embeddings_potts_umap.png", dpi=500)
    plt.close()


    _, ax = stats.plot_embeddings(color = ppl, col_name = "perplexity")
    ax.set_title('ESM-2 embeddings')
    plt.show()
    plt.savefig("embeddings_ppl.png", dpi=500)
    plt.close()

    _, ax = stats.plot_embeddings(color = ppl, col_name = "perplexity", umap=True)
    ax.set_title('ESM-2 embeddings')
    plt.show()
    plt.savefig("embeddings_umap_ppl.png", dpi=500)
    plt.close()

    _, ax = stats.plot_embeddings()
    ax.set_title('ESM-2 embeddings')
    plt.show()
    plt.savefig("embeddings.png", dpi=500)
    plt.close()

    _, ax = stats.plot_embeddings(umap=True)
    ax.set_title('ESM-2 embeddings')
    plt.show()
    plt.savefig("embeddings_umap.png", dpi=500)
    plt.close()

    _ = stats.qda()






        

        