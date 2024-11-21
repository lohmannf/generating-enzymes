import matplotlib.pyplot as plt
import numpy as np
import argparse

from genzyme.evaluation.utils import compare_histograms
from genzyme.data import loaderFactory


def main(args):

    if args.dataset == "ired":
        res_dirs = {
            "sedd": "./gen_data/sedd/ired.fasta",
            "dfm": "./gen_data/dfm/ired.fasta",
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
            #"deep ebm": "./gen_data/deep_ebm/frozen_seed_31_gwg_T_100/potts.fasta",
            #"potts uniform": "./gen_data/potts_uniform_T_0.01_unified.fasta",
            #"random": "./gen_data/random.fasta",
            "random informed": "./gen_data/random_trained.fasta",
            }
    
    elif args.dataset == "mid1":
        res_dirs = {
            "sedd": "./gen_data/mid1/sedd/mid1.fasta",
            "dfm": "./gen_data/mid1/dfm/mid1.fasta",
            "frozen": "./gen_data/mid1/zymctrl/frozen.fasta",
            "pretrained": "./gen_data/mid1/zymctrl/zymctrl_pretrained.fasta",
            "ft lr 8e-05": "./gen_data/mid1/zymctrl/zymctrl_lr_8e-05.fasta",
            "deep ebm T=5": "./gen_data/mid1/deep_ebm/frozen_seed_31_uniform_T_5.fasta",
            "deep ebm T=10": "./gen_data/mid1/deep_ebm/frozen_seed_31_uniform_T_10.fasta",
            "potts gwg T=0.01": "./gen_data/mid1/potts/frozen_seed_31_local_T_0.01.fasta",
            "potts gwg T=1": "./gen_data/mid1/potts/frozen_seed_31_local_T_1.fasta",
            "random": "./gen_data/mid1/random/random_uninformed.fasta",
            "random informed": "./gen_data/mid1/random/random_informed.fasta",
            }
        
    else:
        raise NotImplementedError

    # TVD Computation #########################

    tvd = {}

    sim_ref = loaderFactory("ctrl")
    sim_ref.load(args.dataset)
    if args.use_wildtype:
        sim_ref.set_data(np.array([sim_ref.reference[:args.seq_len]]))
    sim_ref.unify_seq_len(args.seq_len)

    train_dat = loaderFactory("ctrl")
    train_dat.load(args.dataset)
    train_dat.unify_seq_len(args.seq_len)

    for ref, (r_name, r_path) in enumerate(res_dirs.items()):
        # load the reference data for TVD computation
        y = loaderFactory()
        y.load('fasta', path = res_dirs[r_name], replace = {'[UNK]':'*'})
        y.unify_seq_len(len(sim_ref.get_data()[0]))

        if r_name.startswith("potts") and y.length > 10000:
                y.data = np.random.choice(y.data, 10000, replace=False)

        for curr in range(ref+1, len(res_dirs)):
            c_name, c_path = list(res_dirs.items())[curr]

            try:
                gen_dat = loaderFactory(c_name.split()[0])
            except KeyError:
                gen_dat = loaderFactory()

            gen_dat.load('fasta', path = c_path, replace = {'[UNK]':'*'})
            gen_dat.unify_seq_len(args.seq_len)

            if c_name.startswith("potts") and gen_dat.length > 10000:
                gen_dat.data = np.random.choice(gen_dat.data, 10000, replace=False)

            # get TVD to reference for all similarity metrics
            print(c_name, r_name)
            bins = [3,20,30,3]
            tvd[(ref, curr)] = compare_histograms(gen_dat, y, sim_ref, n_bins = bins, subsample_gen = not args.use_wildtype, n_samples = 1000)

        if args.use_wildtype:
            # add distance to training data
            tvd[(ref, len(res_dirs))] = compare_histograms(train_dat, y, sim_ref, n_bins = bins, subsample_gen = not args.use_wildtype, n_samples = 1000)

    # Plotting ##############################
    sims = ['Hamming', 'One-hot', 'Blosum', 'Weighted Hamming']
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize=(10, 9), layout="constrained")

    for i, name in enumerate(sims):

        data = np.zeros((len(res_dirs)+1, len(res_dirs)+1))
        for (x,y),v in tvd.items():
            data[x,y] = v[i]
            data[y,x] = v[i]

        im = axs[i//2, i%2].imshow(data, vmin=0, vmax=1)
        axs[i//2, i%2].tick_params(axis='x', labelrotation=90)
        axs[i//2, i%2].set_xticks(np.arange(len(res_dirs)+1), labels=list(res_dirs.keys())+["train data"], fontsize="small")
        axs[i//2, i%2].set_yticks(np.arange(len(res_dirs)+1), labels=list(res_dirs.keys())+["train data"], fontsize="small")
        axs[i//2, i%2].set_title(name)

        for k in range(len(res_dirs)+1):
            for j in range(k+1, len(res_dirs)+1):
                text = axs[i//2, i%2].text(j, k, np.round(data[k, j],2),
                            ha="center", va="center", color="w", fontsize="xx-small")
                text = axs[i//2, i%2].text(k, j, np.round(data[k, j],2),
                            ha="center", va="center", color="w", fontsize="xx-small")
                
    fig.colorbar(im, ax=axs, shrink=0.6, aspect=20, orientation="vertical", label="total variation distance")

    plt.show()
    plt.savefig(args.output_file, dpi=500)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog = "TVD Matrix computation",
                                     description = "Create a matrix of the total variation distance between similarity histograms")

    parser.add_argument("--use_wildtype", action="store_true", help="Compute similarity to wildtype sequence")
    parser.add_argument("-o", "--output_file", default = "../tvd_matrix.png", help="Output file for plot")
    parser.add_argument("-l", "--seq_len", type=int, default = 290, help="Length to which sequences are truncated / filtered")
    parser.add_argument("-d", "--dataset", default = "ired", help="Name of the dataset")

    args = parser.parse_args()
    main(args)