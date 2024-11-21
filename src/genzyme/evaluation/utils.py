import numpy as np
import os
from tqdm import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from genzyme.evaluation.evaluator import Evaluator
from genzyme.evaluation.similarity import SimilarityStats
from genzyme.data import DataLoader
from genzyme.evaluation.tvd import total_variation_distance

def pad_integers(raw_ints):
    ''' 
    Add prefixed 0 padding to enable correct lexicographic sorting of integers
    '''
    
    lens = [len(str(num)) for num in raw_ints]
    max_len = max(lens)
    pad_ints = ['0'*(max_len-lens[i])+ str(raw_ints[i]) for i in range(len(raw_ints))]

    return pad_ints

def eval_per_step(model_name: str, gen_data: DataLoader, train_data: DataLoader):

    if not hasattr(gen_data, 'step'):
            raise AttributeError('Generated data does not contain training step information')
    
    labs = pad_integers(np.unique(gen_data.step))
        
    for i, step in tqdm(enumerate(np.unique(gen_data.step)), desc = 'Performing evaluation per step', total = len(labs)):
          
        curr_data = DataLoader()
        curr_data.set_data(gen_data.get_data()[gen_data.step == step])
        curr_data.set_label(gen_data.label[gen_data.step == step])

        eval = Evaluator(train_data, curr_data, run_name = f'{model_name}_{labs[i]}')
        eval.run_evaluation(similarity=False)



def animate_plot(img_dir: str, img_suffix: str):
    '''
    Turn a sequence of images into a video

    Parameters
    ----------
    img_dir: str
        The parent directory of the images. Final video will be saved here aswell

    img_suffix: str
        File suffix according to which the images are selected

    Returns
    -------
    None
    '''
     
    imgs = [img_dir+'/'+img for img in os.listdir(img_dir) if img.endswith(img_suffix)]

    clip = ImageSequenceClip(imgs, fps=1)
    clip.write_videofile(img_dir+'/'+img_suffix.strip('png')+'mp4')
    clip.close()
        



def compare_histograms(x: DataLoader, 
                       y: DataLoader,
                       ref: DataLoader,
                       n_bins: list = [1, 50, 50, 1],
                       upper: list = [None]*4,
                       lower: list = [None]*4,
                       **sim_kwargs):
     
    '''
    Determine the total variation distances between the 4 similarity histograms
    of x and y

    Parameters
    ----------
    x, y: DataLoader
        The generated datasets to compare

    ref: DataLoader
        Data (e.g. training data) to use as reference for computing similarities

    n_bins: list
        Number of equally spaced bins to use in blosum and one-hot similarity.
        Step size in terms of # mutations to use for bin placement in hamming and weighted hamming similarity.
        default = [1, 50, 50, 1]

    upper, lower: list
        Upper and lower bound for the bins of each similarity histogram. 
        Will use the maximum and minimum respectively if None.
        Default = [None, None, None, None]

    **sim_kwargs
        Additional kwargs passed to SimilarityStats.compute_similarity


    Returns
    --------
    tvd: Tuple[float]
        The total variation distance between the 4 similarity histograms
        (Hamming, One-hot, Blosum, Weighted Hamming)
    '''


    sim_x = SimilarityStats(ref, x)
    res_x = sim_x.compute_similarity(**sim_kwargs)

    sim_y = SimilarityStats(ref, y)
    res_y = sim_y.compute_similarity(**sim_kwargs)

    tvd = []

    for i, (sx, sy, bins, upper, lower) in enumerate(zip(res_x[:4], res_y[:4], n_bins, upper, lower)):
        
        sx = np.array(sx)[np.isfinite(np.array(sx))]
        sy = np.array(sy)[np.isfinite(np.array(sy))]
        
        if lower is None:
            lower = min(min(sx), min(sy))
        if upper is None:
            upper = max(max(sx), max(sy))

        if i==0 or i==3:
            lower = max(lower, 1/(len(ref.get_data()[0])+1))
            step_sz = bins
            inv_lower = 1/lower-1
            inv_upper = 1/upper-1
            try:
                bins = 1/(np.arange(inv_upper, inv_lower, step_sz)[::-1]+1.+1e-8)
            except Exception as e:
                print(inv_upper, inv_lower, upper, lower)
                raise e
            bins = np.concatenate([[0.], bins, [upper]])

        hist_x, _ = np.histogram(sx, bins = bins, range = (lower, upper))
        hist_y, _ = np.histogram(sy, bins = bins, range = (lower, upper))

        tvd.append(total_variation_distance(hist_x/len(sx), hist_y/len(sy)))


    return tuple(tvd)
        
    
            