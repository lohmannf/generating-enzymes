import numpy as np

def total_variation_distance(bins1: np.ndarray, 
                             bins2: np.ndarray):
    
    '''
    Calculate the total variation distance between two discrete PMFs

    Parameters
    ----------
    bins1: np.ndarray
        Bins of first histogram

    bins2: np.ndarray
        Bins of second histogram

    Returns
    -------
    tvd: float
        Total variation distance between the two histograms
    '''

    if bins1.shape != bins2.shape:
        raise ValueError('Can only compare histograms with same number of bins')
    
    bins1 /= np.sum(bins1)
    bins2 /= np.sum(bins2)

    return 1/2 * np.sum(np.abs(bins1-bins2))