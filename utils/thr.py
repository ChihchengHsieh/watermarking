import numpy as np

def get_best_thrs(fpr, tpr, thresholds):
    youdens_j = tpr - fpr
    best_index = np.argmax(youdens_j)
    best_thr = thresholds[best_index]
    return best_thr