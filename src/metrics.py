import numpy as np
import nolds
from scipy.spatial.distance import cdist

def compute_hausdorff_distance(traj_a, traj_b):
    """
    Standard Hausdorff Distance (Max-Min).
    Use this for rigorous 'worst-case' geometric evaluation.
    """
    if len(traj_a) > 5000:
        traj_a = traj_a[::2]
        traj_b = traj_b[::2]
        
    d_matrix = cdist(traj_a, traj_b, metric='euclidean')
    
    d_ab = np.max(np.min(d_matrix, axis=1))
    d_ba = np.max(np.min(d_matrix, axis=0))
    
    return max(d_ab, d_ba)

def compute_modified_hausdorff_distance(traj_a, traj_b):
    """
    Modified Hausdorff Distance (MHD) - Average of Minimums.
    As defined in Eq 6 & 7 of the paper.
    d_MH(A, B) = 0.5 * ( mean(min_b ||a-b||) + mean(min_a ||b-a||) )
    
    This metric is much less sensitive to outliers and matches the training loss.
    """
    if len(traj_a) > 5000:
        traj_a = traj_a[::2]
        traj_b = traj_b[::2]
        
    # Compute Euclidean distance matrix
    d_matrix = cdist(traj_a, traj_b, metric='euclidean')
    
    # Mean of minimum distances (A -> B)
    # Note: The paper uses squared norm for optimization, 
    # but usually reports Euclidean distance for metric interpretation.
    # We use Euclidean here to be comparable to spatial units.
    term_1 = np.mean(np.min(d_matrix, axis=1))
    
    # Mean of minimum distances (B -> A)
    term_2 = np.mean(np.min(d_matrix, axis=0))
    
    return 0.5 * (term_1 + term_2)

def compute_lyapunov_exponent(trajectory, dt):
    """
    Estimates the Maximum Lyapunov Exponent (MLE).
    """
    data_1d = trajectory[:, 0]
    
    try:
        # lyap_r estimates largest LE
        mle = nolds.lyap_r(data_1d, emb_dim=3, min_tsep=None, tau=1)
        return mle / dt
    except Exception as e:
        return None