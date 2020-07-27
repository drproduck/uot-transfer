import torch
import numpy as np
from codebase.distance import batch_eudist_sq
import pdb
import time


def transfer_with_map(xt, S, Xs=None, xs=None, batch_size=128):
    """Transports source samples Xs onto target ones Xt
    Parameters
    ----------
    xt: The sampled target
    Xs : array-like, shape (n_source_samples, n_features)
        The source
    xs: The sampled source (is a subset of Xs) which coupling is computed
    S: the learned coupling   
    batch_size : int, optional (default=128)
        The batch size for out of sample inverse transform
    Returns
    -------
    transp_Xs : array-like, shape (n_source_samples, n_features)
        The transport source samples.
    """

    if Xs is None:

        # perform standard barycentric mapping
        transp = S / torch.sum(S, dim=-1, keepdim=True)

        # set nans to 0
        transp[~ torch.isfinite(transp)] = 0

        # compute transported samples
        transp_Xs = transp @ xt

    else:
        # transport the source samples
        # n = S.shape[0]
        # marg = torch.ones((n, 1)).to(0) / n
        # rowsum = torch.sum(S, dim=-1, keepdim=True)
        # remainder = marg - rowsum
        # transp_xs = (remainder * xs + S @ xt) / marg

        # perform standard barycentric mapping
        transp = S / torch.sum(S, dim=-1, keepdim=True)

        # set nans to 0
        transp[~ torch.isfinite(transp)] = 0

        # compute transported samples
        transp_xs = transp @ xt

        
        transp_Xs = []
        # perform out-of-sample mapping
        idx = 0
        
        time_mv2gpu = 0
        while idx < Xs.shape[0]:
            # get the nearest neighbor in the source domain
            next_idx = np.min([idx + batch_size, Xs.shape[0]])
            Xs_b = Xs[idx:next_idx]
            
            time_s = time.time()
            Xs_b = Xs_b.to(0)
            time_mv2gpu += time.time() - time_s
            
            D0 = batch_eudist_sq(Xs_b, xs)
            min_idx = torch.argmin(D0, -1)

            # define the transported points
            transp_Xs_b = transp_xs[min_idx, :] + Xs_b - xs[min_idx, :]
            transp_Xs.append(transp_Xs_b)
            
            idx = next_idx
            
        transp_Xs = torch.cat(transp_Xs, dim=0)

    return transp_Xs, time_mv2gpu
