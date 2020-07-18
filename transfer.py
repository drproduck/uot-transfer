import torch
import numpy as np
from codebase.distance import batch_eudist_sq
import pdb


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
        # perform out of sample mapping
        indices = torch.arange(Xs.shape[0])
        batch_ind = [
            indices[i:i + batch_size]
            for i in range(0, len(indices), batch_size)]

        # transport the source samples
        transp = S / torch.sum(S, 1, keepdim=True)
        transp[~ torch.isfinite(transp)] = 0
        transp_xs = transp @ xt

        transp_Xs = []
        for bi in batch_ind:
            # get the nearest neighbor in the source domain
            D0 = batch_eudist_sq(Xs[bi], xs)
            idx = torch.argmin(D0, -1)


            # define the transported points
            transp_Xs_b = transp_xs[idx, :] + Xs[bi] - xs[idx, :]

            transp_Xs.append(transp_Xs_b)

        transp_Xs = torch.cat(transp_Xs, dim=0)

    return transp_Xs