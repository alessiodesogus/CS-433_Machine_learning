# -*- coding: utf-8 -*-
"""Cost Function"""

import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    e=y-tx@w
    return e.T@e / (2*len(y))
    # TODO: compute loss by MSE
    # ***************************************************

def compute_loss_mae(y, tx, w):
    """Calculate the loss using MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    N = tx.shape[0]
    e = y - tx@w 
    MAE = (1/2*N) * np.sum(np.abs(e))
    return MAE
    # TODO: compute loss by MAE
    # ***************************************************
