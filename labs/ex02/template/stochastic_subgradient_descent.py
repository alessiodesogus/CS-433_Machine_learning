# -*- coding: utf-8 -*-
"""Stochastic Subgradient Descent"""

from helpers import batch_iter
from costs import compute_loss_mae
from subgradient_mae import compute_subgradient_mae

def stochastic_subgradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic SubGradient Descent algorithm (SubSGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic subgradient
        max_iters: a scalar denoting the total number of iterations of SubSGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SubSGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SubSGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # ***************************************************
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1) :
        
            grad = compute_subgradient_mae(y_batch, tx_batch, w)
            w = w - gamma*grad
            loss = compute_loss_mae(y, tx, w)
            ws.append(w)
            losses.append(loss)
        # TODO: implement stochastic subgradient descent.
        # ***************************************************
        
        print(
            "SubSGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws