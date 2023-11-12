# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

from helpers import batch_iter
from costs import compute_loss
from gradient_descent import compute_gradient


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    # ***************************************************
    stoch_gradient = compute_gradient(y, tx, w)
    return stoch_gradient
    # TODO: implement stochastic gradient computation. It's the same as the usual gradient.
    # ***************************************************


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # ***************************************************
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1) :
        
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma*grad
            loss = compute_loss(y, tx, w)
            ws.append(w)
            losses.append(loss)
        
        # TODO: implement stochastic gradient descent.
        # ***************************************************
        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws