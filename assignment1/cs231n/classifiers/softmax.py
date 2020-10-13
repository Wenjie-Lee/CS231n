from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
        f = X[i].dot(W)     # 1xC
        f -= np.max(f)      # shift to minus~0
        ef = np.exp(f)
        sumef = np.sum(ef)
        loss += -f[y[i]] + np.log(sumef)
        for j in range(num_class):
            # gradLi/Wj = -X + exp(WjXi) / sum(exp(WjXi)) of j, when j == y[i]
            #   ~       = exp(WjXi) / sum(exp(WjXi)) of j, when j != y[i]
            temp = np.exp(f[j]) / np.sum(sumef)
            if j == y[i]:
                dW[:,j] += (-1 + temp) * X[i]
            else:
                dW[:,j] += temp * X[i]

    loss = loss / num_train + reg * np.sum(W * W) / 2
    dW = dW / num_train + reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]

    f = X.dot(W)        # NxC
    f -= np.max(f, axis=1).reshape(-1,1)    # 计算出来的向量都是横向量，要转换一下
    ef = np.exp(f)
    sumef = np.sum(ef, axis=1).reshape(-1,1)# 同上

    temp = ef / sumef
    # 只计算0对角线上的值
    loss = -np.sum(np.log(temp[range(num_train), list(y)]))
    loss = loss / num_train + reg * np.sum(W * W) / 2

    dtemp = temp.copy()
    # 同样对0对角线上特殊处理
    dtemp[range(num_train), list(y)] += -1
    dW = (X.T).dot(dtemp)   # X.T(DxN) dot dtemp(NxC) -> dW(DxC)
    dW = dW / num_train + reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
