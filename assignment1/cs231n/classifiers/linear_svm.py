from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    # # 计算所得分类分数
    # scores = X.dot(W)
    # # 根据正确分类y得到的分数，是一个列向量 1xN
    # correct_class_score = scores[y]
    # # 计算边界值
    # margin = scores - correct_class_score[:,0] + 1
    # # 处理掉边界值矩阵中的负数使之为0，只留下满足边界值条件的元素值

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, y[i]] += -X[i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # 计算所得分类分数 NxC
    scores = X.dot(W)
    # 根据正确分类y得到的分数，是一个列向量 Nx1
    # 内容是(0,num_train-1)->(y(0),y(-1))
    correct_class_score = scores[range(num_train), list(y)].reshape(-1,1)
    # 计算边界值
    margins = np.maximum(0, scores - correct_class_score + 1)
    # 处理掉边界值矩阵中的负数使之为0，只留下满足边界值条件的元素值
    margins[range(num_train), list(y)] = 0

    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    temp = np.zeros((num_train, num_classes))
    # temp 和 margins 同形，将margins>0的元素位置设为1
    temp[margins > 0] = 1
    # 将矩阵0对角线部分重置为0
    temp[range(num_train), list(y)] = 0
    # 计算每一种类的需要计算loss的个数，赋到0对角线上
    temp[range(num_train), list(y)] = -np.sum(temp, axis=1)

    dW = (X.T).dot(temp)
    dW = dW / num_train + reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
