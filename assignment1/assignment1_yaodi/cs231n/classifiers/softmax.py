import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
      score = X[i].dot(W)
      score -= np.max(score)
      L = -score[y[i]] + np.log(np.sum(np.exp(score)))
      loss += L
      for j in range(num_class):
          temp = np.exp(score[j]) / np.sum(np.exp(score))
          if j == y[i]:
              dW[:, j] += (temp - 1) * X[i]
          else:
              dW[:, j] += temp * X[i]
  loss /= num_train
  loss += 0.5 * reg * np.sum( W*W )
  dW /= num_train
  dW += reg * W
      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train = X.shape[0]
  score = X.dot(W)
  maxscore = np.max(score, axis = 1)
  maxscore = maxscore.reshape(-1,1)
  score -= maxscore
  correct_score = score[range(num_train),y]
  loss = loss - np.sum(correct_score) + np.sum(np.log(np.sum(np.exp(score), axis = 1)))
  loss /= num_train
  loss += 0.5 * reg * np.sum( W*W )
  
  temp = np.exp(score) / np.sum(np.exp(score),axis = 1).reshape(-1, 1)
  temp[range(num_train), y] -= 1
  dW = (X.T).dot(temp)
  dW /= num_train
  dW += reg * W
  
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

