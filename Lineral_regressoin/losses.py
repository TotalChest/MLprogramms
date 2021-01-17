import numpy as np
import scipy
from scipy.special import expit
from scipy.special import logsumexp


class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef
        self.is_multiclass_task = False

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """

        return np.logaddexp(0, -y * X.dot(w)).mean() + \
               self.l2_coef * np.dot(w[1:], w[1:])

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """
        
        return X.T.dot((-y) * expit(-y * X.dot(w))) / X.shape[0] + \
               2 * self.l2_coef * np.hstack([0, w[1:]])

class MultinomialLoss(BaseLoss):
    """
    Loss function for multinomial regression.
    It should support l2 regularization.
    
    w should be 2d numpy.ndarray.
    First dimension is class amount.
    Second dimesion is feature space dimension.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef
        self.is_multiclass_task = True

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 2d numpy.ndarray

        Returns
        -------
        : float
        """
        N = X.shape[0]
        A = X.dot(w.T)
        L = -A[np.arange(N), y].sum()
        L += logsumexp(A, axis=1).sum()
        L /= N
        L += self.l2_coef * (w[:, 1:] ** 2).sum()
        return L

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 2d numpy.ndarray

        Returns
        -------
        : 2d numpy.ndarray
        """
        A = X.dot(w.T)
        A = A - np.max(A, axis=1)[:, None]
        B = np.exp(A) / np.exp(A).sum(axis=1)[:, None]
        G = X.T.dot(B).T
        for y_i in np.unique(y):
            G[y_i, :] = G[y_i, :] - X[np.where(y == y_i)].sum(axis=0)
        G = G + self.l2_coef * 2 * np.hstack([np.zeros(w.shape[0])[:, None], w[:, 1:]])
        return G
