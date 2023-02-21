"""
Distance function implementations between two graphs.
"""
import numpy as np
from scipy.sparse import dia_matrix, linalg
from numpy.linalg import svd,eig

def hamming_based_distance(A, B):
    """
    A and B are numpy arrays with the same number of rows and columns.
    We assume that the nodes from A to B are in correspondence (identical, no new nodes or nodes disappearing).
    """
    N = A.shape[0]
    return np.sum(np.abs(A-B)) / (N*(N-1))

def delta_c(x, y):
    if x == y:
        return 0
    else:
        return 1

def jaccard_based_distance(A1, A2):
    # we assume that the nodes from A1 to A2 are in correspondence (identical, no new nodes or nodes disappearing)
    N = A1.shape[0]
    Delta = np.abs(A1 - A2)
    Union_rough = A1 + A2
    Union = np.apply_along_axis(lambda x: np.array([delta_c(x[i], 0) for i in range(len(x))]), 1, Union_rough)
    return np.sum(Delta) / np.sum(Union)

def poly_distance(A1, A2, order_max, weights=None, alpha=1):
    # we assume that the nodes from A1 to A2 are in correspondence (identical, no new nodes or nodes disappearing)
    N = A1.shape[0]
    if weights is None:
        weights = np.array([1 / (N - 1) ** (alpha * (k - 1)) for k in range(1, order_max + 1)])

    decomp1 = svd(A1)
    poly1 = np.array([weights[k-1] * decomp1[1]**k for k in range(1, order_max+1)])
    exp1 = decomp1[0] @ np.diag(np.apply_along_axis(np.sum, 0, poly1)) @ decomp1[2]
    
    decomp2 = svd(A2)
    poly2 = np.array([weights[k-1] * decomp2[1]**k for k in range(1, order_max+1)])
    exp2 = decomp2[0] @ np.diag(np.apply_along_axis(np.sum, 0, poly2)) @ decomp2[2]
    
    Delta = exp1 - exp2
    return 1/N**2 * np.sum(Delta**2)

from numpy.linalg import eig
from numpy import sort, diag, apply_along_axis, sqrt, isinf, isnan

def eigen_distance(A1, A2, f=lambda x: x, p=2, type="laplacian"):
    N = A1.shape[0]
    
    if type == "laplacian":
        L1 = diag(apply_along_axis(sum, 1, A1)) - A1
        L2 = diag(apply_along_axis(sum, 1, A2)) - A2
    elif type == "norm_laplacian":
        D1 = diag(1.0 / sqrt(apply_along_axis(sum, 1, A1)))
        D1[isinf(D1) | isnan(D1)] = 0
        L1 = diag([1] * N) - D1 @ A1 @ D1
        
        D2 = diag(1.0 / sqrt(apply_along_axis(sum, 1, A2)))
        D2[isinf(D2) | isnan(D2)] = 0
        L2 = diag([1] * N) - D2 @ A2 @ D2
    else:
        L1 = A1
        L2 = A2
    
    _, decomp1 = eig(L1)
    poly1 = f(sort(decomp1))
    _, decomp2 = eig(L2)
    poly2 = f(sort(decomp2))
    delta = poly1 - poly2
    return (sum(delta ** p) ** (1 / p)) 

import numpy as np
from scipy.sparse import diags
from scipy.linalg import svd

def heat_distance(A1, A2, alpha, p=2):
    def f(x):
        return np.exp(-alpha*x)
        
    D1 = diags(1 / np.sqrt(np.sum(A1, axis=1)))
    D1[D1 == np.inf] = 0
    L1 = np.eye(A1.shape[0]) - D1 @ A1 @ D1
    
    D2 = diags(1 / np.sqrt(np.sum(A2, axis=1)))
    D2[D2 == np.inf] = 0
    L2 = np.eye(A2.shape[0]) - D2 @ A2 @ D2
    
    decomp1 = svd(L1, full_matrices=False)
    poly1 = np.diag(f(decomp1[1]))
    decomp2 = svd(L2, full_matrices=False)
    poly2 = np.diag(f(decomp2[1]))
    delta = decomp1[0] @ poly1 @ decomp1[2] - decomp2[0] @ poly2 @ decomp2[2]
    
    return (np.sum(delta**p))**(1/p)
