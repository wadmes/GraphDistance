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
    # decomp1 is a diagnoal matrix, we reduce it to a vector
    poly1 = f(sort(decomp1.diagonal()))
    _, decomp2 = eig(L2)
    poly2 = f(sort(decomp2.diagonal()))
    delta = poly1 - poly2
    return sum(delta ** p) ** (1 / p)

import numpy as np
from scipy.sparse import diags
from scipy.linalg import svd

def heat_distance(A1, A2, alpha, p=2):
    def f(x):
        return np.exp(-alpha*x)
        
    # let D1 be the diagonal matrix with the sum of the rows of A1 on the diagonal
    D1 = diags(1 / np.sqrt(np.sum(A1, axis=1)))
    D1 = np.array(D1.todense())
    D1[D1 == np.inf] = 0
    L1 = np.eye(A1.shape[0]) - D1 @ A1 @ D1
    
    D2 = diags(1 / np.sqrt(np.sum(A2, axis=1)))
    D2 = np.array(D2.todense())
    D2[D2 == np.inf] = 0
    L2 = np.eye(A2.shape[0]) - D2 @ A2 @ D2
    
    decomp1 = svd(L1, full_matrices=False)
    poly1 = np.diag(f(decomp1[1]))
    decomp2 = svd(L2, full_matrices=False)
    poly2 = np.diag(f(decomp2[1]))
    delta = decomp1[0] @ poly1 @ decomp1[2] - decomp2[0] @ poly2 @ decomp2[2]
    
    return (np.sum(delta**p))**(1/p)


# node_label_A is a tuple of ([node_index], successors_index, predecessors_index) of node A
# node_label_B is a tuple of ([node_index], successors_index, predecessors_index) of node B
# distance_map is a list of k arrays, where [i,j] represents the distance between label i and label j
def multi_set_dist(node_label_A, node_label_B, distance_map):
    dist = 0
    dist += distance_map[node_label_A[0], node_label_B[0]]
    dist += single_set_dist(node_label_A[1], node_label_B[1], distance_map)
    dist += single_set_dist(node_label_A[2], node_label_B[2], distance_map)
    return dist

# A is the multiset, B is another multiset, distance_map is the distance map
def single_set_dist(A,B,distance_map):
    if len(A) == 0 or len(B) == 0:
        if len(A) == 0 and len(B) == 0:
            return 0
        else:
            return 1
    # only_in_a is the set of successors in A but not in B
    only_in_a = []
    only_in_a.extend(A)
    for item in B:
        try: 
            only_in_a.remove(item)
        except:
            pass
    only_in_b = []
    only_in_b.extend(B)
    for item in A:
        try:
            only_in_b.remove(item)
        except:
            pass
    dist = 0
    if len(only_in_a) == 0:
        dist = len(only_in_b) / (len(A)*len(B))
    elif len(only_in_b) == 0:
        dist = len(only_in_a) / (len(A)*len(B))
    else:
        for i in only_in_a:
            for j in only_in_b:
                dist += distance_map[i,j] / (len(A)*len(B))
    return dist

# A B are networkx graph
# node_A, node_B are the root node in A and B, respectively
# label_map is a list of k lists, where each list's index is the node index, value is the numberde label
# distance_map is a list of k arrays, where [i,j] represents the distance between label i and label j
# k is the number of iterations
# addable_types is defined in config.py
def WL_dist(A,B,addable_types,k = 4):
    iter = 0
    label_map_A = [[addable_types.index(A.nodes[x]['type']) for x in A.nodes]]
    label_map_B = [[addable_types.index(B.nodes[x]['type']) for x in B.nodes]]
    distance_map = [np.ones((len(addable_types),len(addable_types)))]
    distance_map[0] = distance_map[0] - np.eye(len(addable_types)) 
    while iter < k:
        iter += 1
        multiset2label = {}
        label2multiset = []
        label_map_A.append([])
        label_map_B.append([])
        
        for node_index, node in enumerate(A.nodes):
            # each node is associated with three multi-sets
            # one for the node itself, one for its predecessors, one for its successors
            # the multi-set is represented by a list of labels
            # the list is sorted in ascending order
            node_label = [[],[],[]]
            node_label[0] = label_map_A[iter-1][node_index]
            for pred in A.predecessors(node):
                node_label[1].append(label_map_A[iter-1][list(A.nodes).index(pred)])
            for succ in A.successors(node):
                node_label[2].append(label_map_A[iter-1][list(A.nodes).index(succ)])
            node_label[1].sort()
            node_label[1] = tuple(node_label[1]) 
            node_label[2].sort()
            node_label[2] = tuple(node_label[2])
            if tuple(node_label) not in list(multiset2label.keys()):
                multiset2label[tuple(node_label)] = len(label2multiset)
                label2multiset.append(node_label)
            label_map_A[-1].append(multiset2label[tuple(node_label)])
        for node_index, node in enumerate(B.nodes):
            node_label = [[],[],[]]
            node_label[0] = label_map_B[iter-1][node_index]
            for pred in B.predecessors(node):
                node_label[1].append(label_map_B[iter-1][list(B.nodes).index(pred)])
            for succ in B.successors(node):
                node_label[2].append(label_map_B[iter-1][list(B.nodes).index(succ)])
            node_label[1].sort()
            node_label[1] = tuple(node_label[1]) 
            node_label[2].sort()
            node_label[2] = tuple(node_label[2])
            if tuple(node_label) not in list(multiset2label.keys()):
                multiset2label[tuple(node_label)] = len(label2multiset)
                label2multiset.append(node_label)
            label_map_B[-1].append(multiset2label[tuple(node_label)])

        print("iteration: ", iter, "len(multiset2label): ", len(multiset2label))
        # update the distance map
        distance_map.append(np.zeros([len(label2multiset),len(label2multiset)]))
        for i in range(len(label2multiset)):
            for j in range(len(label2multiset)):
                if i == j:
                    distance_map[iter][i,j] = 0
                else:
                    distance_map[iter][i,j] = multi_set_dist(label2multiset[i],label2multiset[j],distance_map[iter-1])
    return label_map_A, label_map_B, distance_map