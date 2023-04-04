from config import *
import circuitgraph as cg
from G_distance import *
from pathlib import Path
import pickle
import tqdm
import random



# load ./saved_data/opt.graph_data.pkl if it exists, otherwise generate graphs from ./benchmark/opt.graph_data/*.v
# and save them into ./saved_data/opt.graph_data.pkl
save_path = Path("./saved_data")
save_path.mkdir(parents=True, exist_ok=True)
try:
    with open(save_path / opt.graph_data + 'pkl', "rb") as f:
        circuit_files = pickle.load(f)
except:
    # graph data is in ./benchmark/opt.graph_data/*.v, read them and generate graphs
    graph_path= Path("./benchmark") / opt.graph_data
    graph_files = list(graph_path.glob("*.v"))
    bbs = [cg.BlackBox("dff",["d","clk"],["o"])]
    circuit_files = [cg.from_file(f, blackboxes=bbs).graph for f in graph_files]
    # assign name to each graph
    for i, circuit in enumerate(circuit_files):
        # only preserve the file name (before .v and after /)
        circuit.name = graph_files[i].name.split(".")[0]
    # save the circuit files into ./saved_data/opt.graph_data
    save_path = Path("./saved_data")
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / (opt.graph_data + '.pkl'), "wb") as f:
        pickle.dump(circuit_files, f)

# Each graph is heterogenous, and encoded as a set of dictoinaries that map edge type to numpy array A
hete_adj_matrices = []
possible_edges = [(x,y) for x in addable_types for y in addable_types]
max_num_nodes = max([circuit.number_of_nodes() for circuit in circuit_files])
print(opt.graph_data,": Max number of nodes: ", max_num_nodes)

# sort circuit_files based on the number of nodes
circuit_files = sorted(circuit_files, key=lambda x: x.number_of_nodes())

import timeit
for graph_index in range(len(circuit_files)):
    name = circuit_files[graph_index].name
    print(name, '; circuit_files[graph_index].number_of_nodes() ', circuit_files[graph_index].number_of_nodes())
    start = timeit.default_timer()
    orig_label_map = [addable_types.index(circuit_files[graph_index].nodes[x]['type']) for x in circuit_files[graph_index].nodes]
    label_map,_, distance_map= WL_dist(circuit_files[graph_index],circuit_files[graph_index],addable_types)
    stop = timeit.default_timer()
    print(name, '; WL_dist Time: ', stop - start)
    
    # here, distance matrix is a node-by-node matrix, where distance_matrix[i,j] is the distance between node i and node j
    distance_matrix = np.zeros((circuit_files[graph_index].number_of_nodes(), circuit_files[graph_index].number_of_nodes()))
    for i in range(circuit_files[graph_index].number_of_nodes()):
        for j in range(circuit_files[graph_index].number_of_nodes()):
            distance_matrix[i,j] = distance_map[-1][label_map[-1][i], label_map[-1][j]]
            distance_matrix[j,i] = distance_matrix[i,j]
    # get the gram matrix G from the distance matrix, G_{ij} = (D_{1j}^2  + D_{i1}^2 - D_{ij}^2)/2
    G = np.zeros_like(distance_matrix)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            G[i,j] = (distance_matrix[0,j]**2 + distance_matrix[i,0]**2 - distance_matrix[i,j]**2)/2
    # eigenvalue decomposition to G
    start = timeit.default_timer()
    eigvals, eigvecs = np.linalg.eigh(G)
    stop = timeit.default_timer()
    print(name, '; Eigenvalue decomposition Time: ', stop - start)
    save_path = Path("./single-graph-result")
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / (name + '_' + opt.graph_data + '_' + opt.distance_metric +  '.pkl'), "wb") as f:
        pickle.dump((eigvals, eigvecs), f)

    X = eigvecs * sqrt(eigvals)
    X = np.matmul(eigvecs, np.diag(np.sqrt(eigvals)))
    # only keep the first 2 dimensions
    X = X[:,-2:]
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    # scatter plot X, where the color of each point  corresponding to its orig_label_map,i.e., color[i] = orig_label_map[i]
    plt.scatter(X[:,0], X[:,1], c=orig_label_map, cmap='tab10')
    plt.savefig(save_path / (name + '_' + opt.graph_data  + '_' + opt.distance_metric +  '.png'))



    