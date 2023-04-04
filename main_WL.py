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
for circuit in circuit_files:
    hete_adj_matrix = {possible_edge: np.zeros((max_num_nodes, max_num_nodes)) for possible_edge in possible_edges}
    for edge in circuit.edges:
        hete_adj_matrix[(circuit.nodes[edge[0]]['type'], circuit.nodes[edge[1]]['type'])][list(circuit.nodes).index(edge[0]), list(circuit.nodes).index(edge[1])] = 1

    hete_adj_matrices.append(hete_adj_matrix)

import timeit
distance_matrix = np.zeros((len(circuit_files), len(circuit_files)))
for i in range(len(circuit_files)):
    for j in range(i+1, len(circuit_files)):
        start = timeit.default_timer()
        print("Calculating distance between ", circuit_files[i].name, " and ", circuit_files[j].name, " ...")
        label_map_A, label_map_B, distance_map= WL_dist(circuit_files[i],circuit_files[j],addable_types)
        dist = single_set_dist(label_map_A[-1], label_map_B[-1],distance_map[-1])
        distance_matrix[i,j] = dist
        distance_matrix[j,i] = distance_matrix[i,j]
        stop = timeit.default_timer()
        print("Time: ", stop - start, "; Distance: ", distance_matrix[i,j])

# sort names first, and sort distance matrix accordingly (for distance matrix, both rows and cols are sorted)
names = [circuit.name for circuit in circuit_files]
names = np.array(names)
distance_matrix = distance_matrix[names.argsort()]
distance_matrix = distance_matrix[:,names.argsort()]
names = names[names.argsort()]

# get the gram matrix G from the distance matrix, G_{ij} = (D_{1j}^2  + D_{i1}^2 - D_{ij}^2)/2
G = np.zeros_like(distance_matrix)
for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        G[i,j] = (distance_matrix[0,j]**2 + distance_matrix[i,0]**2 - distance_matrix[i,j]**2)/2
# eigenvalue decomposition to G
eigvals, eigvecs = np.linalg.eigh(G)


save_path = Path("./result")
save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / (opt.graph_data + '_' + opt.distance_metric +  '.pkl'), "wb") as f:
    pickle.dump((G,names), f)

X = eigvecs * sqrt(eigvals)
X = np.matmul(eigvecs, np.diag(np.sqrt(eigvals)))
# only keep the first 2 dimensions
X = X[:,-2:]
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# plot the graph and label each node by graph name
plt.scatter(X[:,0], X[:,1])
for i, txt in enumerate(names):
    plt.annotate(txt, (X[i,0], X[i,1]))
plt.savefig(save_path / (opt.graph_data  + '_' + opt.distance_metric +  '.png'))

# plot distance matrix
import pandas as pd
# create a pandas DataFrame to store the distance matrix
df = pd.DataFrame(distance_matrix, index=names, columns=names)
# plot the heatmap using seaborn
sns.heatmap(df, annot=True, cmap='coolwarm')
# show the plot
save_path = Path("./distance_plot")
save_path.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path / (opt.graph_data  + '_' + opt.distance_metric +  '.png'))







