from config import *
import circuitgraph as cg
from G_distance import *
from pathlib import Path
import pickle


def distance(hete_adj_matrix_A,hete_adj_matrix_B, distance_type):

    distances = {}
    for key in hete_adj_matrix_A.keys():
        num_node_A = hete_adj_matrix_A[key].shape[0]
        num_node_B = hete_adj_matrix_B[key].shape[0]
        if distance_type in ['hamming', 'jaccard','poly']:
            # enumerate all permutation matrix
            # for each permutation matrix, calculate the hamming distance
            # return the minimum hamming distance
            import itertools
            permutations = np.array(list(itertools.permutations([x for x in range(num_node_A)])))
            # random pick 100 permutations if the number of permutations is larger than 100
            if permutations.shape[0] > 10:
                permutations = permutations[np.random.choice(permutations.shape[0], 100, replace=False), :]
            # Iterate through all permutations
            min_distance = np.inf
            for permutation in permutations:
                P = np.zeros((num_node_A, num_node_A))
                # Fill the matrix with the permutation
                P[np.arange(num_node_A), permutation] = 1
                if distance_type == 'hamming':
                    dist = hamming_based_distance(P @ hete_adj_matrix_A[key], hete_adj_matrix_B[key])
                elif distance_type == 'jaccard':
                    dist = jaccard_based_distance(P @ hete_adj_matrix_A[key], hete_adj_matrix_B[key])
                elif distance_type == 'poly':
                    dist = poly_distance(P @ hete_adj_matrix_A[key], hete_adj_matrix_B[key],order_max = 3)
                if dist < min_distance:
                    min_distance = dist
            distances[key] = min_distance
        elif distance_type == 'eigen':
            distances[key] = eigen_distance(hete_adj_matrix_A[key], hete_adj_matrix_B[key])
        elif distance_type == 'heat':
            distances[key] = heat_distance(hete_adj_matrix_A[key], hete_adj_matrix_B[key],alpha = 1)
    return distances
        



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
        hete_adj_matrix[(circuit.nodes[edge[0]]['type'], circuit.nodes[edge[0]]['type'])][list(circuit.nodes).index(edge[0]), list(circuit.nodes).index(edge[1])] = 1

    hete_adj_matrices.append(hete_adj_matrix)

import timeit
distance_matrix = np.zeros((len(circuit_files), len(circuit_files)))
for i in range(len(circuit_files)):
    for j in range(i+1, len(circuit_files)):
        start = timeit.default_timer()
        print("Calculating distance between ", circuit_files[i].name, " and ", circuit_files[j].name, " ...")
        dist= distance(hete_adj_matrices[i], hete_adj_matrices[j], opt.distance_metric)
        distance_matrix[i,j] = sum(dist.values())
        distance_matrix[j,i] = distance_matrix[i,j]
        stop = timeit.default_timer()
        print("Time: ", stop - start, "; Distance: ", distance_matrix[i,j])

print(distance_matrix)


