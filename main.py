from config import *
import circuitgraph as cg
from G_distance import *
from pathlib import Path
import pickle

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
    with open(save_path / opt.graph_data + 'pkl', "wb") as f:
        pickle.dump(circuit_files, f)

# Each graph is heterogenous, and encoded as a set of dictoinaries that map edge type to numpy array A
hete_adj_matrices = []
possible_edges = [(x,y) for x in addable_types for y in addable_types]
max_num_nodes = max([circuit.number_of_nodes() for circuit in circuit_files])
for circuit in circuit_files:
    hete_adj_matrix = {possible_edge: np.zeros((max_num_nodes, max_num_nodes)) for possible_edge in possible_edges}
    for edge in circuit.edges:
        hete_adj_matrix[(circuit.nodes[edge[0]]['type'], circuit.nodes[edge[0]]['type'])][circuit.nodes.index(edge[0]), circuit.nodes.index(edge[1])] = 1