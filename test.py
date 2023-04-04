
from config import *
import circuitgraph as cg
from G_distance import *
from pathlib import Path
import pickle
import tqdm
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sample data
object_names = ['A', 'B', 'C', 'D']
distance_matrix = np.array([[0, 2, 4, 6],
                            [2, 0, 3, 5],
                            [4, 3, 0, 2],
                            [6, 5, 2, 0]])

# create a pandas DataFrame to store the distance matrix
df = pd.DataFrame(distance_matrix, index=object_names, columns=object_names)

# plot the heatmap using seaborn
sns.heatmap(df, annot=True, cmap='coolwarm')

# show the plot
plt.savefig('./test.png')

exit()
save_path = Path("./result")  
save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / (opt.graph_data + '_' + opt.distance_metric +  '.pkl'), "rb") as f:
    G, names = pickle.load(f)
eigvals, eigvecs = np.linalg.eigh(G)
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
