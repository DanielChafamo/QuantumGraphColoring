import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt  


class GraphColoring(object): 
  def __init__(self, ncolors, edges=None, nnodes=0):
    if edges is None:
        edges = []
    self.edges = edges
    self.nnodes = nnodes 
    self.ncolors = ncolors

  def rand_graph(self, nnodes, p=0.5):  
    self.nnodes = nnodes 
    for i in range(nnodes):
      for j in range(i+1):
        if i!=j and np.random.rand() <= p:
          self.edges.append([i,j]) 
    return self.edges

  def render_graph(self): 
    options = { 
      'nodelist': list(range(self.nnodes)),
      'node_size': 3000, 
      'width': 3, 
    }
    G = nx.Graph()
    G.add_edges_from(self.edges) 
    nx.draw(G, with_labels=True, **options)
    plt.show()
