import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
import random


class Graph(object): 
  def __init__(self): 
    self.nnodes = 0
    self.edges = [] 

  def from_edge_list(self, edges, nnodes):
    self.edges = edges
    self.nnodes = nnodes 

  def rand_graph(self, nnodes, p=0.5):  
    self.nnodes = nnodes 
    for i in range(nnodes):
      for j in range(i+1):
        if random.rand() <= p:
          self.edges.append([i,j]) 
    return self.edges

  def render_graph(self):
    options = { 
      'node_size': 100, 
      'width': 3,
    }
    G = nx.Graph()
    G.add_edges_from(self.edges) 
    nx.draw(G, with_labels=True, **options)
    plt.show()

    


