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

  def check_solution(self, solution):
    assigned = np.where(np.array(list(bits))=='1')[0] % self.ncolors 
    if len(self.nnodes) != idx:
      return False 
    for i, j in self.edges:
      if assigned[i] == assigned[j]:
        return False 
    return True

  def solution_from_bits(self, bits):
    colors = ['blue', 'green', 'red', 'yellow', 'black']
    idx = np.where(np.array(list(bits))=='1')[0] % self.ncolors 
    solution = [colors[i] for i in idx]
    return solution
    
  def render_graph(self, solution=None):
    if type(solution) == str:
      solution = self.solution_from_bits(solution)
    if solution and len(solution) != self.nnodes:
      print("Multiple colors assigned to single node.")
      return
    if solution and not self.check_solution(solution):
      print("Graph coloring not satisfied!")

    options = { 
      'nodelist': list(range(self.nnodes)),
      'node_size': 2000, 
      'width': 1,
      'alpha': 0.7,
      'node_color': solution
    }
    G = nx.Graph()
    G.add_edges_from(self.edges) 
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, **options)
    plt.show()


