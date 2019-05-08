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
    self.pos = None

  def rand_graph(self, nnodes, p=0.5):  
    """ Generate a random graph  
    Args:
        nnodes: number of nodes in graph 
        p: probability of having an edge between any two nodes

    Returns:
        edges: list of edges in graph 
    """
    self.nnodes = nnodes 
    for i in range(nnodes):
      for j in range(i+1):
        if i!=j and np.random.rand() <= p:
          self.edges.append([i,j]) 
    return self.edges

  def edgenum_rand_graph(self, nnodes, nedges):
    """ Generate a random graph with given number of edges
    Args:
        nnodes: number of nodes in graph 
        nedges: number of edges 

    Returns:
        edges: list of edges in graph 
    """
    if nedges > nnodes*(nnodes-1)/2:
      print("Not possible to have {} edges amongst {} nodes".format(nedges, nnodes))
    self.nnodes = nnodes 
    for i in range(nedges):
      f,t = np.random.randint(0,self.nnodes,2) 
      while t == f or [t,f] in self.edges or [f,t] in self.edges:
        f,t = np.random.randint(0,self.nnodes,2) 
      self.edges.append([f,t])

    return self.edges

  def check_solution(self, solution):
    """ Assess whether the provided solution satisfies the graph coloring problem
    Args:
        solution (string): proposed solution 
            the first 'ncolors' bits correspond to the first node and so on
            index with value '1' gives asssigned color to that node

    """
    assigned = np.where(np.array(list(solution))=='1')[0] % self.ncolors 

    if self.nnodes != len(assigned):
      print("Multiple colors assigned to single node.")
      return False 
    for i, j in self.edges:
      if assigned[i] == assigned[j]:
        print("Edge ({}, {}) colored the same.".format(i,j))
        return False 
    return True

  def solution_from_bits(self, bits):
    """ Generates list of colors from a sequence of bits solution

    """
    colors = ['blue', 'green', 'red', 'yellow', 'black']
    idx = np.where(np.array(list(bits))=='1')[0] % self.ncolors 
    solution = [colors[i] for i in idx]
    return solution
    
  def render_graph(self, solution=None):
    """ Draws graph using networkx. Applies coloring given by solution if provided

    """
    if solution:
      if not self.check_solution(solution):
        print("Graph coloring not satisfied!")
      else:
        print("Graph coloring satisfied!")
      solution = self.solution_from_bits(solution) 
      if len(solution) != self.nnodes:
        return

    options = { 
      'nodelist': list(range(self.nnodes)),
      'node_size': 2000, 
      'width': 1,
      'alpha': 0.7,
      'node_color': solution
    }
    G = nx.Graph()
    G.add_edges_from(self.edges) 
    if self.pos is None:
      self.pos = nx.spring_layout(G)
    nx.draw(G, self.pos, with_labels=True, **options)
    plt.show()



