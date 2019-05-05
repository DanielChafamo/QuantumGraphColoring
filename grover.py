import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
from graph import GraphColoring

from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit.tools.visualization import plot_histogram

# 
# [ X ] = NOT
# CNOT = XOR
# 
# 
# Multiple-Control Toffoli (MCT) Gate implement almost all booleans
# Espresso heuristic logic minimizer.


class GraphColorGrover(object):
  def __init__(self, gc, niter=100):
    self.gc = gc 
    self.niter = niter  

  def run(self):
    constraints = self.graphcover_constraints(self.gc, self.gc.ncolors)
    CNF = self.dimacs_format(constraints, self.gc.nnodes*self.gc.ncolors)
    oracle = LogicalExpressionOracle(CNF, optimization='espresso')
    grover = Grover(oracle, incremental=True, mct_mode='advanced')

    backend = BasicAer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=self.niter)
    self.result = grover.run(quantum_instance)
    return self.result

  @staticmethod
  def graphcover_constraints(gc, ncolors):
    constraints = []
    for i in range(gc.nnodes):
      # no more than one color per node 
      for j in range(ncolors):
        for k in range(j):
          constraints.append([-(i*ncolors+j+1), -(i*ncolors+k+1)])
      # at least one color per node 
      constraints.append([i*ncolors+j+1 for j in range(ncolors)])
    # different color for neighbours
    for i, j in gc.edges:
      for k in range(ncolors):
        constraints.append([-(i*ncolors+k+1), -(j*ncolors+k+1)])
    return constraints

  @staticmethod
  def dimacs_format(constraints, nvars): 
    dimacs = "p cnf {} {}\n".format(nvars, len(constraints))
    for c in constraints: 
      dimacs += " ".join([str(i) for i in c]) + " 0\n"
    return dimacs

  @staticmethod
  def extract_solution(result):
    if exact:
      i= np.where(result['eigvecs'][0])[0][0]
    else:
      p = result['eigvecs'][0]
      i = random.choices(range(len(p)), weights=[np.linalg.norm(i) for i in p])

    bitsolution = bin(i)[2:]
    pad = int(np.log2(result['eigvecs'].shape[1]))-len(bitsolution)
    return pad*'0' + bitsolution

  @staticmethod
  def visualize(result):
    # graph color vis
    plot_histogram(result['measurement'])
    plt.show()





