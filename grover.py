import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
from graph import GraphColoring

from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy 
from qiskit import IBMQ

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

  def run_sim(self):
    """ Run grover's on a quantum simulator 

    """
    self.grover = self.generate_grover()

    nqbits = self.oracle.circuit.width()
    print("Oracle with number of qubits: {}".format(nqbits))

    backend = BasicAer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=self.niter)
    self.result = self.grover.run(quantum_instance) 

    return self.result['top_measurement'][::-1]

  def run_IBMQ(self):
    """ Run grover's on a real device

    """
    self.grover = self.generate_grover()

    nqbits = self.oracle.circuit.width()
    print("Oracle with number of qubits: {}".format(nqbits))

    IBMQ.load_accounts()
    backend = self.find_least_busy(nqbits)

    quantum_instance = QuantumInstance(backend, shots=self.niter)
    self.result = self.grover.run(quantum_instance) 

    return self.result['top_measurement'][::-1]

  def generate_grover(self, incr=False, n_iters=None):
    """ Generate grover oracle and the full grover circuit using qiskit's 
    implementation

    """
    if n_iters is None:
      n_iters = int(2**((self.gc.nnodes*self.gc.ncolors - 1)/2.))
    constraints = self.graphcover_constraints()
    CNF = self.dimacs_format(constraints, self.gc.nnodes*self.gc.ncolors)
    self.oracle = LogicalExpressionOracle(CNF, optimization='espresso')
    
    return Grover(self.oracle, incremental=incr, num_iterations=n_iters, mct_mode='advanced')

  def graphcover_constraints(self):
    """ Render Graph Coloring problem into a set of boolean constraints whose 
    satisfaction will give a solution for the problem.

    Returns:
        (int list) list: each entry is a disjunctive constraint 
    """
    nnodes, ncolors = self.gc.nnodes, self.gc.ncolors
    constraints = []
    for i in range(nnodes):
      # no more than one color per node 
      for j in range(ncolors):
        for k in range(j):
          constraints.append([-(i*ncolors+j+1), -(i*ncolors+k+1)])
      # at least one color per node 
      constraints.append([i*ncolors+j+1 for j in range(ncolors)])

    # different color for neighbours
    for i, j in self.gc.edges:
      for k in range(ncolors):
        constraints.append([-(i*ncolors+k+1), -(j*ncolors+k+1)])

    return constraints

  def stats(self):
    """ Extract number of qbits, number of gates needed to run this instance 
    # depth of circuit (number of ops on the critical path) circuit.depth()

    """
    circuit = self.oracle.circuit
    nqubits = circuit.width()
    ops = groverGC.oracle.circuit.count_ops()
    total_nops = sum(ops.values())

  def visualize(self, nsamples=20): 
    """ Plot a histogram of the top 'nsamples' results from the grover run
    
    Returns:
        dict: top nsamples solutions and their count
    """
    d = self.result['measurement']
    ks = sorted(d.keys(), key=lambda a:d[a])[-nsamples:]
    d = {k:d[k] for k in ks} 
    plot_histogram(self.result['measurement']) 

    return d

  @staticmethod
  def dimacs_format(constraints, nvars): 
    """ Translate set of boolean constraints into a CNF in dimacs, the format
    accepted by qiskit's  LogicalExpressionOracle

    """
    dimacs = "p cnf {} {}\n".format(nvars, len(constraints))
    for c in constraints: 
      dimacs += " ".join([str(i) for i in c]) + " 0\n"

    return dimacs

  @staticmethod
  def find_least_busy(n_qubits=4):
    fltr = lambda x: x.configuration().n_qubits > n_qubits and not x.configuration().simulator
    large_enough_devices = IBMQ.backends(filters=fltr)
    backend = least_busy(large_enough_devices)
    print("Using backend: " + backend.name())

    return backend








