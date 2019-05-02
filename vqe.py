import numpy as np 
from qiskit.quantum_info import Pauli 
from qiskit_aqua import Operator, QuantumInstance
from qiskit_aqua.algorithms import VQE
from qiskit_aqua.components.optimizers import NELDER_MEAD, CG, COBYLA
from qiskit_aqua.components.variational_forms import RY, RYRZ


class GraphColorVQE(object):
  def __init__(self, graph, ncolors, niter=100):
    self.graph = graph
    self.ncolors = ncolors
    self.niter = niter 

  def generate_ising_hamiltonian(self, cost=10.):
    """Generate ising Hamiltonian for graph coloring problem. 
    Returns:
        numpy.ndarray: the ising Hamiltonian 
    """
    H = np.array([self.graph.nnodes*self.ncolors, self.graph.nnodes*self.ncolors])
    # one color per node
    for i in range(self.graph.nnodes):
      for c1 in range(self.ncolors):
        for c2 in range(self.ncolors):
          if c1 == c2:
            H[i*self.ncolors+c1, i*self.ncolors+c2] = -cost
          else:
            H[i*self.ncolors+c1, i*self.ncolors+c2] = cost
    # different color for connected edges
    for i, j in self.graph.edges:
      for c in range(self.ncolors):
        H[i*self.ncolors+c, i*self.ncolors+c] = cost/2 
    return H

  def get_qubitops(self, H):
    """Generate Pauling based Hamiltonian operator for the graph coloring problem. 
    Returns:
        operator.Operator, float: operator for the Hamiltonian 
    """
    num_nodes = H.shape[0]
    pauli_list = [] 
    for i in range(num_nodes):
        for j in range(i+1):
            if H[i, j] != 0:
                xp = np.zeros(num_nodes, dtype=np.bool)
                zp = np.zeros(num_nodes, dtype=np.bool)
                zp[i] = True
                zp[j] = True
                pauli_list.append([ H[i, j], Pauli(zp, xp)]) 
    self.nqbits = num_nodes
    return Operator(paulis=pauli_list) 

  def vqe(self):
    """Run variational quantum eigensolver

    """
    var_form = RYRZ(num_qubits=self.nqubits, 
                    depth=1, entanglement="full", 
                    initial_state=None)
    opt = CG(maxiter=self.niter)
    backend = Aer.get_backend('statevector_simulator')
    quantum_instance = QuantumInstance(backend=backend)
    vqe = VQE(P_Op, var_form, opt)
    return vqe.run(quantum_instance)

  def extract_solution(self, resbit):
    colormap = {}
    






