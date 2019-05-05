import numpy as np 
import random
from qiskit.quantum_info import Pauli 
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import L_BFGS_B, NELDER_MEAD, CG, COBYLA
from qiskit.aqua.components.variational_forms import RY, RYRZ
from qiskit import Aer


class GraphColorVQE(object):
  def __init__(self, gc, niter=10, verbose=0):
    self.graphcover = gc 
    self.niter = niter 
    self.verbose = verbose

  def run(self, exact=False):
    Hamiltonian = self.generate_ising_hamiltonian(self.graphcover)
    Operator = self.get_qubitops(Hamiltonian, self.verbose)

    if exact:
      exact_eigensolver = ExactEigensolver(Operator, k=1)
      self.result = exact_eigensolver.run()
    else:
      self.result = self.vqe(Operator, self.niter, Hamiltonian.shape[0])

    solution = self.extract_solution(self.result, exact)
    return solution

  @staticmethod
  def generate_ising_hamiltonian(gc, cost=100.):
    """Generate ising Hamiltonian for graph coloring problem. 
    Returns:
        numpy.ndarray: the ising Hamiltonian 
    """
    H = np.zeros([gc.nnodes*gc.ncolors, gc.nnodes*gc.ncolors])
    # one color per node
    for i in range(gc.nnodes):
      for c1 in range(gc.ncolors):
        for c2 in range(gc.ncolors):
          if c1 == c2:
            H[i*gc.ncolors+c1, i*gc.ncolors+c2] = -cost
          else:
            H[i*gc.ncolors+c1, i*gc.ncolors+c2] = cost

    # different color for connected edges
    for i, j in gc.edges:
      if i != j:
        for c in range(gc.ncolors):
          H[i*gc.ncolors+c, j*gc.ncolors+c] = cost/100 
          H[j*gc.ncolors+c, i*gc.ncolors+c] = cost/100 
    return H

  @staticmethod
  def get_qubitops(H, verbose):
    """Generate Pauling based Hamiltonian operator for the graph coloring problem. 
    Returns:
        operator.Operator, float: operator for the Hamiltonian 
    """
    num_nodes = H.shape[0]
    pauli_list = [] 
    s = ""
    for i in range(num_nodes):
        xp = np.zeros(num_nodes, dtype=np.bool)
        zp = np.zeros(num_nodes, dtype=np.bool)
        zp[i] = True
        pauli_list.append([ H[i, i], Pauli(zp, xp)]) 
        s += ' {}*Z[{}]'.format(H[i,i], i)
        for j in range(i):
            if H[i, j] != 0:
                xp = np.zeros(num_nodes, dtype=np.bool)
                zp = np.zeros(num_nodes, dtype=np.bool)
                zp[i] = True
                zp[j] = True
                pauli_list.append([ H[i, j], Pauli(zp, xp)]) 
                s += ' + {}*Z[{}]*Z[{}]'.format(H[i,j], i, j) 
    if verbose > 0:
      print(s)
    return Operator(paulis=pauli_list) 

  @staticmethod
  def vqe(Operator, niter, nqubits):
    """Run variational quantum eigensolver

    """
    var_form = RYRZ(num_qubits=nqubits, 
                    depth=1, entanglement="full", 
                    initial_state=None)
    opt = CG(maxiter=niter)
    backend = Aer.get_backend('statevector_simulator')
    quantum_instance = QuantumInstance(backend=backend)
    vqe = VQE(Operator, var_form, opt)
    return vqe.run(quantum_instance)

  @staticmethod
  def extract_solution(result, exact=False):
    if exact:
      i= np.where(result['eigvecs'][0])[0][0]
    else:
      p = result['eigvecs'][0]
      i = random.choices(range(len(p)), weights=[np.linalg.norm(i) for i in p])

    bitsolution = bin(i)[2:]
    pad = int(np.log2(result['eigvecs'].shape[1]))-len(bitsolution)
    bitsolution = pad*'0' + bitsolution
    return bitsolution[::-1]

    

