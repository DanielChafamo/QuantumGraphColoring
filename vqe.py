import numpy as np 
import random
from qiskit.quantum_info import Pauli 
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import SPSA, L_BFGS_B, NELDER_MEAD, CG, COBYLA
from qiskit.aqua.components.variational_forms import RY, RYRZ
from qiskit import Aer
from qiskit.providers.ibmq import least_busy 
from qiskit import IBMQ


class GraphColorVQE(object):
  def __init__(self, gc, niter=10, verbose=0):
    self.graphcover = gc 
    self.niter = niter 
    self.verbose = verbose

  def run_sim(self):
    """ Use VQE on a simulator to determine ground energy eigenvectors of the 
    hamiltonian 

    """
    self.operator, var_form, opt = self.generate_VQE_args()

    backend = Aer.get_backend('statevector_simulator')
    quantum_instance = QuantumInstance(backend=backend)
    vqe = VQE(self.operator, var_form, opt) 

    self.result = vqe.run(quantum_instance)
    solution = self.extract_solution(self.result, False)
    return solution

  def run_exact(self):
    """ Use an exact eigensolver to determine ground energy eigenvectors of the 
    hamiltonian 

    """
    self.operator, var_form, opt = self.generate_VQE_args()

    exact_eigensolver = ExactEigensolver(self.operator, k=1)
    self.result = exact_eigensolver.run()

    solution = self.extract_solution(self.result, True)
    return solution

  def run_IBMQ(self):
    """ Use VQE on a real device to determine ground energy eigenvectors of the 
    hamiltonian 

    """
    self.operator, var_form, opt = self.generate_VQE_args()

    nqbits = self.operator.num_qubits
    IBMQ.load_accounts()
    backend = self.find_least_busy(nqbits)

    quantum_instance = QuantumInstance(backend=backend)
    vqe = VQE(self.operator, var_form, opt) 

    self.result = vqe.run(quantum_instance)
    solution = self.extract_solution(self.result, False)
    return solution

  def generate_VQE_args(self):
    """ Generate Operator, Variational Form and Optimizer for graph coloring problem

    """
    Hamiltonian = self.generate_ising_hamiltonian(self.graphcover)
    Operator = self.get_qubitops(Hamiltonian, self.verbose)

    var_form = RYRZ(num_qubits=Hamiltonian.shape[0], 
                    depth=5, entanglement="linear", 
                    initial_state=None)
    opt = SPSA(max_trials=self.niter)
    print("Operator with number of qubits: {}".format(Operator.num_qubits))

    return Operator, var_form, opt

  @staticmethod
  def find_least_busy(n_qubits=5):
    fltr = lambda x: x.configuration().n_qubits > n_qubits and not x.configuration().simulator
    large_enough_devices = IBMQ.backends(filters=fltr)
    backend = least_busy(large_enough_devices)
    print("Using backend: " + backend.name())

    return backend

  @staticmethod
  def generate_ising_hamiltonian(gc, cost=100., factor=100.):
    """Generate ising Hamiltonian for the given graph coloring problem. 

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
          H[i*gc.ncolors+c, j*gc.ncolors+c] = cost/factor 
          H[j*gc.ncolors+c, i*gc.ncolors+c] = cost/factor 
    return H

  @staticmethod
  def get_qubitops(H, verbose):
    """Generate Pauling based Hamiltonian operator for the graph coloring problem. 
    Returns:
        operator.Operator: operator for the Hamiltonian 
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
  def extract_solution(result, exact=False):
    """Extract a bit solution from the eigenvector generated in the VQE run
    Returns:
        string: solution of the graph cover in 01 format
    """
    if exact:
      i= np.where(result['eigvecs'][0])[0][0]
    else:
      p = result['eigvecs'][0] 
      i = max(range(len(p)), key=lambda i: abs(p[i]))

    bitsolution = bin(i)[2:]
    pad = int(np.log2(result['eigvecs'].shape[1]))-len(bitsolution)
    bitsolution = pad*'0' + bitsolution
    return bitsolution[::-1]

  def stats(self):
    """ Extract number of qbits, number of gates needed to run this instance 
    depth of circuit (number of ops on the critical path) circuit.depth()
    """
    nqbits = self.operator.num_qubits


    

