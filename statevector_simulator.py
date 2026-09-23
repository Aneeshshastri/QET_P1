"""
Statevector Simulator - Core Selection Problem Statement 1

Build a statevector simulator for an n-qubit system using NumPy.

Your simulator must support:
    - Single qubit gates: X, H, Z
    - Two qubit gates: CNOT, CZ
    - Half entropy of the system: the von-Neumann entanglement entropy

You may ONLY use NumPy (and the Python standard library). No Qiskit,
no other quantum computing libraries.
"""

import numpy as np


class StatevectorSimulator:
    """A statevector simulator for an n-qubit quantum system."""

    def __init__(self, num_qubits: int):
        """
        Initialize the simulator in the |0...0> state.

        Args:
            num_qubits: Number of qubits n. The statevector must be a
                complex NumPy array of size 2**n.
        """
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=np.complex128)  
        self.state[0] = 1 # initialize to |0...0>
        #useful stuff
        self.I=np.array([[1.0,0.0],
                         [0.0,1.0]], dtype=np.complex128)
        self.X=np.array([[0.0,1.0],
                         [1.0,0.0]], dtype=np.complex128)
        self.H=np.array([[1.0,1.0],
                         [1.0,-1.0]], dtype=np.complex128)/np.sqrt(2)
        self.Z=np.array([[1.0,0.0],
                         [0.0,-1.0]], dtype=np.complex128)
        self.C0=np.array([[1.0,0.0],
                          [0.0,0.0]],dtype=np.complex128) # |0><0|
        self.C1=np.array([[0.0,0.0],
                          [0.0,1.0]],dtype=np.complex128) # |0><0|
    def x(self, qubit: int) -> None:
        X=1
        for i in range(self.num_qubits):
            if i==qubit:
                X=np.kron(X,self.X)
            else:
                X=np.kron(X,self.I)
        self.state=np.dot(X,self.state)



    def h(self, qubit: int) -> None:
        X=1
        for i in range(self.num_qubits):
            if i==qubit:
                X=np.kron(X,self.H)
            else:
                X=np.kron(X,self.I)
        self.state=np.dot(X,self.state)

    def z(self, qubit: int) -> None:
        X=1
        for i in range(self.num_qubits):
            if i==qubit:
                X=np.kron(X,self.Z)
            else:
                X=np.kron(X,self.I)
        self.state=np.dot(X,self.state)

    def cnot(self, control: int, target: int) -> None:
        X0=1
        X1=1
        for i in range(self.num_qubits):
            if i==control:
                X0=np.kron(X0,self.P0)
                X1=np.kron(X1,self.P1)
            elif i==target:
                X0=np.kron(X0,self.I)
                X1=np.kron(X1,self.X)
            else:
                X0=np.kron(X0,self.I)
                X1=np.kron(X1,self.I)
        self.state=np.dot(X0+X1,self.state)

    def cz(self, control: int, target: int) -> None:
        X0=1
        X1=1
        for i in range(self.num_qubits):
            if i==control:
                X0=np.kron(X0,self.P0)
                X1=np.kron(X1,self.P1)
            elif i==target:
                X0=np.kron(X0,self.I)
                X1=np.kron(X1,self.Z)
            else:
                X0=np.kron(X0,self.I)
                X1=np.kron(X1,self.I)
        self.state=np.dot(X0+X1,self.state)
        

    def half_entropy(self) -> float:
        """
        Compute the entanglement entropy across the half bipartition.

        Returns:
            The half-cut entanglement entropy in bits.
        """
        "I'll split the system as the first and last n/2 qubits"

        subsystem_matrix=self.state.reshape((2**(self.num_qubits//2),2**((self.num_qubits+1)//2)))
        _, S, _ = np.linalg.svd(subsystem_matrix)
        eigenvalues = S**2
        
      
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues)) #Shannon?

    def get_statevector(self) -> np.ndarray:
        return self.state

    def get_probabilities(self) -> np.ndarray:
        return (np.abs(self.state))**2

    def reset(self) -> None:
        self.state=np.zeros(2**self.num_qubits, dtype=np.complex128)  
        self.state[0]=1

    def grover_2qubit(self, marked_state: int) -> None:
        """
        Run Grover's search algorithm on 2 qubits using your simulator.

        The marked state should have probability ~1 after one iteration.

        Args:
            marked_state: Index (0 to 3) of the state the oracle marks.
        """
        # set the state vector to its mean position
        N=2**self.num_qubits
        self.state=np.ones(N,dtype=np.complex128)/np.sqrt(N)

        Oracle=np.identity(N)
        Oracle[marked_state][marked_state]=-1
        diffuser=(np.ones((N,N),dtype=np.complex128)*2)/N - np.identity(N)

        num_iter=int(np.floor(np.pi/4*np.sqrt(N)))

        for i in range(num_iter):
            self.state= np.dot(diffuser,np.dot(Oracle,self.state))



if __name__ == "__main__":
    # Run 2-qubit Grover search marking the state |11> (index 3).
    simulator = StatevectorSimulator(2)
    marked_state = 3
    simulator.grover_2qubit(marked_state)
    final_state = simulator.get_statevector()
    probabilities = simulator.get_probabilities()

    print("Final statevector:", final_state)
    print("Probabilities:", probabilities)
    print("Measured state:", np.argmax(probabilities))
