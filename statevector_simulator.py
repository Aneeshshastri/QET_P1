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
        self.state = None  # initialize to |0...0>

    def x(self, qubit: int) -> None:
        """Apply the Pauli-X (NOT) gate to the given qubit."""
        raise NotImplementedError

    def h(self, qubit: int) -> None:
        """Apply the Hadamard gate to the given qubit."""
        raise NotImplementedError

    def z(self, qubit: int) -> None:
        """Apply the Pauli-Z gate to the given qubit."""
        raise NotImplementedError

    def cnot(self, control: int, target: int) -> None:
        """Apply a CNOT gate with the given control and target qubits."""
        raise NotImplementedError

    def cz(self, control: int, target: int) -> None:
        """Apply a controlled-Z gate with the given control and target qubits."""
        raise NotImplementedError

    def half_entropy(self) -> float:
        """
        Compute the entanglement entropy across the half bipartition.

        Returns:
            The half-cut entanglement entropy in bits.
        """
        raise NotImplementedError

    def get_statevector(self) -> np.ndarray:
        """Return the current statevector as a NumPy array."""
        raise NotImplementedError

    def get_probabilities(self) -> np.ndarray:
        """Return the current probabilities as a Numpy array"""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the simulator back to the |0...0> state."""
        raise NotImplementedError

    def grover_2qubit(self, marked_state: int) -> None:
        """
        Run Grover's search algorithm on 2 qubits using your simulator.

        The marked state should have probability ~1 after one iteration.

        Args:
            marked_state: Index (0 to 3) of the state the oracle marks.
        """
        raise NotImplementedError


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
