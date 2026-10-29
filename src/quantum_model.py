"""Qiskit VQC construction and training.

The imports are intentionally local so the rest of the application can still
show its overview if optional quantum dependencies are not installed.
"""

from time import perf_counter


def build_vqc(n_qubits: int, maxiter: int = 20, shots: int = 256, seed: int = 42):
    if n_qubits not in (2, 4, 8):
        raise ValueError("n_qubits must be 2, 4, or 8")
    if maxiter < 1 or shots < 1:
        raise ValueError("maxiter and shots must be positive")
    from qiskit.circuit.library import TwoLocal, ZFeatureMap
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_aer.primitives import SamplerV2
    from qiskit_machine_learning.algorithms import VQC

    feature_map = ZFeatureMap(feature_dimension=n_qubits, reps=1).decompose()
    ansatz = TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz",
        reps=1,
        entanglement="linear",
    ).decompose()
    sampler = SamplerV2(default_shots=shots, seed=seed)
    # SciPy's COBYLA requires MAXFUN to be at least the number of trainable
    # parameters plus two. Keep the requested budget when it is sufficient,
    # otherwise raise it to the smallest valid value.
    optimizer = COBYLA(maxiter=max(maxiter, ansatz.num_parameters + 2))
    model = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
    )
    return model, feature_map.compose(ansatz)


def train_quantum_model(x_train, y_train, n_qubits=4, maxiter=20, shots=256, seed=42):
    model, circuit = build_vqc(n_qubits, maxiter=maxiter, shots=shots, seed=seed)
    start = perf_counter()
    model.fit(x_train[:, :n_qubits], y_train)
    return model, circuit, perf_counter() - start
