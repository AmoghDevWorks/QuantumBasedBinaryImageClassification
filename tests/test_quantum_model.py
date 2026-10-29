import pytest

from src.quantum_model import build_vqc


def test_invalid_quantum_configuration():
    with pytest.raises(ValueError):
        build_vqc(3)


def test_circuit_construction_when_qiskit_is_available():
    pytest.importorskip("qiskit")
    _, circuit = build_vqc(2, maxiter=1, shots=32)
    assert circuit.num_qubits == 2
    assert circuit.depth() > 0
