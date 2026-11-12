"""Six-qubit Qiskit portfolio circuit and exact Poisson-binomial reference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def validate_probabilities(probabilities: Sequence[float]) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or p.size != 6:
        raise ValueError("Stage 3 requires exactly six success probabilities.")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Success probabilities must be finite and in [0, 1].")
    return p


def poisson_binomial_distribution(probabilities: Sequence[float]) -> np.ndarray:
    """Return P(K=k), k=0..6, using an independent dynamic-programming recurrence."""
    p = validate_probabilities(probabilities)
    distribution = np.array([1.0], dtype=float)
    for probability in p:
        updated = np.zeros(distribution.size + 1, dtype=float)
        updated[:-1] += distribution * (1.0 - probability)
        updated[1:] += distribution * probability
        distribution = updated
    return distribution


def probability_at_least(probabilities: Sequence[float], threshold: int = 3) -> float:
    p = validate_probabilities(probabilities)
    if not 0 <= threshold <= len(p):
        raise ValueError("threshold must be between zero and six.")
    return float(poisson_binomial_distribution(p)[threshold:].sum())


def build_stage3_circuit(probabilities: Sequence[float]) -> QuantumCircuit:
    """Prepare independent Bernoulli qubits and measure success bits in matching clbits."""
    p = validate_probabilities(probabilities)
    circuit = QuantumCircuit(6, 6, name="stage3_six_customer_bernoulli")
    for q, probability in enumerate(p):
        circuit.ry(2.0 * np.arcsin(np.sqrt(probability)), q)
    circuit.measure(range(6), range(6))
    return circuit


def stage3_statevector_distribution(probabilities: Sequence[float]) -> np.ndarray:
    """Ideal computational-basis probabilities in integer/little-endian bit order."""
    state = Statevector.from_instruction(build_stage3_circuit(probabilities).remove_final_measurements(inplace=False))
    result = np.abs(state.data) ** 2
    return result / result.sum()


def successes_from_counts(counts: dict[str, int], *, n_qubits: int = 6) -> np.ndarray:
    """Aggregate counts by Hamming weight; Qiskit displays q[n-1]...q[0]."""
    result = np.zeros(n_qubits + 1, dtype=np.int64)
    for bitstring, count in counts.items():
        normalized = bitstring.replace(" ", "")
        if len(normalized) != n_qubits or set(normalized) - {"0", "1"}:
            raise ValueError(f"Invalid measured bitstring: {bitstring!r}")
        result[normalized.count("1")] += int(count)
    return result


def probability_from_counts(counts: dict[str, int], threshold: int = 3) -> float:
    successes = successes_from_counts(counts)
    shots = int(successes.sum())
    if shots < 1 or not 0 <= threshold <= 6:
        raise ValueError("Counts must contain shots and threshold must be between zero and six.")
    return float(successes[threshold:].sum() / shots)


def probability_from_distribution(distribution: np.ndarray, threshold: int = 3) -> float:
    if distribution.shape != (64,) or not 0 <= threshold <= 6:
        raise ValueError("Expected a 64-state six-qubit distribution and valid threshold.")
    return float(sum(value for index, value in enumerate(distribution) if index.bit_count() >= threshold))
