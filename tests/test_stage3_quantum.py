import itertools

import numpy as np
import pytest

from atomiq.stage3_quantum import (
    build_stage3_circuit,
    poisson_binomial_distribution,
    probability_at_least,
    probability_from_counts,
    stage3_statevector_distribution,
    successes_from_counts,
)


def test_stage3_ry_preparation_has_requested_per_qubit_success_probabilities():
    probabilities = [0.05, 0.2, 0.35, 0.55, 0.8, 0.95]
    distribution = stage3_statevector_distribution(probabilities)
    marginals = [sum(p for mask, p in enumerate(distribution) if (mask >> q) & 1) for q in range(6)]
    assert marginals == pytest.approx(probabilities, abs=1e-12)
    assert len(build_stage3_circuit(probabilities).data) == 12


def test_stage3_poisson_binomial_matches_independent_enumeration():
    probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    exact = poisson_binomial_distribution(probabilities)
    brute = np.zeros(7)
    for outcomes in itertools.product((0, 1), repeat=6):
        probability = np.prod([p if outcome else 1 - p for p, outcome in zip(probabilities, outcomes)])
        brute[sum(outcomes)] += probability
    assert exact == pytest.approx(brute, abs=1e-12)
    assert probability_at_least(probabilities, 3) == pytest.approx(brute[3:].sum())


def test_counts_use_qiskit_little_endian_clbit_order_and_success_threshold():
    counts = {"100000": 3, "000001": 2, "001100": 1}
    by_weight = successes_from_counts(counts)
    assert by_weight.tolist() == [0, 5, 1, 0, 0, 0, 0]
    assert probability_from_counts(counts, threshold=3) == 0.0
    # Display order reverses register order: this is q5=1 for the first key,
    # q0=1 for the second; both have exactly one success.
    assert int("100000", 2).bit_count() == 1
    assert int("000001", 2).bit_count() == 1


def test_six_customer_circuit_mapping_angles_joint_distribution_and_unique_marginals():
    probabilities = np.array([0.071, 0.193, 0.347, 0.526, 0.731, 0.914])
    circuit = build_stage3_circuit(probabilities)
    assert circuit.num_qubits == 6
    assert circuit.num_clbits == 6
    assert sum(item.operation.name == "measure" for item in circuit.data) == 6
    rotations = [item for item in circuit.data if item.operation.name == "ry"]
    assert len(rotations) == 6
    for qubit, (instruction, probability) in enumerate(zip(rotations, probabilities, strict=True)):
        assert circuit.find_bit(instruction.qubits[0]).index == qubit
        assert instruction.operation.params[0] == pytest.approx(2 * np.arcsin(np.sqrt(probability)))
    measurements = [item for item in circuit.data if item.operation.name == "measure"]
    for qubit, instruction in enumerate(measurements):
        assert circuit.find_bit(instruction.qubits[0]).index == qubit
        assert circuit.find_bit(instruction.clbits[0]).index == qubit

    quantum = stage3_statevector_distribution(probabilities)
    classical = np.zeros(64)
    for outcomes in itertools.product((0, 1), repeat=6):
        index = sum(bit << q for q, bit in enumerate(outcomes))
        classical[index] = np.prod([
            p if bit else 1 - p for p, bit in zip(probabilities, outcomes, strict=True)
        ])
    assert quantum.sum() == pytest.approx(1.0, abs=1e-14)
    assert quantum == pytest.approx(classical, abs=1e-13)
    marginals = [sum(value for index, value in enumerate(quantum) if (index >> q) & 1) for q in range(6)]
    assert marginals == pytest.approx(probabilities, abs=1e-13)
    assert len(set(marginals)) == 6


def test_stage3_bitstring_display_order_maps_c5_to_q5_and_c0_to_q0():
    counts = {"000001": 7, "100000": 3}
    # Qiskit displays c5...c0, with c_i measured from q_i.
    decoded = {}
    for displayed, count in counts.items():
        decoded.update({q: int(displayed[5 - q]) for q in range(6)})
        if displayed == "000001":
            assert decoded == {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        else:
            assert decoded == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1}
    assert successes_from_counts(counts).tolist() == [0, 10, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("probabilities", [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    [1e-14, 1 - 1e-14, 0.25, 0.5, 0.75, 0.125],
])
def test_stage3_probability_endpoints_and_near_endpoints(probabilities):
    exact = poisson_binomial_distribution(probabilities)
    circuit_distribution = stage3_statevector_distribution(probabilities)
    quantum_by_weight = np.zeros(7)
    for mask, probability in enumerate(circuit_distribution):
        quantum_by_weight[mask.bit_count()] += probability
    assert exact.sum() == pytest.approx(1.0, abs=2e-14)
    assert circuit_distribution.sum() == pytest.approx(1.0, abs=2e-14)
    assert quantum_by_weight == pytest.approx(exact, abs=2e-13)


@pytest.mark.parametrize("values", [[0.1] * 5, [-0.1] + [0.1] * 5, [1.1] + [0.1] * 5])
def test_stage3_rejects_invalid_probability_vectors(values):
    with pytest.raises(ValueError):
        build_stage3_circuit(values)
