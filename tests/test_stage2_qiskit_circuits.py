import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from atomiq.qiskit_noise import (
    IllustrativeBackendProfile,
    assignment_matrix,
    calibration_assignment_matrices,
    mitigate_readout,
    resource_report,
    run_aer_counts,
)
from atomiq.stage2 import Candidate, Stage2Problem, evaluate_selection
from atomiq.stage2_qiskit_circuits import (
    bind_angles,
    build_parameterized_qaoa,
    cvar_lower_tail,
    exact_distribution,
    optimize_qaoa,
)


def _problem():
    rows = [(0.8, 30, 1, 1), (0.7, 25, 1, 1), (0.6, 20, 2, 2), (0.1, 10, 4, 4)]
    candidates = tuple(Candidate(f"S{i}", *row) for i, row in enumerate(rows))
    return Stage2Problem(candidates, selected_count=2, budget_limit=3, capacity_limit=3)


def test_standard_qaoa_is_parameterized_and_uses_the_exact_diagonal_cost_phase():
    ansatz = build_parameterized_qaoa(_problem(), depth=1, kind="standard")
    assert ansatz.circuit.num_parameters == 2
    gamma = 0.031
    state = Statevector.from_instruction(bind_angles(ansatz, [gamma, 0.0])).data
    expected = np.exp(-1j * gamma * ansatz.costs) / 4
    assert state == pytest.approx(expected, abs=1e-10)


def test_xy_qaoa_preserves_exact_cardinality_and_topology_changes_circuit():
    problem = _problem()
    path = build_parameterized_qaoa(problem, depth=1, kind="xy", topology="path")
    cycle = build_parameterized_qaoa(problem, depth=1, kind="xy", topology="cycle")
    assert path.circuit.num_parameters == cycle.circuit.num_parameters == 2
    distribution = exact_distribution(cycle, [0.2, 0.4])
    assert all(mask.bit_count() == 2 for mask, probability in enumerate(distribution) if probability > 1e-10)
    assert cycle.circuit.count_ops()["rxx"] > path.circuit.count_ops()["rxx"]


def test_xy_additive_oracle_matches_exact_net_value_phase_on_feasible_k_sector():
    problem = _problem()
    ansatz = build_parameterized_qaoa(problem, kind="xy")
    assert ansatz.circuit.metadata["fixed_sector_feasible_additive_cost"] is False
    state = Statevector.from_instruction(bind_angles(ansatz, [0.017, 0.0])).data
    for mask, amplitude in enumerate(state):
        if mask.bit_count() == problem.selected_count and abs(amplitude) > 1e-10:
            ids = [c.customer_id for i, c in enumerate(problem.candidates) if (mask >> i) & 1]
            result = evaluate_selection(problem, ids)
            # This fixture has binding resource limits, hence its full feasibility oracle is used.
            if result.feasible:
                assert amplitude == pytest.approx(np.exp(1j * 0.017 * result.objective) / np.sqrt(6), abs=1e-9)


def test_ten_qubit_exact_six_xy_circuit_has_actual_dicke_sector_support():
    candidates = tuple(Candidate(f"T{i}", 0.1 + i * 0.01, 20 + i, 1, 1) for i in range(10))
    problem = Stage2Problem(candidates, selected_count=6, budget_limit=100, capacity_limit=100)
    ansatz = build_parameterized_qaoa(problem, kind="xy", depth=1)
    assert ansatz.circuit.num_qubits == 10
    assert ansatz.circuit.metadata["fixed_sector_feasible_additive_cost"] is True
    assert "mcphase" not in ansatz.circuit.count_ops()
    distribution = exact_distribution(ansatz, [0.2, 0.3])
    assert distribution.sum() == pytest.approx(1.0)
    assert all(mask.bit_count() == 6 for mask, probability in enumerate(distribution) if probability > 1e-10)


def test_cvar_splits_boundary_probability_mass_correctly():
    assert cvar_lower_tail(np.array([0.0, 10.0]), np.array([0.1, 0.9]), 0.25) == pytest.approx(6.0)


@pytest.mark.parametrize("optimizer,objective", [("COBYLA", "expectation"), ("SPSA", "cvar")])
def test_cobyla_and_spsa_fit_parameterized_qiskit_ansatz(optimizer, objective):
    ansatz = build_parameterized_qaoa(_problem(), kind="xy")
    result = optimize_qaoa(ansatz, optimizer=optimizer, objective_kind=objective, seed=11, max_evaluations=6)
    assert len(result.parameters) == 2
    assert result.evaluations <= 6
    assert result.evaluations >= 4
    assert np.isfinite(result.objective_value)


def test_illustrative_noise_profile_has_valid_asymmetric_readout_and_thermal_channels():
    profile = IllustrativeBackendProfile()
    assert profile.readout_assignment(0)[1, 0] != profile.readout_assignment(0)[0, 1]
    model = profile.noise_model()
    assert set(model.basis_gates) >= {"rz", "sx", "x", "id", "cx"}
    assert model.to_dict()["errors"]


def test_readout_mitigation_recovers_a_known_noisy_distribution():
    profile = IllustrativeBackendProfile()
    matrices = [profile.readout_assignment(q) for q in range(2)]
    true = np.zeros(4)
    true[1] = 1.0  # Qiskit integer index 1 means q0=1 and q1=0.
    observed = assignment_matrix(matrices) @ true
    counts = {format(index, "02b"): int(round(probability * 100_000)) for index, probability in enumerate(observed)}
    result = mitigate_readout(counts, matrices)
    assert result["projected_distribution"] == pytest.approx(true, abs=5e-5)


def test_aer_calibration_and_circuit_execution_include_connectivity_and_duration_model():
    profile = IllustrativeBackendProfile()
    matrices, _ = calibration_assignment_matrices(2, profile=profile, shots=4096, seed=5)
    assert matrices[0] == pytest.approx(profile.readout_assignment(0), abs=0.035)
    problem = _problem()
    ansatz = build_parameterized_qaoa(problem, kind="xy", topology="complete")
    circuit = bind_angles(ansatz, [0.2, 0.3])
    resources = resource_report(circuit, profile, seed=3)
    assert resources["logical_qubits"] == 4
    assert resources["transpiled_depth"] > 0
    assert resources["routing_swap_count_before_basis_translation"] > 0
    counts, _, compiled = run_aer_counts(circuit, profile=profile, shots=64, seed=7, noise=True)
    assert sum(counts.values()) == 64
    assert compiled.count_ops().get("id", 0) == 4
