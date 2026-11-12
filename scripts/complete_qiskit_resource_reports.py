"""Add parameter-specific transpiler/connectivity reports to saved Stage 2 Aer runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomiq.qiskit_noise import IllustrativeBackendProfile, resource_report
from atomiq.stage2_development_fixtures import four_item_binding_problem
from atomiq.stage2_qiskit_circuits import bind_angles, build_parameterized_qaoa


def main() -> None:
    path = Path("results/stage2_qiskit_noise/qiskit_stage2_noise_results.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problem = four_item_binding_problem()
    profile = IllustrativeBackendProfile()
    seed = int(manifest["configuration"]["evaluation_seed"])
    for run in manifest["runs"]:
        ansatz = build_parameterized_qaoa(
            problem, depth=manifest["configuration"]["depth"],
            kind=run["ansatz"], topology="cycle",
        )
        circuit = bind_angles(ansatz, run["fit"]["parameters"])
        run["evaluation"]["resource_report"] = resource_report(circuit, profile, seed=seed)
    depth_two = manifest["depth_two_interpolated_warm_start"]
    ansatz2 = build_parameterized_qaoa(problem, depth=2, kind="xy", topology="cycle")
    depth_two["evaluation"]["resource_report"] = resource_report(
        bind_angles(ansatz2, depth_two["fit"]["parameters"]), profile, seed=seed
    )
    noise_fit = manifest["noise_optimized_spsa"]
    ansatz_noise = build_parameterized_qaoa(problem, depth=1, kind="xy", topology="cycle")
    noise_fit["evaluation"]["resource_report"] = resource_report(
        bind_angles(ansatz_noise, noise_fit["parameters"]), profile, seed=seed
    )
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Added parameter-specific resource reports to {path}")


if __name__ == "__main__":
    main()
