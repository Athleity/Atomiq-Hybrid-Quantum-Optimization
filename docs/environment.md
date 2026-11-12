# Reproducible local setup

Validated runtime: CPython 3.12.14, Windows 11 x86-64. The task host's bundled Python path is local to this machine, so recreate the venv from a standard CPython 3.12.14 install; do not copy `.venv` between machines.

## Setup in PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.in` lists reviewed direct dependencies. `requirements-lock.txt` is the exact `pip freeze` from the installed environment and pins all resolved packages. Regenerate it only after an intentional dependency change:

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding ascii requirements-lock.txt
```

The lock records exact versions but not artifact hashes. Exact Python version and platform matter for this tested Windows wheel set.

## Included stack

- Classical tabular work: NumPy 2.5.4, pandas 3.1.1, SciPy 1.18.1, scikit-learn 1.9.1, CatBoost 1.2.10, and LightGBM 4.7.0.
- Data/config/testing and plots: openpyxl 3.1.5, PyYAML 6.0.3, pytest 9.2.0, and Matplotlib 3.11.2.
- Quantum stages: Qiskit 2.6.2, Qiskit Aer 0.18.2, Qiskit Machine Learning 0.9.1, and Qiskit Optimization 0.7.0.

The whole dependency set resolved without conflicts and `pip check` reported no broken requirements. Imports for all listed modeling/quantum packages, a scikit-learn toy fit, and a seeded two-qubit Aer Bell circuit were exercised successfully. These are environment smoke checks, not challenge experiments or pytest results.

## Limitations and API notes

This host has 8 logical CPUs and 7.8 GiB RAM. No `nvidia-smi` utility was available, so no NVIDIA GPU is confirmed. No IBM hardware credentials/backend were configured. Treat local Aer as the available quantum execution target until hardware access is independently established.

Qiskit Aer 0.18.2 is in reduced-maintenance mode according to its maintainers. The separate `qiskit-algorithms` package is excluded because its maintainers state it is no longer officially supported by IBM. QAOA/VQE work can use Qiskit circuits/primitives plus SciPy, or add that package only after reviewing a concrete Claude dependency and exercising its APIs.

This environment is for code development and local simulation. It does not establish quantum advantage or real-hardware execution.

## Verified upstream references (checked 2026-11-11)

- [scikit-learn on PyPI](https://pypi.org/project/scikit-learn/) lists Python >=3.11 and CPython 3.12 Windows wheels.
- [Qiskit on PyPI](https://pypi.org/project/qiskit/) lists Python >=3.10; [Aer 0.18.2 on PyPI](https://pypi.org/project/qiskit-aer/0.18.2/) includes CPython 3.12 Windows wheels and its reduced-maintenance notice.
- [Qiskit Machine Learning 0.9.1 on PyPI](https://pypi.org/project/qiskit-machine-learning/) lists Python >=3.10; [Qiskit Optimization 0.7.0 installation docs](https://qiskit-community.github.io/qiskit-optimization/getting_started.html) state it installs alongside a compatible Qiskit version.
- [Qiskit Algorithms maintainers' repository](https://github.com/qiskit-community/qiskit-algorithms) says the project is no longer officially supported by IBM.
