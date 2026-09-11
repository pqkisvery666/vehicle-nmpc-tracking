# Vehicle Path-Tracking MPC

A complete **nonlinear model predictive control (NMPC)** implementation for vehicle
path tracking — from first-principles model derivation to closed-loop verification.
Built on a **kinematic bicycle model** and a **parametric variable-curvature closed
track**, with a path-tracking NMPC implemented using **CasADi + IPOPT**, plus
comparative experiments, automated verification and a solver benchmark.

- Code comments are in Chinese; the full derivations live in the code headers and in the
  [technical notes](vehicle-nmpc-tracking/docs/notes-mpc.md)
- Controller features: **path-frame cost**, **lateral corridor (hard) constraint**,
  **steering-rate constraint**, **slack-variable soft constraint**, and warm starting
- 4 closed-loop scenarios × 2 controller configurations: steady cruise / startup transient /
  offset recovery / model mismatch
- `verify.py` provides **12 automated checks** (correctness + constraint satisfaction); all passing
- `benchmark_solvers.py` compares 4 NLP solvers (IPOPT / fatrop / sqpmethod / qrsqp)
- Runs on Windows / Linux / macOS, no GPU required

## Project structure

```
vehicle-nmpc-tracking/          # repository root（最外层就是这个目录）
├── model.py                    # kinematic bicycle model + RK4 (NumPy fast path / CasADi symbolic path)
├── track.py                    # closed-track generation + nearest-point / lateral-error queries
├── mpc.py                      # NMPC: path-frame cost, corridor / rate / slack constraints, multi-solver
├── simulate.py                 # 4-scenario closed-loop driver (argparse + data cache + plotting)
├── verify.py                   # 12 automated checks (regression tests + doc-reference check)
├── benchmark_solvers.py        # solver benchmark
├── animation.py                # track-following animation GIF (reads cached data)
├── requirements.txt
├── README.md                   # this file
├── docs/notes-mpc.md           # technical notes: derivation / constraint design / verification / limitations
└── results/                    # figures, cached data, run config, benchmark results

## Quick start

```bash
cd vehicle-nmpc-tracking
pip install -r requirements.txt

python simulate.py            # run all 4 scenarios: console stats + figures in results/ + cached data
python verify.py              # 12 automated checks
python animation.py           # generate the track-following GIF
python benchmark_solvers.py   # solver benchmark
```

## Results

![Constrained NMPC tracking animation]vehicle-nmpc-tracking/results/animation.gif)

*The constrained controller completes one full lap (target 6 m/s; dashed lines mark the corridor).*

![Steady-cruise trajectories](results/steady_track.png)

*Steady cruise: trajectories of the constrained controller and the relaxed baseline — both track the
centreline well; the difference is whether the commanded control is physically feasible.*

![Startup transient time series](results/startup_timeseries.png)

*Startup transient: the constrained controller saturates exactly at the 3 m/s² acceleration limit,
while the relaxed baseline commands 10 m/s² — not executable on a real vehicle.*

## Key measured results

| Scenario | Peak lateral error | Peak acceleration | Notes |
|---|---|---|---|
| Steady cruise | 0.014 m | 0.42 m/s² | centimetre-level tracking; the steering-rate constraint is active |
| Startup transient | 0.012 m | 3.00 m/s² (at limit) | relaxed baseline commands 10.0 m/s² — physically infeasible |
| Offset recovery | 2.50 m (initial) | 3.00 m/s² | hard constraint infeasible → slack absorbs 1.0 m, then converges back into the corridor |
| Model mismatch | 0.044 m | 0.42 m/s² | real wheelbase 20% larger than the model; feedback keeps the error at centimetre level |

Real-time performance: IPOPT solves in **8.9 ms** on average (100 ms control period →
**11× real-time margin**). Full metrics, figures and the solver comparison are in
[vehicle-nmpc-tracking/README.md](vehicle-nmpc-tracking/README.md).
