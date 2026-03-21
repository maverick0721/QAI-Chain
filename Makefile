PY311=.venv311/bin/python

.PHONY: reproduce
reproduce:
	PYTHONPATH=. $(PY311) experiments/run_standard_constrained_transfer.py --envs SafetyPointGoal1-v0 --algos CPO P3O PPOLag --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --total-steps 3000 --steps-per-epoch 1000
	PYTHONPATH=. $(PY311) experiments/run_quantum_uncertainty_comparison.py
	PYTHONPATH=. $(PY311) experiments/run_parameter_efficiency_matched.py
	PYTHONPATH=. $(PY311) experiments/run_adversarial_input_perturbation.py
	PYTHONPATH=. $(PY311) experiments/run_qubit_scaling_trend.py
