"""Independent cross-validation of our quantum kernels against PennyLane gate circuits.

Confirms the closed-form kernels we use (qkla.feature_maps) equal what a recognised quantum framework
produces from explicit gate-level state simulation. Small d (statevector). No training, no GPU, free.

Run: PYTHONPATH=. python scripts/pennylane_crossval.py
"""
import json, numpy as np, torch
import pennylane as qml
from qkla.feature_maps import quantum_kernel_matrix, iqp_kernel_matrix

np.random.seed(0); torch.manual_seed(0)
d, n, bw = 4, 12, 0.7
X = np.random.randn(n, d)
dev = qml.device("default.qubit", wires=d)

# ---- angle/product kernel: H then RZ(bw*x_j) on each qubit ----
@qml.qnode(dev)
def angle_state(x):
    for j in range(d):
        qml.Hadamard(wires=j)
        qml.RZ(bw * x[j], wires=j)
    return qml.state()

S = np.stack([angle_state(X[i]) for i in range(n)])
K_pl_angle = np.abs(S.conj() @ S.T) ** 2
K_ours_angle = quantum_kernel_matrix(torch.tensor(X), bandwidth=bw, layers=1).numpy()
diff_angle = float(np.abs(K_pl_angle - K_ours_angle).max())

# ---- IQP kernel: H, RZ (linear), IsingZZ (cross term); global phase cancels in |overlap|^2 ----
def iqp_state_factory(c):
    @qml.qnode(dev)
    def iqp_state(x):
        for j in range(d):
            qml.Hadamard(wires=j)
        for j in range(d):
            qml.RZ(-2.0 * bw * x[j], wires=j)
        for j in range(d):
            for k in range(j + 1, d):
                qml.IsingZZ(-2.0 * c * bw**2 * x[j] * x[k], wires=[j, k])
        return qml.state()
    return iqp_state

results = {"d": d, "n": n, "bandwidth": bw, "pennylane_version": qml.__version__,
           "angle_kernel_max_abs_diff": diff_angle, "iqp": {}}
print(f"angle/product kernel  max|PennyLane - ours| = {diff_angle:.2e}")
for c in [0.0, 0.5, 1.0]:
    f = iqp_state_factory(c)
    Si = np.stack([f(X[i]) for i in range(n)])
    K_pl = np.abs(Si.conj() @ Si.T) ** 2
    K_ours = iqp_kernel_matrix(torch.tensor(X), bandwidth=bw, coupling=c).numpy()
    diff = float(np.abs(K_pl - K_ours).max())
    results["iqp"][str(c)] = diff
    print(f"IQP kernel (c={c})     max|PennyLane - ours| = {diff:.2e}")

results["verdict"] = "MATCH" if (diff_angle < 1e-5 and all(v < 1e-5 for v in results["iqp"].values())) else "MISMATCH"
print("VERDICT:", results["verdict"])
import os; os.makedirs("results", exist_ok=True)
json.dump(results, open("results/pennylane_crossval.json", "w"), indent=2)
print("saved -> results/pennylane_crossval.json")
