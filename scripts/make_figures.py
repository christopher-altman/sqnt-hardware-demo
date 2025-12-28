import numpy as np
import matplotlib.pyplot as plt
from sqnt_hardware_demo.train_demo import sweep_alphas

def main():
    alphas = np.linspace(0.0, 1.0, 11)
    accs = sweep_alphas(alphas, n=12, seed=0, topo0="chain", topo1="complete")

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(alphas, accs, marker="o")
    plt.xlabel("Topology mixture parameter α  (chain → complete)")
    plt.ylabel("Training-set accuracy")
    plt.title("SQNT demo: accuracy vs superposed topology")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "figures/sqnt_mixture_curve.png"
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
