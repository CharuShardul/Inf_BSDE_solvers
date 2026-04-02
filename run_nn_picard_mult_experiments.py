import os
import numpy as np
import matplotlib.pyplot as plt
from NN_Picard_mult_alt import NNPicardSolver
import logging

# Suppress TensorFlow logging at module level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)  # Only show ERROR level


# Setup logging at module level (called once, not per instance)
if not logging.root.handlers:
    logging.basicConfig(filename='NN_Picard_mult_alt.log', level=logging.INFO)

def main():
    d_values = [1, 2, 3, 4, 5, 6, 7, 10, 15, 50]
    n_runs = 5
    re_calculate = True  # Set to False to load existing error data instead of recalculating

    output_dir = "Numerical_experiments/error_plots/NN_Picard_mult_experiments"
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "d_values": d_values,
        "l2_u": {},
        "l2_ub": {},
    }

    if re_calculate:
        for d in d_values:
            print("\n===== d={} =====".format(d))
            results["l2_u"][d] = []
            results["l2_ub"][d] = []

            for run in range(1, n_runs + 1):
                print("-> Run {}/{} for d={}".format(run, n_runs, d))

                solver = NNPicardSolver(
                    d=d,
                    K_z=0.1,
                    num_pic=5,
                    M=20000,
                    M_err=1000,
                    Ntilde=21,
                    batch_size=1000,
                    epochs=100,
                    activation='relu',
                    initial_learning_rate=5e-4,
                    lr_decay=0.97,
                    decay_steps=1000,
                    a=2.0,
                    b=2.0,
                    theta=1.5,
                    theta_bar=1.5,
                    sig_X=2.0,
                )

                solver.main()

                if solver.u_err is None or solver.ub_err is None:
                    raise RuntimeError(f"Solver did not compute u_err/ub_err for d={d} run={run}")

                l2_u_norm = float(np.sqrt(np.mean(np.square(solver.u_err))))
                l2_ub_norm = float(np.sqrt(np.mean(np.square(solver.ub_err))))

                results["l2_u"][d].append(l2_u_norm)
                results["l2_ub"][d].append(l2_ub_norm)

                np.save(os.path.join(output_dir, f"l2_u_d{d}_run{run}.npy"), l2_u_norm)
                np.save(os.path.join(output_dir, f"l2_ub_d{d}_run{run}.npy"), l2_ub_norm)
                np.save(os.path.join(output_dir, f"u_err_d{d}_run{run}.npy"), solver.u_err)
                np.save(os.path.join(output_dir, f"ub_err_d{d}_run{run}.npy"), solver.ub_err)

                print(f"   L2_u={l2_u_norm:.6e}, L2_ub={l2_ub_norm:.6e}")

        np.save(os.path.join(output_dir, "results_l2_u.npy"), results["l2_u"])
        np.save(os.path.join(output_dir, "results_l2_ub.npy"), results["l2_ub"])

    else:
        results["l2_u"] = np.load(os.path.join(output_dir, "results_l2_u.npy"), allow_pickle=True).item()
        results["l2_ub"] = np.load(os.path.join(output_dir, "results_l2_ub.npy"), allow_pickle=True).item() 

    # Boxplot for u
    fig, ax = plt.subplots(figsize=(12, 7))
    data_u = [results["l2_u"][d] for d in d_values]
    ax.boxplot(data_u, labels=[str(d) for d in d_values], showmeans=True)
    ax.set_title("L^2 Norm of errors (M_err=1000) across runs")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("L^2 norm of u error")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "boxplot_l2_u.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Boxplot for ub
    fig, ax = plt.subplots(figsize=(12, 7))
    data_ub = [results["l2_ub"][d] for d in d_values]
    ax.boxplot(data_ub, labels=[str(d) for d in d_values], showmeans=True)
    ax.set_title("L^2 Norm of ub Error (M_err=1000) across runs")
    ax.set_xlabel("dimension d")
    ax.set_ylabel("L^2 norm of ub error")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "boxplot_l2_ub.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\nAll experiments completed. Outputs saved to:", output_dir)


if __name__ == '__main__':
    main()
