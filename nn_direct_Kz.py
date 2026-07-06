import os
import numpy as np
import matplotlib.pyplot as plt
from NN_direct_scheme import NNDirecSolver


def main():
    Kz_values = np.round(np.arange(0.0, 5.6, 0.4), 1)
    n_runs = 5
    re_calculate = False  # Set to True to rerun experiments and regenerate saved data

    output_dir = "Numerical_experiments/NN_direct_Kz"
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "Kz_values": Kz_values.tolist(),
        "l2_u": {},
        "l2_ub": {},
    }

    if re_calculate:
        for K_z in Kz_values:
            print(f"\n===== K_z={K_z} =====")
            results["l2_u"][float(K_z)] = []
            results["l2_ub"][float(K_z)] = []

            for run in range(1, n_runs + 1):
                print(f"-> Run {run}/{n_runs} for K_z={K_z}")

                solver = NNDirecSolver(
                    d=1,
                    K_z=float(K_z),
                    Mx=512,
                    M=3000,
                    M_err=1000,
                    Ntilde=21,
                    activation='relu',
                    initial_learning_rate=5e-4,
                    lr_decay=0.6,
                    n_decays=10,
                    n_steps=3000,
                    batch_size=512,
                    epochs=500,
                    a=2.0,
                    b=2.0,
                    theta=1.5,
                    theta_bar=1.5,
                    sig_X=2.0,
                    c=2.0,
                    update_frequency=100,
                )

                solver.main()

                if solver.u_err is None or solver.ub_err is None:
                    raise RuntimeError(f"Solver did not compute u_err/ub_err for K_z={K_z} run={run}")

                l2_u_norm = float(np.sqrt(np.mean(np.square(solver.u_err))))
                l2_ub_norm = float(np.sqrt(np.mean(np.square(solver.ub_err))))

                results["l2_u"][float(K_z)].append(l2_u_norm)
                results["l2_ub"][float(K_z)].append(l2_ub_norm)

                np.save(os.path.join(output_dir, f"l2_u_Kz_{K_z}_run_{run}.npy"), l2_u_norm)
                np.save(os.path.join(output_dir, f"l2_ub_Kz_{K_z}_run_{run}.npy"), l2_ub_norm)
                np.save(os.path.join(output_dir, f"u_err_Kz_{K_z}_run_{run}.npy"), solver.u_err)
                np.save(os.path.join(output_dir, f"ub_err_Kz_{K_z}_run_{run}.npy"), solver.ub_err)

                print(f"   L2_u={l2_u_norm:.6e}, L2_ub={l2_ub_norm:.6e}")

        np.save(os.path.join(output_dir, "results_l2_u.npy"), results["l2_u"])
        np.save(os.path.join(output_dir, "results_l2_ub.npy"), results["l2_ub"])

    else:
        results["l2_u"] = np.load(os.path.join(output_dir, "results_l2_u.npy"), allow_pickle=True).item()
        results["l2_ub"] = np.load(os.path.join(output_dir, "results_l2_ub.npy"), allow_pickle=True).item()

    fig = plt.figure(figsize=(10, 5), dpi=100, tight_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    data_u = np.array([results["l2_u"][float(K_z)] for K_z in Kz_values])
    data_ub = np.array([results["l2_ub"][float(K_z)] for K_z in Kz_values])

    data_u = np.log(data_u)
    data_ub = np.log(data_ub)

    ax1.boxplot(data_u.tolist(), labels=[str(K_z) for K_z in Kz_values], showmeans=True, patch_artist=True, 
                boxprops={'facecolor':'lightblue'})
    ax1.set_title(r"log of $L^2_{\mu_0}$ errors for u")
    ax1.set_xlabel("$K_z$")
    ax1.set_ylabel(r"$\Delta u_{K_z}$")
    ax1.set_xticklabels([str(K_z) for K_z in Kz_values], rotation=45)
    #ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.boxplot(data_ub.tolist(), labels=[str(K_z) for K_z in Kz_values], showmeans=True, patch_artist=True, 
                boxprops={'facecolor':'peachpuff'})
    ax2.set_title(r"log of $L^2_{\mu_0}$ errors for $\bar{u}$")
    ax2.set_xlabel("$K_z$")
    ax2.set_ylabel(r"$\Delta \bar{u}_{K_z}$")
    ax2.set_xticklabels([str(K_z) for K_z in Kz_values], rotation=45)
    #ax2.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(os.path.join(output_dir, "boxplot_NNdir_l2_err_Kz.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\nDirect scheme K_z experiments completed. Outputs saved to:", output_dir)


if __name__ == '__main__':
    main()
