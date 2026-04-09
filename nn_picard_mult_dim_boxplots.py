import os
import numpy as np
import matplotlib.pyplot as plt
from NN_Picard_mult_alt import NNPicardSolver
import logging
#from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.integrate import quad

# Suppress TensorFlow logging at module level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)  # Only show ERROR level


# Setup logging at module level (called once, not per instance)
if not logging.root.handlers:
    logging.basicConfig(filename='NN_Picard_mult_alt.log', level=logging.INFO)


def an_u(x, d):
        """Analytical solution for u."""
        return (1 / d) * np.sum(np.arctan(x), axis=-1, keepdims=True)
    
def an_ub(x, d):
    """Analytical solution for ub (gradient of u)."""
    return (1 / d) * (1 / (1 + x**2))

def main():
    d_values = [1, 2, 3, 4, 5, 6, 7, 10, 15, 50]
    n_runs = 5
    re_calculate = False  # Set to False to load existing error data instead of recalculating

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
                    M=40000,
                    M_err=1000,
                    Ntilde=21,
                    batch_size=1000,
                    epochs=120,
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

    
    # L^2 norm calculation for u and \bar{u} for relative errors
    
    l2_an_u = []
    l2_an_ub = []
    
    #sample_x1 = np.random.normal(loc=0.0, scale=2.0, size=(1000, 1))
    u_val = np.sqrt(quad(lambda x: norm.pdf(x, loc=0.0, scale=2.0) * (np.arctan(x))**2, -np.inf, np.inf)[0])
    ub_val = np.sqrt(quad(lambda x: norm.pdf(x, loc=0.0, scale=2.0) * (1/(1+x**2)), -np.inf, np.inf)[0])
    
    l2_an_u = [(1/np.sqrt(d)) * u_val for d in d_values]
    l2_an_ub = [(1/np.sqrt(d)) * ub_val for d in d_values]

    '''sample_x = [np.random.normal(loc=0.0, scale=2.0, size=(1000, d_values[i])) for i in range(len(d_values))]
    for i, d in enumerate(d_values):
        an_u_vals = an_u(sample_x[i], d)
        an_ub_vals = an_ub(sample_x[i], d)
        l2_an_u.append(np.sqrt(np.mean(np.square(an_u_vals))))
        l2_an_ub.append(np.sqrt(np.mean(np.sum(np.square(an_ub_vals), axis=-1)))) 
    '''  

   
    fig = plt.figure(figsize=(10, 5), dpi=100, tight_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Boxplot for u
    data_u = np.array([results["l2_u"][d] for d in d_values])
    data_u = data_u / np.array(l2_an_u)[:, None]            # Relative error for u
    #data_u = np.sqrt(np.array(d_values))[:, None] * data_u

    # Curve fitting for u
    #weights = 1 / np.array(d_values)
    #params, _ = curve_fit(lambda d, c, k: c + k*(np.sqrt(d)), d_values, np.mean(data_u, axis=1))
                           #sigma=weights, absolute_sigma=True)
    #c_fit, k_fit = params

    ax1.boxplot(data_u.tolist(), labels=[str(d) for d in d_values], showmeans=True, patch_artist=True, 
                boxprops={'facecolor':'lightblue'})
    ax1.set_title("Relative errors of $u$, $\Delta u^n_d$")
    ax1.set_xlabel("dimension d")
    ax1.set_ylabel("$\Delta u^n_d$")
    #ax1.plot(np.arange(len(d_values)), c_fit + k_fit*np.sqrt(d_values), linestyle='--', color='red', label=r"$\sqrt{d}$ scaling")  # Reference line for sqrt(d) scaling
    ax1.set_xticklabels(d_values)
    ax1.legend(frameon=False)
    #ax1.grid(True, linestyle='--', alpha=0.5)

    # Boxplot for ub
    data_ub = np.array([results["l2_ub"][d] for d in d_values])
    data_ub = data_ub / np.array(l2_an_ub)[:, None]         # Relative error for ub
    #data_ub = np.sqrt(np.array(d_values))[:, None] * data_ub
    
    ax2.boxplot(data_ub.tolist(), labels=[str(d) for d in d_values], showmeans=True, patch_artist=True, 
                boxprops={'facecolor':'peachpuff'})
    #ax2.plot(d_values, np.sqrt(d_values), linestyle='--', color='red', label=r"$\sqrt{d}$ scaling")  # Reference line for sqrt(d) scaling
    ax2.set_title(r"Relative errors of $\bar{u}$, $\Delta \bar{u}^n_d$")
    ax2.set_xlabel("dimension d")
    ax2.set_ylabel(r"$\Delta \bar{u}^n_d$")
    ax2.set_xticklabels(d_values)
    ax2.legend(frameon=False)
    #ax2.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "boxplot_l2_err_d.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\nAll experiments completed. Outputs saved to:", output_dir)


if __name__ == '__main__':
    main()
