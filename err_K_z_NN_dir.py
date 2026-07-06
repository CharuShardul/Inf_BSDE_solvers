from NN_direct_scheme import NNDirecSolver
import numpy as np
import matplotlib.pyplot as plt

"""Run experiments for different K_z values and make boxplots for the errors for u and ub."""

K_z_values = np.round(np.arange(0.0, 6.5, 0.5), 1)
re_calculate = False  # Set to False to load existing error data instead of recalculating

err_u = []
err_ub = []

if re_calculate:
    for K_z in K_z_values:
        print(f"\nRunning experiments for K_z = {K_z}")
        solver = NNDirecSolver(d=1, K_z=K_z)
        solver.main()
        err_u.append(solver.u_err)
        err_ub.append(solver.ub_err)
        
else:
    for K_z in K_z_values:
        err_u.append(np.load('Numerical_experiments/error_plots/Direct_scheme/u_err_Kz_{}_1d.npy'.format(K_z)))
        err_ub.append(np.load('Numerical_experiments/error_plots/Direct_scheme/ub_err_Kz_{}_1d.npy'.format(K_z)))

fig = plt.figure(figsize=(10, 5), tight_layout=True)

percentiles = 99.9        # Percentiles for y-limits in boxplot
lower = min(np.percentile(np.log(err_u), 100 - percentiles), np.percentile(np.log(err_ub), 100 - percentiles))
upper = max(np.percentile(np.log(err_u), percentiles), np.percentile(np.log(err_ub), percentiles))

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title(r"log-errors in $u(x)$ v/s $K_z$")
ax1.set_xlabel(r"$K_z$")
ax1.set_ylabel(r"$\log (\Delta u_{d})$,\ d=1")
ax1.boxplot(np.log(err_u).tolist(), labels=K_z_values,  widths=0.45, boxprops={'facecolor':'lightblue'}, 
            patch_artist=True, showfliers=True)

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title(r"log-errors in $\bar{u}(x)$ v/s $K_z$")
ax2.set_xlabel(r"$K_z$")
ax2.set_ylabel(r"$\log (\Delta \bar{u}_{d})$,\ d=1")
ax2.boxplot(np.log(err_ub).tolist(), labels=K_z_values, widths=0.45, boxprops={'facecolor':'peachpuff'}, 
            patch_artist=True, showfliers=True)

#ax1.set_ylim(bottom=lower)
#ax2.set_ylim(bottom=lower)
#plt.show()
plt.savefig('plots_article/err_K_z_NNdir.pdf', dpi=300)

