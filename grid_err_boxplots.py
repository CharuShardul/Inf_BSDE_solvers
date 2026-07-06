# Generate boxplots of errors for different values of r in the vect_grid_scheme

import numpy as np
import matplotlib.pyplot as plt
import vect_grid_scheme as vg

r = [0, 2]
R = 3
Ntilde = 4
NbP = 10

# Set to True to recalculate errors, False to load from files
re_calculate = False

if re_calculate:
    data = [vg.GridScheme(d=1, r=i, R=R, M=50000, Ntilde=Ntilde, NbP=NbP, K_z=0.5, plot_iter=False) for i in r]
    for i in range(len(r)):
        data[i].PicIter()
        np.save('Numerical_experiments/Grid_scheme/box_plots/u_err_{}d_r{}.npy'.format(data[i].d, data[i].r),
                np.array(data[i].picerr_u))
        np.save('Numerical_experiments/Grid_scheme/box_plots/ub_err_{}d_r{}.npy'.format(data[i].d, data[i].r), 
                np.array(data[i].picerr_ub))
    
err_u = [np.load('Numerical_experiments/Grid_scheme/box_plots/u_err_1d_r{}.npy'.format(r[i])) for i in range(len(r))]
err_ub = [np.load('Numerical_experiments/Grid_scheme/box_plots/ub_err_1d_r{}.npy'.format(r[i])) for i in range(len(r))]

print("shapes:", np.transpose(err_u[0][2:][:, :, 0]).shape, np.transpose(err_ub[0][2:][:, :, 0]).shape)

fig = plt.figure(figsize=(10, 5), dpi=75)

offset = 0.09
width = 0.18
delta = R/Ntilde
x_axis = delta * (np.linspace(-Ntilde, Ntilde, 2*Ntilde+1))
print("len:", len(x_axis))

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title(r"Errors in $u(x)$ for different $p$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$\Delta u^n_p, \quad n=3, \dots, 10$")
bp1 = ax1.boxplot((err_u[0][3:][:, :, 0]), positions=x_axis - offset, widths=width,
            boxprops={'facecolor':'lightblue'}, patch_artist=True, showfliers=False)
bp2 = ax1.boxplot((err_u[1][3:][:, :, 0]), positions=x_axis + offset, widths=width, 
            boxprops={'facecolor':'red'}, patch_artist=True, showfliers=False)
ax1.set_xticks(x_axis)
ax1.set_xticklabels([f"{x:.1f}" for x in x_axis])
ax1.legend([bp1["boxes"][0], bp2["boxes"][1]], [f"$p = {r[0]}$", f"$p = {r[1]}$"], loc="best")

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title(r"Errors in $\bar{u}(x)$ for different $p$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$\Delta \bar{u}^n_p, \quad n=3, \dots, 10$")
bp1 = ax2.boxplot((err_ub[0][3:][:, :, 0]), positions=x_axis - offset, widths=width,
            boxprops={'facecolor':'lightblue'}, patch_artist=True, showfliers=False)
bp2 = ax2.boxplot((err_ub[1][3:][:, :, 0]), positions=x_axis + offset, widths=width,
            boxprops={'facecolor':'red'}, patch_artist=True, showfliers=False)
ax2.set_xticks(x_axis)
ax2.set_xticklabels([f"{x:.1f}" for x in x_axis])
ax2.legend([bp1["boxes"][0], bp2["boxes"][1]], [f"$p = {r[0]}$", f"$p = {r[1]}$"], loc="best")

plt.tight_layout()
plt.savefig('plots_article/boxplot_grid_r.pdf', dpi=300)
#plt.show()



