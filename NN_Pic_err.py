import numpy as np
import matplotlib.pyplot as plt

dims = np.array([1,2,3,4,5,6,7,10,15,50])
err_u = []
err_ub = []
for i in range(7):
    err_u.append(np.load('Numerical_experiments/error_plots/u_err_{}d.npy'.format(i+1)))
    err_ub.append((i+1)*np.load('Numerical_experiments/error_plots/ub_err_{}d.npy'.format(i+1)))
err_u.append(np.load('Numerical_experiments/error_plots/u_err_10d.npy'))
err_ub.append((10)*np.load('Numerical_experiments/error_plots/ub_err_10d.npy'))

err_u.append(np.load('Numerical_experiments/error_plots/u_err_15d.npy'))
err_ub.append((15)*np.load('Numerical_experiments/error_plots/ub_err_15d.npy'))

err_u.append(np.load('Numerical_experiments/error_plots/u_err_50d.npy'))
err_ub.append((50)*np.load('Numerical_experiments/error_plots/ub_err_50d.npy'))

#print("err_u shapes:", [e.shape for e in err_u])
#print("err_ub shapes:", [e.shape for e in err_ub])

fig = plt.figure(figsize=(11, 5), dpi=75)
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("$u(x)$")
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title(r"$\bar{u}(x)$")

ax1.set_xlabel(r"$d$")
ax1.set_ylabel(r"$\Delta u^n_d,\quad$  $n=5$")

ax2.set_xlabel(r"$d$")
ax2.set_ylabel(r"$d\times \Delta \bar{u}^n_d,\quad$ $n=5$")

ax1.boxplot(err_u)#, positions=([1,2,3,4,5,6,7,10,15]))
#ax2.boxplot(((1/np.sqrt(dims))[:, None] * np.array(err_ub)).tolist())#, positions=([1,2,3,4,5,6,7,10,15]))
ax2.boxplot(err_ub)
ax1.set_xticklabels([1,2,3,4,5,6,7,10,15,50])
ax2.set_xticklabels([1,2,3,4,5,6,7,10,15,50])
plt.savefig("plots_article/NNPic_d_boxplot.pdf", dpi=300, bbox_inches='tight')

plt.show()
