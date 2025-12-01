from vect_grid_scheme import GridScheme
import numpy as np
import matplotlib.pyplot as plt
from time import time

time_init = time()
#A.PicIter()
#print(np.max(A.picerr_u[0]))
num_exp = 7
# Default values for class GridScheme: d=1, Ntilde=10, M=2000, K_z=1.0, r=1, R=4, NbP = 6, plot_iter=True.
r_range = np.arange(0, num_exp)
exp_r = [GridScheme(r=r, M=40000, NbP=12, K_z=0.5, plot_iter=False) for r in r_range]

for i, r in enumerate(r_range):
    exp_r[i].PicIter()
    print("r = ", r, ", max err_u = ", np.max(exp_r[i].picerr_u[-1]), ", max err_ub = ", np.max(exp_r[i].picerr_ub[-1])) 

fig = plt.figure(figsize=(12, 5), dpi = 75, tight_layout=True)
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("Max log error in $u(x)$ vs Picard iterations")
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title(r"Max log error in $\bar{u}(x)$ vs Picard iterations")
for i, r in enumerate(r_range):
    ax1.plot(np.arange(1, exp_r[i].NbP + 1), np.log(np.max(exp_r[i].picerr_u, axis=1)), label="r = {}".format(r))
    ax2.plot(np.arange(1, exp_r[i].NbP + 1), np.log(np.max(exp_r[i].picerr_ub, axis=1)), label="r = {}".format(r))

ax1.set_xlabel("Picard iteration")
ax2.set_xlabel("Picard iteration")
ax1.set_ylabel("Max log error in $u(x)$")
ax2.set_ylabel(r"Max log error in $\bar{u}(x)$")
ax1.legend(loc="upper right")
ax2.legend(loc="upper right")
plt.show()
