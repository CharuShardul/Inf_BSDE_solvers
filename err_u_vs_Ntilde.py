import numpy as np
import matplotlib.pyplot as plt
import vect_grid_scheme as vg

r = 1
R = 5

Ntilde = [4, 5, 6, 7, 8, 10, 12, 14, 17]
log_ntilde = np.round(np.log(Ntilde), 2)
NbP = 10

skiplast = 2 	# number of last points to skip in linear fit

# Set to True to recalculate errors, False to load from files
re_calculate = True #False

arr_err_u = []
arr_err_ub = []

for k in range(5):		# different runs to get more data for box plots
	if re_calculate:
		
		data = [vg.GridScheme(d=1, r=r, R=R, M=int((n**4/R**4)*300), Ntilde=n, NbP=NbP, K_z=0.1, plot_iter=False) for n in Ntilde]
		print("a, b, theta, theta_bar:", data[0].a, data[0].b, data[0].theta, data[0].theta_bar)
		for i in range(len(Ntilde)):
			data[i].PicIter()
			np.save('Numerical_experiments/Grid_scheme/log_log_plot/u_err_r{}_ntilde{}_k{}.npy'.format(data[i].r, data[i].Ntilde - data[i].r, k),
					np.array(data[i].picerr_u))
			np.save('Numerical_experiments/Grid_scheme/log_log_plot/ub_err_r{}_ntilde{}_k{}.npy'.format(data[i].r, data[i].Ntilde - data[i].r, k), 
					np.array(data[i].picerr_ub))

	if re_calculate:
		err_u = [np.array(data[i].picerr_u) for i in range(len(Ntilde))]
	else:
		err_u = [np.load('Numerical_experiments/Grid_scheme/log_log_plot/u_err_r{}_ntilde{}_k{}.npy'.format(r, n, k)) for n in Ntilde]
	log_err_u = [np.max(np.log(err_u[i][-1][:, 0])) for i in range(len(Ntilde))]
	arr_err_u.append(log_err_u)

	if re_calculate:
		err_ub = [np.array(data[i].picerr_ub) for i in range(len(Ntilde))]
	else:
		err_ub = [np.load('Numerical_experiments/Grid_scheme/log_log_plot/ub_err_r{}_ntilde{}_k{}.npy'.format(r, n, k)) for n in Ntilde]
	log_err_ub = [np.max(np.log(err_ub[i][-1][:, 0])) for i in range(len(Ntilde))]
	arr_err_ub.append(log_err_ub)

arr_err_u = np.array(arr_err_u)
print("arr_err_u shape:", arr_err_u.shape)
arr_err_ub = np.array(arr_err_ub)
print("arr_err_ub shape:", arr_err_ub.shape)

fig = plt.figure(figsize=(10, 5), tight_layout=True)

ax1 = fig.add_subplot(121)
ax1.set_title(r"Max log-errors in $u(x)$ v/s $\log (\tilde{N})$")
ax1.set_xlabel(r"$\log (\tilde{N})$")
ax1.set_ylabel(fr"$\sup_x\ \log (\Delta u_{r}^{{10}})$")
bp = ax1.boxplot(arr_err_u, positions=log_ntilde, widths=0.11, boxprops={'facecolor':'lightblue'}, patch_artist=True)
ax1.set_xticklabels(ax1.get_xticks(), rotation=45)

#medians = np.array(bp['medians'][0].get_ydata())
medians = np.array([line.get_ydata()[0] for line in bp['medians']])
mean_vals = np.mean(arr_err_u, axis=0)
print("shapes:", log_ntilde[:-skiplast].shape, medians[:-skiplast].shape)
#slope, intercept = np.polyfit(log_ntilde[:-skiplast], medians[:-skiplast], 1)
slope, intercept = np.polyfit(log_ntilde[:-skiplast], mean_vals[:-skiplast], 1)
fitline = slope * log_ntilde + intercept
ax1.plot(log_ntilde[:-skiplast], fitline[:-skiplast], linestyle=':', marker='o', color='grey', label=f'slope={slope:.2f}')
ax1.legend(frameon=False)

ax2 = fig.add_subplot(122)
ax2.set_title(r"Max log-errors in $\bar{u}(x)$ v/s $\log (\tilde{N})$")
ax2.set_xlabel(r"$\log (\tilde{N})$")
ax2.set_ylabel(fr"$\sup_x\ \log (\Delta \bar u_{r}^{{10}})$")
bp = ax2.boxplot(arr_err_ub, positions=log_ntilde, widths=0.11, boxprops={'facecolor':'peachpuff'}, patch_artist=True)
ax2.set_xticklabels(ax2.get_xticks(), rotation=45)

#medians = np.array(bp['medians'][0].get_ydata())
medians = np.array([line.get_ydata()[0] for line in bp['medians']])
mean_vals = np.mean(arr_err_ub, axis=0)
#slope, intercept = np.polyfit(log_ntilde[:-skiplast], medians[:-skiplast], 1)
slope, intercept = np.polyfit(log_ntilde[:-skiplast], mean_vals[:-skiplast], 1)
fitline = slope * log_ntilde + intercept
ax2.plot(log_ntilde[:-skiplast], fitline[:-skiplast], linestyle=':', marker='o', color='grey', label=f'slope={slope:.2f}')
ax2.legend(frameon=False)
plt.savefig('plots_article/err_u_vs_ntilde_boxplot_r{}.pdf'.format(r), dpi=300)
#plt.show()





