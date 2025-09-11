import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from numba import njit
import time


# sigma and its derivative for the SDE and the Malliavin weights
epsilon = 0.9

@njit(fastmath=True)
def sig(x):
    return np.ones_like(x)#1.0 + epsilon * np.tanh(x)

@njit(fastmath=True)
def delsig(x):
    return np.zeros_like(x)#epsilon * (1.0 - np.tanh(x)**2)

# Function to evolve all paths on a fixed time grid until their exponential clocks E_i are reached.
@njit(fastmath=True)
def _sde_kernel(x0, t_max, E, Eb, dW, dt):
    """
    Evolve all paths on a time grid until their exponential and gamme clocks E and Eb are reached.
    
    Parameters
    ----------
    x0 : 3D array (M, grid_size, d)         -- initial states, will be updated in place
    E : 2D array (M, grid_size)             -- exponential times
    Eb : 2D array (M, grid_size)            -- gamma times
    dW : 3D array (M, grid_size, t_max, d)  -- Brownian increments (M sample paths for each grid point)
    dt : float                              -- time step size
    """
    M, P = x0.shape[:2]  # number of paths, number of grid points
    #print(M, P)
    
    U = np.zeros((M, P, t_max, 1))     # Initializing Malliavin weights
    delX = np.ones((M, P, t_max, 1))   # Initializing gradient of X
    X = np.zeros((M, P, t_max, 1))   # Initializing X
    X_E = np.zeros((M, P, 1))          # X at exponential times
    X_Eb = np.zeros((M, P, 1))         # X at gamma times
    U_Eb = np.zeros((M, P, 1))         # Malliavin weights at gamma times

    X[:, :, 0, :] = x0

    for p in range(P):
        for i in range(M):
            T1 = int(E[i, p]//dt) + 1
            T2 = int(Eb[i, p]//dt) + 1
            for k in range(max(T1, T2) - 1):
                # Euler update for all grid points of path i and initial point p
                U[i, p, k+1, :] = U[i, p, k, :] + (sig(x0[i, p, :])/sig(X[i, p, k, :])) * delX[i, p, k, :] * dW[i, :, k, :]  
                delX[i, p, k+1, :] = delX[i, p, k, :] + delsig(X[i, p, k, :]) * delX[i, p, k, :] * dW[i, :, k, :]
                X[i, p, k+1, :] = X[i, p, k, :] + sig(X[i, p, k, :]) * dW[i, :, k, :]
            U[i, p] = (1/Eb[i, p]) * U[i, p]
            X_E[i, p, :] = X[i, p, T1-1, :] + (E[i, p] - (T1-1)*dt) * sig(X[i, p, T1-1, :]) * dW[i, :, T1-1, :]/dt
            X_Eb[i, p, :] = X[i, p, T2-1, :] + (Eb[i, p] - (T2-1)*dt) * sig(X[i, p, T2-1, :]) * dW[i, :, T2-1, :]/dt
            U_Eb[i, p, :] = U[i, p, T2-1, :] + ((Eb[i, p] - (T2-1)*dt) * (sig(x0[i, p, :])/sig(X[i, p, T2-1, :])) *
                                                   delX[i, p, T2-1, :] * dW[i, :, T2-1, :]/dt)
            #print("X, delX, U shapes", X.shape, delX.shape, U.shape)

    #print("X, delX, U shapes", X.shape, delX.shape, U.shape)
    return X_E, X_Eb, U_Eb

# The main class implementing the grid scheme for the 1-dimensional FBSDE
class GridScheme:
    def __init__(self, Ntilde=10, M=2000, K_z=0.1, r=1, R=4, NbP = 6, dt = 0.02, plot_iter=True):
        # Parameter initialization
        self.d = 1  # dimension X and BM
        self.dp = 1  # dimension of Y
        self.NbP = NbP  # number of Picard iterations
        self.R = R      # x domain size along each axis
        self.M = M  # number of Monte-Carlo samples
        self.r = r
        self.plot_iter = plot_iter

        # Time step for the SDE
        self.dt = dt

        # Spatial grid
        self.delta = R/Ntilde  # mesh size of the grid
        self.Ntilde = Ntilde+r  # 2*Ntilde+1 points in the grid in each axis
        self.x_grid = self.delta * (self.coordpoint(np.arange((2 * self.Ntilde + 1) ** self.d)) - self.Ntilde)
        self.grid_size = int((2 * self.Ntilde + 1) ** self.d)

        # Initialization of the functions u and ub (\bar{u})
        self.u = np.zeros((self.grid_size, self.d))
        self.ub = np.zeros((self.grid_size, self.d))
        # Choosing the f_0 component of the generator and defining the resulting generator
        self.c = 2.0  # Monotonicity constant mu = (c-1) = 1
        self.K_z = K_z  # z-Lipschitz constant K_{f,z} = 1

        # M samples of the exponential and gamma distributed times and Brownian motion.
        self.a = 2.0  # 7.0             # >0.5
        self.b = 2.0  # 7.0             # >1.0
        self.theta = 1.5
        self.theta_bar = 1.5

        # Log max errors
        self.log_max_u = None   #0.0
        self.log_max_ub = None  #0.0
        self.L1_err_u = None    #0.0
        self.L1_err_ub = None   #0.0

        # Parameters for sigma (from the SDE)
        #self.epsilon = 0.9

    # Function to convert a 1-d encoding of position vector into d-dimensions
    #@jit(nopython=True, fastmath=True)
    def coordpoint(self, x):
        result = np.zeros((len(x), self.d))
        for i in range(self.d):
            q, r = np.divmod(x, 2 * self.Ntilde + 1)
            #print("q and r shapes", i, q.shape, r.shape)
            result[:, i] = r
            x = q
        return result

    #@jit(nopython=True, fastmath=True)
    def coordpointinverse(self, x):
        return (np.sum(x * ((2 * self.Ntilde + 1) ** np.arange(self.d)), axis=-1)).astype('int32')

    #@jit(nopython=True, fastmath=True)
    def iterationbinary(self, vect, i=0):
        if vect[i] == 0:
            vect[i] += 1
        else:
            vect[i] = 0
            vect = self.iterationbinary(vect, i + 1)
        return vect

    #@jit(nopython=True, fastmath=True)
    def interpol(self, x, gridx):
        '''
        Performs linear interpolation
        x:      numpy array of coordinates of x in R^d
        gridx:  value of the function on the grid space converted to 1-d
        return: value of the interpolated function at point x
        '''
        # Projection of x outside the domain
        x[x > (self.Ntilde * self.delta)] = self.Ntilde * self.delta
        x[x < -(self.Ntilde * self.delta)] = -self.Ntilde * self.delta

        # coordinates of the cell where x lives
        aux = np.floor(x / self.delta) + self.Ntilde
        
        compteur = np.zeros(self.d)
        result = np.prod(1 - np.abs(x / self.delta + self.Ntilde - aux), axis=-1, keepdims=True) \
                 * gridx[self.coordpointinverse(aux)]
        
        for _ in range(2 ** self.d - 1):
            compteur = self.iterationbinary(compteur)
            xaux2 = aux + compteur[None, :]
            t = xaux2 > 2 * self.Ntilde
            result = result + np.prod(1 - np.abs(x / self.delta + self.Ntilde - xaux2), axis=-1, keepdims=True) * \
                     gridx[self.coordpointinverse(xaux2-t)]
        return result


    # Analytical solution for u
    #@jit(nopython=True, fastmath=True)
    def an_u(self, x):
        return (1/self.d)*(np.sum(np.arctan(x), axis=-1, keepdims=True))

    #@jit(nopython=True, fastmath=True)
    def an_ub(self, x):
        return (1 / self.d) * (1 / (1 + x**2)) * sig(x)
        #return (1/d)*np.sum(np.array([1/(1+x[i]**2) for i in range(len(x))]))

    #@jit(nopython=True, fastmath=True)
    def an_Delta_u(self, x):
        # Returns the value of the Laplacian of u at x (numpy array of coordinates)
        return (-2 / self.d) * np.sum(x / ((1 + x**2)**2), axis=-1, keepdims=True)
        #return (-2/d)*np.sum(np.array([x[i]/((1+x[i]**2)**2) for i in range(len(x))]))

    #@jit(nopython=True, fastmath=True)
    def f_0(self, x, y, z):
        # takes array inputs for x even for uni-dimensional case
        #return -c*y + np.cos(y + np.sqrt(np.sum(np.square(x)))) + K_z*np.sin(np.sqrt(np.sum(np.square(z))))
        #return -c * y + np.cos(y + x)
        #return -c * y + K_z * np.sin(np.sqrt(np.sum(np.square(z))))
        #return -c * y + np.cos(y + np.sqrt(np.sum(np.square(x))) + 1) + K_z * np.sin(np.sqrt(np.sum(np.square(z))))
        return -self.c * y + np.cos(y + np.sqrt(np.sum(np.square(x), axis=-1, keepdims=True))) \
               + self.K_z * np.sin(np.sqrt(np.sum(np.square(z), axis=-1, keepdims=True)))

    #@jit(nopython=True, fastmath=True)
    def f(self, x, y, z):
        # takes array inputs for x even for uni-dimensional case
        return self.f_0(x, y, z) - 0.5 * sig(x)**2 * self.an_Delta_u(x) - self.f_0(x, self.an_u(x), self.an_ub(x))

    #@jit(nopython=True, fastmath=True)
    def sampleE(self):
        # M samples of exponentially distributed time
        E = np.random.exponential(scale=1/self.theta, size=[self.M, self.grid_size])
        return E

    #@jit(nopython=True, fastmath=True)
    def sampleEb(self):
        # M samples of gamma distributed time
        Ebar = np.random.gamma(shape=0.5, scale=1/self.theta_bar, size=[self.M, self.grid_size])
        return Ebar


    #@jit(nopython=True, fastmath=True)
    def dW_tx(self, dt, t_max):
        # One sample for each x and t
        #W_E = x[None, :, None, :] + np.sqrt(dt) * np.random.randn(self.M, self.grid_size, t_max, self.d)
        dW_E = np.sqrt(dt) * np.random.randn(self.M, t_max, self.d)[:, None, :, :]
        return dW_E

    #def X(self, x, t):


    # Defining a utility function to to reshape a 1-d array into d-dimensions and clipping its boundard points
    def clipshape(self, x):
        x = np.reshape(x, shape=tuple([2*self.Ntilde+1]*self.d + [x.shape[-1]]))
        slicer = tuple([slice(self.r, -self.r)]*self.d + [slice(None)])
        return x[slicer]

    # Defining the operators Phi and \bar{Phi} from the paper.
    def phi(self, u_grid, ub_grid):
        # Takes a function defined on the 1-d converted grid space, interpolates it and applies Phi operator for each point
        # in the grid space and returns the new function defined on the grid space as a numpy array.
        E_grid = self.sampleE()
        Eb_grid = self.sampleEb()

        # Maximum of the E_grid and Eb_grid to define the time grid for the SDE
        tE_max = int(np.ceil(np.max(E_grid)/self.dt)) #+ 1
        tEb_max = int(np.ceil(np.max(Eb_grid)/self.dt)) #+ 1
        t_max = max(tE_max, tEb_max)
        #print("tmax", t_max)

        # Brownian motion increments for all paths and grid points.
        dW_grid = self.dW_tx(self.dt, t_max)
        #W_Eb_grid = self.dW_tx(self.x_grid, self.dt, tEb_max)

        temp = np.tile(self.x_grid[None, :, :], (self.M, 1, 1))
        #print("temp shape", temp.shape)
        
        # SDE evaluation at the exponential and gamma times.
        X_E_grid, X_Eb_grid, U_Eb = _sde_kernel(np.tile(self.x_grid[None, :, :], (self.M, 1, 1)),
                                            t_max, E_grid, Eb_grid, dW_grid, self.dt)
                              #lambda x: sig(self.epsilon, x), lambda x: delsig(self.epsilon, x))
        
        #print("shape X_E_grid", X_E_grid.shape)

        # Next, we interpolate u and \bar{u} at the above samples of exponential times.
        exp_interp_grid_u = self.interpol(X_E_grid, u_grid)
        exp_interp_grid_ub = self.interpol(X_E_grid, ub_grid)
        #print("shape", exp_interp_grid_u.shape, exp_interp_grid_ub.shape)
        #print("exp_interp_grid_u", exp_interp_grid_u[0])
        #print("exp_interp_grid_ub", exp_interp_grid_ub[0])

        # Next, we interpolate u and \bar{u} at the above samples of gamma times.
        gam_interp_grid_u = self.interpol(X_Eb_grid, u_grid)
        gam_interp_grid_ub = self.interpol(X_Eb_grid, ub_grid)
        #print("gam_interp_grid_u", gam_interp_grid_u[0])
        #print("gam_interp_grid_ub", gam_interp_grid_ub[0])

        # Now, we can apply the operator on the interpolated versions of u and \bar{u}
        phi_Pu = np.mean((self.f(X_E_grid, exp_interp_grid_u, exp_interp_grid_ub) + self.a * exp_interp_grid_u)
                 * (np.e ** (-E_grid[:, :, None] * (self.a - self.theta))) * 1/self.theta, axis=0)

        #phi_Pu = np.mean((f(W_E_grid, an_u(x_grid)[None, :, :], an_ub(x_grid)[None, :, :]) + a * an_u(x_grid)[None, :, :])
        #                 * (np.e ** (-E_grid[:, :, None] * (a - theta))) * 1 / theta, axis=0)

        phi_bar_Pu = np.mean((self.f(X_Eb_grid, gam_interp_grid_u, gam_interp_grid_ub) + self.b * gam_interp_grid_u)
                             * (np.e ** (-Eb_grid[:, :, None] * (self.b - self.theta)))
                             * np.sqrt(Eb_grid[:, :, None]) * U_Eb #* (X_Eb_grid - self.x_grid[None, :, :])/Eb_grid[:, :, None]
                             * np.sqrt(np.pi/self.theta_bar), axis=0)

        return phi_Pu, phi_bar_Pu


    def PicIter(self):
        if self.d == 1:
            u_grid = np.zeros((self.grid_size, 1))
            ub_grid = np.zeros((self.grid_size, 1))

            pic_err_u = []
            pic_err_ub = []

            #domain = (np.arange(2 * (self.Ntilde - self.r) + 1) - 
            #          (self.Ntilde - self.r)) * self.delta
                  
            #an_u_grid = np.array([self.an_u(x) for x in domain]).reshape(len(domain), 1)
            #an_ub_grid = np.array([self.an_ub(x) for x in domain]).reshape(len(domain), 1)
            #print("shapes 1 an_u", an_u_grid.shape, an_ub_grid.shape)
            an_u_grid = self.an_u(self.clipshape(self.x_grid))
            an_ub_grid = self.an_ub(self.clipshape(self.x_grid))
            #print("shapes 2 an_u", an_u_grid.shape, an_ub_grid.shape)

            if self.plot_iter:
                fig = plt.figure(figsize=(12, 5), dpi=75)
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.set_title("$u(x)$")
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.set_title(r"$\bar{u}(x)$")
                #xx = np.array([[-self.delta * self.Ntilde + (2 * self.delta * self.Ntilde / 1000) * i] for i in range(1001)])
                xx = np.linspace(-self.delta*(self.Ntilde - self.r), self.delta*(self.Ntilde - self.r), 1000)
                yy = np.array([self.an_u(xx[i]) for i in range(len(xx))])
                yyb = np.array([self.an_ub(xx[i]) for i in range(len(xx))])
                #yy_Delta = np.array([self.an_Delta_u(xx[i]) for i in range(len(xx))])
                ax1.plot(xx, yy, color="black", label="Analytical $u(x)$")
                ax2.plot(xx, yyb, color="brown", label=r"Analytical $\bar{u}(x)$")


            # Picard iterations



            for p in range(self.NbP):
                print("Running Picard iteration: ", p+1)
                u_grid, ub_grid = self.phi(u_grid, ub_grid)
                #print("shapes", u_grid.shape, ub_grid.shape)
                if p % 2 == 1 and self.plot_iter:
                #if p == self.NbP-1:
                    ax1.plot(self.clipshape(self.x_grid), self.clipshape(u_grid), 'x',
                             label="Iteration {}".format(p + 1))
                    ax2.plot(self.clipshape(self.x_grid), self.clipshape(ub_grid), 'x',
                             label="Iteration {}".format(p + 1))
                #err_u = np.linalg.norm(an_u_grid - self.clipshape(u_grid), ord=None)
                #err_ub = np.linalg.norm(an_ub_grid - self.clipshape(ub_grid), ord=None)
                #print("shapes err_u :", err_u.shape)
                err_u = an_u_grid - self.clipshape(u_grid)
                err_ub = an_ub_grid - self.clipshape(ub_grid)
                #print("shapes err_u :", err_u.shape)
                pic_err_u.append(err_u)
                pic_err_ub.append(err_ub)

            #print("pic_err_u = ", pic_err_u)
            #print("pic_err_ub = ", pic_err_ub)

            print("Ntilde = ", self.Ntilde, "r = ", self.r, "pic_err_u_max = ", 
                  np.max(pic_err_u[-1]), "log_err_u_max = ", 
                  np.log(np.max(pic_err_u[-1])))
            print("Ntilde = ", self.Ntilde, "r = ", self.r,
                  "pic_err_ub_max = ", np.max(pic_err_ub[-1]),
                  "log_err_ub_max = ", np.log(np.max(pic_err_ub[-1])))

            #print("shapes = ", np.array(pic_err_u).shape, np.array(pic_err_ub).shape)
            self.log_max_u = np.log(np.max(pic_err_u[-1]))
            self.log_max_ub = np.log(np.max(pic_err_ub[-1]))

            self.L1_err_u = np.mean(np.array(pic_err_u)[-1]) #, axis=0)
            self.L1_err_ub = np.mean(np.array(pic_err_ub)[-1]) #, axis=0)
            #print(self.L1_err_u.shape)
            #print("len", len(pic_err_u))
            
            if self.plot_iter:
                ax1.set_xlabel("$x$")
                ax1.set_ylabel("$u(x)$")
                ax1.legend(loc='upper left')
                ax2.set_xlabel("$x$")
                ax2.set_ylabel(r"$\bar{u(x)}$")
                ax2.legend(loc='upper left')

                print("Total time taken: ", time.time() - t_init)
                plt.show()
                fig.savefig("Numerical_experiments/Grid_scheme/1d_generalSDE/u_and_ub_R{}_dt_{}_sig0.png".format(self.R, self.dt), bbox_inches='tight')



            #domain = (np.arange(2 * Ntilde + 1) - Ntilde) * delta
            #domain = (np.arange(Ntilde + 1) - int(Ntilde/2)) * delta

            #err_u = an_u_grid-u_grid[int(Ntilde/2):3*int(Ntilde/2)+1]
            #err_ub = an_ub_grid-ub_grid[int(Ntilde/2):3*int(Ntilde/2)+1]
            #mean_err_u = np.mean(np.abs(an_u_grid-u_grid))# [int(Ntilde/2):3*int(Ntilde/2)+1]))
            #max_err_u = np.max(np.abs(an_u_grid-u_grid))#[int(Ntilde/2):3*int(Ntilde/2)+1]))
            #argmax_u = np.argmax(np.abs(an_u_grid-u_grid))#[int(Ntilde/2):3*int(Ntilde/2)+1]))

            #mean_err_ub = np.mean(np.abs(an_ub_grid - ub_grid))#[int(Ntilde / 2):3 * int(Ntilde / 2) + 1]))
            #max_err_ub = np.max(np.abs(an_ub_grid - ub_grid))#[int(Ntilde / 2):3 * int(Ntilde / 2) + 1]))
            #argmax_ub = np.argmax(np.abs(an_ub_grid - ub_grid))#[int(Ntilde / 2):3 * int(Ntilde / 2) + 1]))

            #print("mean_err_u = ", mean_err_u, "; max_err_u = ", max_err_u, "; argmax = ", argmax_u)
            #print("mean_err_ub = ", mean_err_ub, "; max_err_ub = ", max_err_ub, "; argmax = ", argmax_ub)
            #plt.show()


        '''
        elif self.d == 2:
            # The 2-dimensional grid for plots
            x_axis_0 = self.delta*np.arange(2*(self.Ntilde-self.r)+1) - (self.Ntilde-self.r)
            x_axis_1 = self.delta*np.arange(2*(self.Ntilde-self.r)+1) - (self.Ntilde-self.r)
            x_grid_eval_0, x_grid_eval_1 = np.meshgrid(x_axis_0, x_axis_1)

            #clipped_grid = self.delta * (self.coordpoint(np.arange((2 * (self.Ntilde-self.r) + 1) ** self.d)) - (self.Ntilde-self.r))
            #print("shapes clipped_grid", clipped_grid.shape)

            # Initialization of u and ub on the full grid with boundary points
            u_grid = np.zeros((self.grid_size, 1))
            ub_grid = np.zeros((self.grid_size, self.d))

            pic_err_u = []
            pic_err_ub = []

            if self.plot_iter:
                fig = plt.figure(figsize=(24, 7), dpi=75, tight_layout=True)
            
            r_slice = slice(self.r, -self.r)

            #print("clipped shape", self.clipshape(u_grid).shape,
            #      u_grid.reshape(2*self.Ntilde+1, 2*self.Ntilde+1)[r_slice, r_slice].shape)

            for p in range(self.NbP):
                print("Picard iteration, p = ", p + 1)
                u_grid, ub_grid = self.phi(u_grid, ub_grid)
                print("shapes", u_grid.shape, ub_grid.shape)
                # if (p % 3 == 0):

                err_u = np.max(np.abs(self.an_u(self.clipshape(self.x_grid)) - self.clipshape(u_grid)))
                err_ub = [np.max(np.sum(np.abs(self.an_ub(self.clipshape(self.x_grid))[:, i, None] - self.clipshape(ub_grid[:, i, None]))))
                           for i in range(self.d)]

                #print("shapes: ", self.an_ub(self.x_grid).shape, err_u.shape, err_ub.shape)

                pic_err_u.append(err_u)
                pic_err_ub.append(err_ub)

                if p == self.NbP-1 and self.plot_iter:
                    
                    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
                    ax1.plot_surface(x_grid_eval_0, x_grid_eval_1, self.clipshape(u_grid)[:, :, 0]
                                      ,label=p + 1)
                    ax1.plot_wireframe(x_grid_eval_0, x_grid_eval_1, 
                                       self.an_u(self.clipshape(self.x_grid))[:, :, 0],
                                       color='black', label=p + 1, linestyle='dashed', alpha=0.9, linewidth=0.7)
                    ax1.set_title(r"$u^n(x^1, x^2)$, $n={}$".format(p + 1))

                    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
                    ax2.plot_surface(x_grid_eval_0, x_grid_eval_1, 
                                     self.clipshape(ub_grid[:, 0, None])[:, :, 0]
                                     , color='red', label=p + 1)
                    ax2.plot_wireframe(x_grid_eval_0, x_grid_eval_1, 
                                       self.an_ub(self.clipshape(self.x_grid))[:, :, 0]
                                       , color='black', label=p + 1, linestyle='dashed', alpha=0.9, linewidth=0.7)
                    ax2.set_title(r"$\bar u^{1, n}(x^1, x^2)$, $n=%s$" % (p + 1))

                    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
                    ax3.plot_surface(x_grid_eval_0, x_grid_eval_1, 
                                     self.clipshape(ub_grid[:, 1, None])[:, :, 0]
                                     ,color='red', label=p + 1)
                    ax3.plot_wireframe(x_grid_eval_0, x_grid_eval_1,
                                       self.an_ub(self.clipshape(self.x_grid))[:, :, 1]
                                       ,color='black', label=p + 1, linestyle='dashed', alpha=0.9, linewidth=0.7)
                    ax3.set_title(r"$\bar u^{1, n}(x^1, x^2)$, $n=%s$" % (p + 1))

                    #fig.savefig('Numerical_experiments/Grid_scheme/2d/ub_1_and_ub_2_iter_{}.png'.format(p + 1),
                    #            bbox_inches='tight')
                    plt.show()

            self.L1_err_u = pic_err_u
            self.L1_err_ub = pic_err_ub
            '''

if __name__ == "__main__":
    A = GridScheme(r=2, R=4, M=8000, NbP=10, dt=0.005, plot_iter=True)
    t_init = time.time()
    A.PicIter()
