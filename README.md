# Infinite horizon BSDEs

## Mathematical setting

This project proposes numerical solutions of infinite-horizon backward stochastic differential equations (BSDEs). For an in-depth coverage, refer to the [PhD thesis](https://theses.hal.science/tel-04627360v1) of Charu Shardul done under the supervision of Prof. Emmanuel Gobet and Prof. Adrien Richou. 

We consider BSDEs of the following type:

$$Y_t = Y_T + \int_t^T f(X_s, Y_s, Z_s) ds -\int_t^T Z_s d W_s, \qquad \forall 0\leqslant t\leqslant T < +\infty,$$

where X is the solution of the following d-dimensional SDE:

$$X_t = x + \int_0^t b(X_s) ds + \int_0^t \sigma(X_s) dW_s, \quad 0 \leq t, $$

where $X$ and $W$ have the same dimension $d$. Under certain assumptions, this decoupled system of forward and backward SDEs is equivalent to the following system of elliptic partial differential equations (PDEs):

$$ \mathcal L u(x) + f(x, u(x), \nabla_x u(x)\sigma(x)) = 0, \quad \forall x\in\mathbb R^d, $$
where $\mathcal L$ is the generator of the semi-group associated with the above SDE for $X$.

We propose three algorithms based on two main approaches-
 1. Starting from an initial guess for $u(x)$ and $\bar{u}(x)$ $\left(:=\nabla_x u(x)\sigma(x) \right)$, we propose an iterative procedure by repeatedly applying an operator which is a contraction under some additional conditions.
 2. The second approach, based on neural networks (NNs), is a direct approach which bypasses the need for contraction.

 For the first approach, we have the following files:

1. `vect_grid_scheme.py` — vectorized space grid-based Picard iterations (Brownian forward process).
2. `1d_generalSDE.py` — grid-based Picard iterations where the forward process is a 1‑D SDE.
3. `NN_Picard_mult.py` — Picard iterations with NNs replacing the spatial grid (suitable for higher d).
   
In addition, we have `NN_direct_scheme_2.py`, which implements the second approach based on the direct NN scheme that bypasses the contraction operator (direct solver).



## Numerical Example


We consider an infinite-horizon BSDE denoted by $(Y, Z)$ with X denoting the forward process (Brownian motion or SDE). We will only consider the case when $Y$ is one-dimensional, i.e. a single elliptic PDE. The BSDE has generator f and we seek the stationary pair $(u, \bar u)$ that satisfy a fixed-point formulation similar to the Feynman–Kac representation - 

$$ u^{i+1} = \Phi(u^i, \bar u^i)(x) = \mathbb E\Big[(f(X_E, u(X_E), \bar u(X_E)) + a u(X_E))\ w(E)\Big], $$

$$ \bar{u}^{i+1} = \bar \Phi(u^i, \bar u^i)(x) = \mathbb E\Big[\big(f(X_{\bar{E}}, u(X_{\bar{E}}), \bar u(X_{\bar{E}})) + b u(X_{\bar{E}})\big) \ U_{\bar{E}}^x \ \bar{w}({\bar{E}})\Big], $$

where $w$ and $\bar{w}$ are some exponential functions and $U_t^x$ are the Malliavin weights. For simplicity and without loss of generality, we consider the drift of the forward SDE to be zero,
$$dX^x_t = \sigma(X^x_t) dW_t, \quad X^x_0 = x,$$
where we consider the following two cases for $\sigma(x)$:
1. $\sigma(x) = 1$, meaning that $X^x_t = x + W_t$, eliminating the need to simulate paths of the SDE. We consider this setting for `vect_grid_scheme.py`, `NN_Picard_mult.py` and `NN_direct_scheme_2.py`.
2. $\sigma(x) = 1 + \varepsilon \tanh(x)$ for instance, where $\varepsilon=0.9$. We use this setting for `1d_generalSDE.py`.

Next, for the generator and the analytical solutions for $u$ and $\bar u$, we take the following,

$$f(x, y, z) = f_0(x,y,z) - \dfrac{1}{2}Tr\Big((\nabla^2 u(x))\sigma \sigma^\top\Big) - f_0(x, u(x), \nabla_x u(x)\sigma(x)),$$
where
$$f_0(x, y, z) = -cy + \cos{(y+|x|)} + K_z \sin(|z|)$$

$$u(x) = \frac{1}{d} \sum_{i=1}^d \tan^{-1}(x_i)$$
$$\bar{u}(x) = \dfrac{1}{d}\left[\dfrac{1}{1+x_i^2} \right]_{i\in\{1, \dots, d\}} \sigma(x)$$

The code implements Monte‑Carlo estimation of expectations above and interpolation (grid methods) or direct NN learning of the operators.


## Dependencies

* Using console commands:

    - conda create -n infBSDE python=3.10 numpy matplotlib numba -y

    - conda activate infBSDE

    - pip install tensorflow==2.14

* Using `requirements.txt`:
    - pip install -r requirements.txt

## Notes on extending / experiments

- Replace `an_u`/`an_ub` and `f_0` with other models to test other analytic examples.
- For higher-d NN experiments increase network width/depth, tune M (MC samples), and use batch normalization / learning‑rate schedules.
- For SDE forward processes in >1D, extend `_sde_kernel`, `sigma`, `del_sigma` and Malliavin-weight computations (currently implemented for 1-d).
- For new experiments with different model parameters, create lists of new class objects and store the required data or save the computed error values using `np.save()` command and plot them later.

---

