import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
import time
import logging
from datetime import datetime


class NNPicardSolver:
    """Class for solving infinite-horizon BSDEs using Neural Network Picard iterations."""
    
    def __init__(self, d=1, dp=1, num_pic=5, M=30000, M_err=1000, Ntilde=21, 
                 batch_size=1000, epochs=200, activation='relu',
                 initial_learning_rate=5e-4, lr_decay=0.97, decay_steps=1000,
                 a=2.0, b=2.0, theta=1.5, theta_bar=1.5, sig_X=2.0, K_z=1.0):
        """
        Initialize the NN Picard Solver with parameters.
        
        Args:
            d: dimension of X and BM
            dp: dimension of Y
            num_pic: number of Picard iterations
            M: number of training samples
            M_err: number of samples for error computations
            Ntilde: number of points in each axis of grid
            batch_size: batch size for training
            epochs: number of training epochs
            activation: activation function
            initial_learning_rate: initial learning rate
            lr_decay: learning rate decay rate
            decay_steps: decay steps for learning rate schedule
            a, b, theta, theta_bar: scheme hyperparameters
            sig_X: standard deviation for sampling X
            K_z: Lipschitz constant for z
        """
        self.d = d
        self.dp = dp
        self.num_pic = num_pic
        self.M = M
        self.M_err = M_err
        self.Ntilde = Ntilde
        self.batch_size = batch_size
        self.epochs = epochs
        self.activation = activation
        self.initial_learning_rate = initial_learning_rate
        self.lr_decay = lr_decay
        self.decay_steps = decay_steps
        self.a = a
        self.b = b
        self.theta = theta
        self.theta_bar = theta_bar
        self.sig_X = sig_X
        self.K_z = K_z
        self.u_err = None
        self.ub_err = None
        
        # Setup logging
        logging.basicConfig(filename='NN_Picard_mult.log', level=logging.INFO)
        time_now = datetime.now()
        logging.info('Time:{}'.format(time_now))
        logging.info('K_z={}'.format(K_z))
        
        # Initialize models
        self._build_models()
    
    def coordpoint(self, x):
        """Convert 1-d encoding of position vector into d-dimensions."""
        result = np.zeros(self.d)
        for i in range(self.d):
            q, r = divmod(x, self.Ntilde)
            result[i] = r
            x = q
        return result
    
    def an_u(self, x):
        """Analytical solution for u."""
        return (1 / self.d) * np.sum(np.arctan(x), axis=-1, keepdims=True)
    
    def an_ub(self, x):
        """Analytical solution for ub (gradient of u)."""
        return (1 / self.d) * (1 / (1 + x**2))
    
    def an_Delta_u(self, x):
        """Analytical Laplacian of u."""
        return (-2 / self.d) * np.sum(x / ((1 + x**2)**2), axis=-1, keepdims=True)
    
    def f_0(self, x, y, z):
        """f_0 component of the generator."""
        return -self.a*y + np.cos(y + np.sqrt(np.sum(np.square(x), axis=-1, keepdims=True))) \
               + self.K_z * np.sin(np.sqrt(np.sum(np.square(z), axis=-1, keepdims=True)))
    
    def f(self, x, y, z):
        """Full generator f."""
        return self.f_0(x, y, z) - 0.5*self.an_Delta_u(x) - self.f_0(x, self.an_u(x), self.an_ub(x))
    
    def sampleE(self, num_sample=None):
        """Sample exponential distributed times."""
        if num_sample is None:
            num_sample = self.M
        return np.random.exponential(scale=1/self.theta, size=[num_sample, 1])
    
    def sampleEb(self, num_sample=None):
        """Sample gamma distributed times."""
        if num_sample is None:
            num_sample = self.M
        return np.random.gamma(shape=0.5, scale=1/self.theta_bar, size=[num_sample, 1])
    
    def sampleX(self, sig=None, num_sample=None):
        """Sample from normal distribution."""
        if sig is None:
            sig = self.sig_X
        if num_sample is None:
            num_sample = self.M
        return np.random.normal(loc=0.0, scale=sig, size=[num_sample, self.d])
    
    def phi(self, E, E_bar, x, w_E, w_E_bar, u_E, ub_E, u_E_bar, ub_E_bar, u_x, ub_x):
        """Compute phi function for labels."""
        phi = (1/self.theta) * (self.f(w_E, u_E, ub_E) + self.a * u_E) * (np.e ** (-E * (self.a - self.theta)))
        
        phi_bar_var_red = np.sqrt(np.pi / self.theta_bar).astype('float32') * \
                          (self.f(w_E_bar, u_E_bar, ub_E_bar) + self.b * u_E_bar - self.f(x, u_x, ub_x) - self.b * u_x) * \
                          (np.e ** (-E_bar * (self.b - self.theta_bar))) * \
                          np.sqrt(E_bar).astype('float32') * \
                          tf.cast((w_E_bar - x) / E_bar, tf.float32)
        
        return [phi, phi_bar_var_red]
    
    def label(self, X, prev_model):
        """Generate labels for training."""
        E = self.sampleE(len(X))
        Eb = self.sampleEb(len(X))
        W = np.random.randn(len(X), self.d)
        W_E = X + np.sqrt(E) * W
        W_Eb = X + np.sqrt(Eb) * W
        
        prev = prev_model(W_E)
        u_E = prev[:, :1]
        ub_E = prev[:, 1:]
        
        prev = prev_model(W_Eb)
        u_Eb = prev[:, :1]
        ub_Eb = prev[:, 1:]
        
        prev_X = prev_model(X)
        u_x = prev_X[:, :1]
        ub_x = prev_X[:, 1:]
        
        label = self.phi(E, Eb, X, W_E, W_Eb, u_E, ub_E, u_Eb, ub_Eb, u_x, ub_x)
        label_0 = label[0]
        label_1 = label[1]
        
        return tf.concat([label_0, label_1], axis=1)
    
    def loss_fn(self, y_label, y_pred):
        """Custom loss function."""
        return tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(y_label - y_pred), axis=0, keepdims=True)))
    
    def _build_models(self):
        """Build initial and NN models."""
        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.lr_decay,
            staircase=True)
        
        # Initial model
        self.model_init = tf.keras.Sequential([layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], self.d+1)))])
        self.model_init.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                                loss=self.loss_fn,
                                metrics=['accuracy'])
        
        # Batch normalization layers
        bn_layers = [
            tf.keras.layers.BatchNormalization(
                axis=1,
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(3)]
        
        # NN model
        self.NN = tf.keras.Sequential([layers.Dense(20 + self.d, input_shape=(self.d,), activation=self.activation),
                              bn_layers[0],
                              layers.Dense(20 + self.d, activation=self.activation),
                              bn_layers[1],
                              layers.Dense(1 + self.d, activation=None)])
        
        self.NN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                        loss=self.loss_fn,
                        metrics=['accuracy'])
    
    def main(self):
        """Main training and evaluation loop."""
        print("K_z=", self.K_z)
        start_time = time.time()
        
        if self.d == 1:
            self._run_1d()
        elif self.d == 2:
            self._run_2d()
        else:
            self._run_high_dim()
        
        elapsed_time = time.time() - start_time
        print("elapsed time: ", elapsed_time)
        logging.info('Elapsed time: {}'.format(elapsed_time))
        
        return elapsed_time
    
    def _run_1d(self):
        """Run for 1D case."""
        x_axis = np.linspace(-3.0, 3.0, self.Ntilde).reshape(-1, 1)
        x_axis_1 = np.linspace(-3.0, 3.0, 10*self.Ntilde+1).reshape(-1, 1)
        fig = plt.figure(figsize=(11, 5), dpi=75, tight_layout=True)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("$u(x)$")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title(r"$\bar{u}(x)$")
        
        yy = self.an_u(x_axis_1)
        yyb = self.an_ub(x_axis_1)
        ax1.plot(x_axis_1, yy, color='black', label=r"Analytical ${u}(x)$")
        ax2.plot(x_axis_1, yyb, color='brown', label=r"Analytical $\bar{u}(x)$")
        
        model_0 = self.model_init
        model_1 = self.NN
        
        for p in range(self.num_pic):
            print("Picard iteration ", p + 1)
            
            X_train = self.sampleX()
            Y_train = self.label(X_train, model_0)
            
            model_1.fit(X_train, Y_train,
                        batch_size=self.batch_size,
                        shuffle=False,
                        epochs=self.epochs,
                        verbose=1)
            
            model_0 = model_1
            
            predict = model_1.predict(x_axis)
            if p % 1 == 0:
                ax1.plot(x_axis, predict[:, 0], 'x', label="Iteration {}".format(p + 1))
                ax2.plot(x_axis, predict[:, 1], 'x', label="Iteration {}".format(p + 1))
        
        x_axis_err = self.sampleX(num_sample=self.M_err)
        predict_err = model_1.predict(x_axis_err)
        self.u_err = np.abs(predict_err[:, :1] - self.an_u(x_axis_err)).reshape(self.M_err)
        print("shape 1= ", self.u_err.shape)
        self.ub_err = np.sqrt(np.mean((predict_err[:, 1:] - self.an_ub(x_axis_err)) ** 2, axis=-1))
        print("shape 2= ", self.ub_err.shape)
        np.save('Numerical_experiments/error_plots/u_Kz_{}_{}d.npy'.format(self.K_z, self.d), self.u_err)
        np.save('Numerical_experiments/error_plots/ub_Kz_{}_{}d.npy'.format(self.K_z, self.d), self.ub_err)
        print("ave", np.mean(self.u_err), np.mean(self.ub_err))
        print("max", np.max(self.u_err), np.max(self.ub_err))
        
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$u(x)$")
        ax1.legend(loc='upper left')
        ax2.set_xlabel("$x$")
        ax2.set_ylabel(r"$\bar{u(x)}$")
        ax2.legend(loc='upper left')
        plt.show()
    
    def _run_2d(self):
        """Run for 2D case."""
        x_axis_0 = np.linspace(-3.0, 3.0, self.Ntilde)
        x_axis_1 = np.linspace(-3.0, 3.0, self.Ntilde)
        x_grid_eval_0, x_grid_eval_1 = np.meshgrid(x_axis_0, x_axis_1)
        
        x_eval_points = np.array([[[x_axis_0[i], x_axis_1[j]] for j in range(self.Ntilde)] for i in range(self.Ntilde)]).reshape(-1, 2)
        
        model_0 = self.model_init
        model_1 = self.NN
        
        for p in range(self.num_pic):
            print("Picard iteration, p = ", p+1)
            
            X_train = self.sampleX()
            Y_train = self.label(X_train, model_0)
            
            model_1.fit(X_train, Y_train,
                        batch_size=self.batch_size,
                        shuffle=False,
                        epochs=self.epochs,
                        verbose=1)
            
            model_0 = model_1
            
            predict = model_1.predict(x_eval_points)
            
            fig = plt.figure(figsize=(24, 7), dpi=75)
            ax = fig.add_subplot(1, 3, 1, projection='3d')
            ax.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 0].reshape(self.Ntilde, self.Ntilde), label=p+1)
            ax.plot_wireframe(x_grid_eval_0, x_grid_eval_1, self.an_u(x_eval_points).reshape(self.Ntilde, self.Ntilde), color='black', label=p+1)
            ax.set_title(r"$u^n(x^1, x^2)$, $n={}$".format(p+1))
            
            ax = fig.add_subplot(1, 3, 2, projection='3d')
            ax.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 1].reshape(self.Ntilde, self.Ntilde), color='red', label=p + 1)
            ax.plot_wireframe(x_grid_eval_0, x_grid_eval_1, self.an_ub(x_eval_points)[:, 0].reshape(self.Ntilde, self.Ntilde), color='gray',
                              label=p + 1)
            ax.set_title(r"$\bar u^{1, n}(x^1, x^2)$, $n=%s$" %(p+1))
            
            ax = fig.add_subplot(1, 3, 3, projection='3d')
            ax.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 2].reshape(self.Ntilde, self.Ntilde), color='red', label=p + 1)
            ax.plot_wireframe(x_grid_eval_0, x_grid_eval_1, self.an_ub(x_eval_points)[:, 1].reshape(self.Ntilde, self.Ntilde), color='gray',
                              label=p + 1)
            ax.set_title(r"$\bar u^{1, n}(x^1, x^2)$, $n=%s$" %(p+1))
            
            #plt.savefig('plots_article/ub_1_and_ub_2_iter_{}.pdf'.format(p + 1), bbox_inches='tight', dpi=300)
            plt.show()

        x_axis_err = self.sampleX(num_sample=self.M_err)
        predict_err = model_1.predict(x_axis_err)
        self.u_err = np.abs(predict_err[:, :1] - self.an_u(x_axis_err)).reshape(self.M_err)
        print("shape 1= ", self.u_err.shape)
        self.ub_err = np.sqrt(np.mean((predict_err[:, 1:] - self.an_ub(x_axis_err)) ** 2, axis=-1))
        print("shape 2= ", self.ub_err.shape)
        np.save('Numerical_experiments/error_plots/u_Kz_{}_{}d.npy'.format(self.K_z, self.d), self.u_err)
        np.save('Numerical_experiments/error_plots/ub_Kz_{}_{}d.npy'.format(self.K_z, self.d), self.ub_err)
        print("ave", np.mean(self.u_err), np.mean(self.ub_err))
        print("max", np.max(self.u_err), np.max(self.ub_err))
    
    def _run_high_dim(self):
        """Run for high dimensional case (d >= 3)."""
        errors = []
        
        model_0 = self.model_init
        model_1 = self.NN
        
        for p in range(self.num_pic):
            print("Picard iteration, p = ", p+1)
            
            X_train = self.sampleX()
            Y_train = self.label(X_train, model_0)
            
            model_1.fit(X_train, Y_train,
                        batch_size=self.batch_size,
                        shuffle=False,
                        epochs=self.epochs,
                        verbose=1)
            
            model_0 = model_1
            
            X_test = self.sampleX(sig=0.8*self.sig_X, num_sample=int(0.1*self.M))
            Y_test = model_1.predict(X_test)
            
            u_pred = Y_test[:, :1]
            ub_pred = Y_test[:, 1:]
            
            an_u_test = self.an_u(X_test)
            an_ub_test = self.an_ub(X_test)
            
            L2_err_u = np.sqrt(np.mean(np.mean((u_pred - an_u_test) ** 2, axis=-1)))
            L_inf_err_u = np.max(np.mean(np.abs(u_pred - an_u_test), axis=-1))
            arg_L_inf_err_u = X_test[np.argmax(np.mean(np.abs(u_pred - an_u_test), axis=-1))]
            
            print("mean L^2 error for u:", L2_err_u, "\t L^inf error for u:", L_inf_err_u, "\t the error is maximum at:",
                  arg_L_inf_err_u)
            
            L2_err_ub = np.sqrt(np.mean(np.mean((ub_pred - an_ub_test) ** 2, axis=-1)))
            L_inf_err_ub = np.max(np.mean(np.abs(ub_pred - an_ub_test), axis=-1))
            arg_L_inf_err_ub = X_test[np.argmax(np.mean(np.abs(ub_pred - an_ub_test), axis=-1))]
            
            print("mean L^2 error for u_bar:", L2_err_ub, "\t L^inf error for u_bar:", L_inf_err_ub,
                  "\t the error is maximum at:", arg_L_inf_err_ub)
            
            errors += [(L2_err_u, L_inf_err_u, arg_L_inf_err_u, L2_err_ub, L_inf_err_ub, arg_L_inf_err_ub)]
        
        x_axis_err = self.sampleX(num_sample=self.M_err)
        predict_err = model_1.predict(x_axis_err)
        self.u_err = np.abs(predict_err[:, :1] - self.an_u(x_axis_err)).reshape(self.M_err)
        print("shape 1= ", self.u_err.shape)
        ub_err = np.sqrt(np.mean((predict_err[:, 1:] - self.an_ub(x_axis_err)) ** 2, axis=-1))
        print("shape 2= ", ub_err.shape)
        np.save('Numerical_experiments/error_plots/u_Kz_{}_{}d.npy'.format(self.K_z, self.d), self.u_err)
        np.save('Numerical_experiments/error_plots/ub_Kz_{}_{}d.npy'.format(self.K_z, self.d), self.ub_err)
        print("ave", np.mean(self.u_err), np.mean(self.ub_err))
        print("max", np.max(self.u_err), np.max(self.ub_err))


if __name__ == "__main__":
    # Example usage with default parameters
    solver = NNPicardSolver(d=1, K_z=1.0)
    solver.main()
