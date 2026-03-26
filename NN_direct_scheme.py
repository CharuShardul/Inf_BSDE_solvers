import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
import time
import logging
from datetime import datetime


class NNDirecSolver:
    """Class for direct NN BSDE scheme."""

    def __init__(self, d=1, dp=1, Mx=512, M=3000, M_err=1000, Ntilde=21, activation='relu', initial_learning_rate=5e-4, lr_decay=0.6,
                n_decays=10, n_steps=3000, batch_size=512, epochs=500, a=2.0, b=2.0, theta=1.5, theta_bar=1.5, sig_X=2.0, c=2.0,
                K_z=1.0, update_frequency=100):
        self.d = d
        self.dp = dp
        self.Mx = Mx
        self.M = M
        self.M_err = M_err
        self.Ntilde = Ntilde
        self.activation = activation
        self.initial_learning_rate = initial_learning_rate
        self.lr_decay = lr_decay
        self.n_decays = n_decays
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.a = a
        self.b = b
        self.theta = theta
        self.theta_bar = theta_bar
        self.sig_X = sig_X
        self.c = c
        self.K_z = K_z
        self.update_frequency = update_frequency
        self.u_err = None
        self.ub_err = None

        logging.basicConfig(level=logging.INFO)
        tf.keras.backend.set_floatx('float64')
        time_now = datetime.now()
        logging.info('Time:{}'.format(time_now))
        logging.info('K_z={}'.format(K_z))

        self._build_models()

    def coordpoint(self, x):
        result = np.zeros(self.d)
        for i in range(self.d):
            q, r = divmod(x, self.Ntilde)
            result[i] = r
            x = q
        return result

    def an_u(self, x):
        return (1 / self.d) * tf.reduce_sum(tf.math.atan(x), axis=-1, keepdims=True)

    def an_ub(self, x):
        return (1 / self.d) * (1 / (1 + x**2))

    def an_Delta_u(self, x):
        return (-2 / self.d) * tf.reduce_sum(x / ((1 + x**2)**2), axis=-1, keepdims=True)

    def f_0(self, x, y, z):
        return -self.c * y + tf.cos(y + tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))) \
               + self.K_z * np.sin(np.sqrt(np.sum(np.square(z), axis=-1, keepdims=True)))

    def f(self, x, y, z):
        return self.f_0(x, y, z) - 0.5 * self.an_Delta_u(x) - self.f_0(x, self.an_u(x), self.an_ub(x))

    def sampleE(self, num_sample=None):
        if num_sample is None:
            num_sample = self.Mx
        return np.random.exponential(scale=1 / self.theta, size=[num_sample, 1])

    def sampleEb(self, num_sample=None):
        if num_sample is None:
            num_sample = self.Mx
        return np.random.gamma(shape=0.5, scale=1 / self.theta_bar, size=[num_sample, 1])

    def sampleX(self, sig=None, num_sample=None):
        if sig is None:
            sig = self.sig_X
        if num_sample is None:
            num_sample = self.Mx
        return np.random.normal(loc=0.0, scale=sig, size=[num_sample, self.d])

    def phi(self, E, E_bar, x, w_E, w_E_bar, u_E, ub_E, u_E_bar, ub_E_bar, u_x, ub_x):
        phi_0 = (1 / self.theta) * tf.reduce_mean((self.f(w_E, u_E, ub_E) + self.a * u_E) *
                                                 (np.e ** (-E * (self.a - self.theta))), axis=1)

        phi_bar_var_red = np.sqrt(np.pi / self.theta_bar) * \
            tf.reduce_mean(
                (self.f(w_E_bar, u_E_bar, ub_E_bar) + self.b * u_E_bar -
                 self.f(x[:, None, :], u_x[:, None, :], ub_x[:, None, :]) - self.b * u_x[:, None, :]) *
                (np.e ** (-E_bar * (self.b - self.theta_bar))) *
                tf.sqrt(E_bar) *
                (w_E_bar - x[:, None, :]) / E_bar,
                axis=1
            )

        return [phi_0, phi_bar_var_red]

    def label(self, X, model, training):
        E = self.sampleE(len(X) * self.M).reshape(len(X), self.M, 1)
        Eb = self.sampleEb(len(X) * self.M).reshape(len(X), self.M, 1)
        W = np.random.randn(len(X), self.M, self.d)

        W_E = X[:, None, :] + np.sqrt(E) * W
        W_Eb = X[:, None, :] + np.sqrt(Eb) * W

        stacked_W_E = tf.concat([W_E[i, :, :] for i in range(len(X))], axis=0)
        stacked_W_Eb = tf.concat([W_Eb[i, :, :] for i in range(len(X))], axis=0)

        prev = tf.reshape(model(stacked_W_E, training=training), shape=[len(X), self.M, self.d + 1])
        u_E = prev[:, :, :1]
        ub_E = prev[:, :, 1:]

        prev = tf.reshape(model(stacked_W_Eb, training=training), shape=[len(X), self.M, self.d + 1])
        u_Eb = prev[:, :, :1]
        ub_Eb = prev[:, :, 1:]

        prev_X = model(X, training=training)
        u_x = prev_X[:, :1]
        ub_x = prev_X[:, 1:]

        label_data = self.phi(E, Eb, X, W_E, W_Eb, u_E, ub_E, u_Eb, ub_Eb, u_x, ub_x)
        return tf.concat([label_data[0], label_data[1]], axis=1)

    def loss_fn(self, y_label, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(y_label - y_pred), axis=0, keepdims=True)))

    def _build_models(self):
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                axis=1,
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(3)
        ]

        self.NN = tf.keras.Sequential([
            layers.Dense(20 + self.d, input_shape=(self.d,), activation=self.activation),
            self.bn_layers[0],
            layers.Dense(20 + self.d, activation=self.activation),
            self.bn_layers[1],
            layers.Dense(1 + self.d, activation=None)
        ])

        self.model = NN_model(self, self.NN)

    def main(self):
        self.train_history = self.model.train(training=True)
        self.postprocess()

    def postprocess(self):
        start_time = time.time()
        if self.d == 1:
            x_axis_0 = np.linspace(-3.0, 3.0, self.Ntilde).reshape(-1, 1)
            predict = self.model.predict(x_axis_0)

            plt.figure(figsize=(20, 8), dpi=75)
            plt.subplot(121)
            plt.plot(x_axis_0, predict[:, 0], 'x', label='predicted u')
            plt.subplot(122)
            plt.plot(x_axis_0, predict[:, 1], 'x', label='predicted u_bar')

            x_axis = np.linspace(-3.0, 3.0, 10 * self.Ntilde + 1).reshape(-1, 1)
            plt.subplot(121)
            plt.plot(x_axis, self.an_u(x_axis), color='green', label='analytical u')
            plt.legend()
            plt.subplot(122)
            plt.plot(x_axis, self.an_ub(x_axis), color='blue', label='analytical u_bar')
            plt.legend()

            np.save('Numerical_experiments/error_plots/Direct_scheme/u_Kz_{}_{}d.npy'.format(self.K_z, self.d), predict[:, 0])
            np.save('Numerical_experiments/error_plots/Direct_scheme/ub_Kz_{}_{}d.npy'.format(self.K_z, self.d), predict[:, 1])

            x_axis_err = self.sampleX(num_sample=self.M_err)
            predict_err = self.model.predict(x_axis_err)
            self.u_err = np.abs(predict_err[:, :1] - self.an_u(x_axis_err)).reshape(self.M_err)
            self.ub_err = np.sqrt(np.mean((predict_err[:, 1:] - self.an_ub(x_axis_err)) ** 2, axis=-1))
            np.save('Numerical_experiments/error_plots/Direct_scheme/u_err_Kz_{}_{}d.npy'.format(self.K_z, self.d), self.u_err)
            np.save('Numerical_experiments/error_plots/Direct_scheme/ub_err_Kz_{}_{}d.npy'.format(self.K_z, self.d), self.ub_err)

            print('1D: ave u_err, ub_err:', np.mean(self.u_err), np.mean(self.ub_err))
            print('1D: max u_err, ub_err:', np.max(self.u_err), np.max(self.ub_err))
            plt.show()

        elif self.d == 2:
            x_axis_0 = np.linspace(-3.0, 3.0, self.Ntilde)
            x_axis_1 = np.linspace(-3.0, 3.0, self.Ntilde)
            x_grid_eval_0, x_grid_eval_1 = np.meshgrid(x_axis_0, x_axis_1)
            x_eval_points = np.array([[[x_axis_0[i], x_axis_1[j]] for j in range(self.Ntilde)] for i in range(self.Ntilde)]).reshape(-1, 2)

            predict = self.model.predict(x_eval_points)
            fig = plt.figure(figsize=(24, 7), dpi=75)
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')

            ax1.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 0].reshape(self.Ntilde, self.Ntilde), label="Predicted u(x)")
            ax1.plot_wireframe(x_grid_eval_0, x_grid_eval_1, tf.reshape(self.an_u(x_eval_points), shape=(self.Ntilde, self.Ntilde)),
                            color='black',
                            label="Analytical u(x)")
            ax1.set_title(r"$u(x_1, x_2)$")

            ax2.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 1].reshape(self.Ntilde, self.Ntilde), color='red', label="Predicted u(x)")
            ax2.plot_wireframe(x_grid_eval_0, x_grid_eval_1, self.an_ub(x_eval_points)[:, 0].reshape(self.Ntilde, self.Ntilde),
                            color='gray',
                            label="Analytical u(x)")
            ax2.set_title(r"$\bar{u}^1(x_1, x_2)$")

            ax3.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 2].reshape(self.Ntilde, self.Ntilde), color='red', label="Predicted u(x)")
            ax3.plot_wireframe(x_grid_eval_0, x_grid_eval_1, self.an_ub(x_eval_points)[:, 1].reshape(self.Ntilde, self.Ntilde),
                            color='gray',
                            label="Analytical u(x)")
            ax3.set_title(r"$\bar{u}^2(x_1, x_2)$")
            plt.show()

        else:
            X_test = self.sampleX(sig=0.8 * self.sig_X, num_sample=self.M)
            Y_test = self.model.predict(X_test)

            u_pred = Y_test[:, :1]
            ub_pred = Y_test[:, 1:]

            an_u_test = self.an_u(X_test)
            an_ub_test = self.an_ub(X_test)

            L2_err_u = np.sqrt(np.mean(np.mean((u_pred - an_u_test) ** 2, axis=-1)))
            L_inf_err_u = np.max(np.mean(np.abs(u_pred - an_u_test), axis=-1))
            arg_L_inf_err_u = X_test[np.argmax(np.mean(np.abs(u_pred - an_u_test), axis=-1))]

            L2_err_ub = np.sqrt(np.mean(np.mean((ub_pred - an_ub_test) ** 2, axis=-1)))
            L_inf_err_ub = np.max(np.mean(np.abs(ub_pred - an_ub_test), axis=-1))
            arg_L_inf_err_ub = X_test[np.argmax(np.mean(np.abs(ub_pred - an_ub_test), axis=-1))]

            print('high dim errors u:', L2_err_u, L_inf_err_u, arg_L_inf_err_u)
            print('high dim errors ub:', L2_err_ub, L_inf_err_ub, arg_L_inf_err_ub)

        elapsed_time = time.time() - start_time
        print('Time elapsed', elapsed_time)


class NN_model(tf.keras.Model):
    def __init__(self, solver, base_model):
        super().__init__()
        self.solver = solver
        self.model = base_model
        self.num_steps = solver.n_steps

        boundaries, values = self.lr_schedule()
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-7)

    def lr_schedule(self):
        n_divisions = self.solver.n_decays
        lr_boundaries = np.linspace(0, self.num_steps, n_divisions + 1, dtype=int)[1:].tolist()
        lr_values = np.array([self.solver.initial_learning_rate * (self.solver.lr_decay ** i) for i in range(n_divisions)])
        lr_values = np.append(lr_values, 0.4 * lr_values[-1]).tolist()
        print('LR boundaries:', lr_boundaries)
        print('LR values:', lr_values)
        return lr_boundaries, lr_values

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def loss_fn(self, inputs, y_label, training):
        y_pred = self.model(inputs, training=training)
        return tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(y_label - y_pred), axis=0, keepdims=True)))

    def grad(self, inputs, y_label, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, y_label, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data, y_label):
        grad = self.grad(train_data, y_label, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

    def train(self, training=True):
        solver = self.solver
        valid_data = solver.sampleX(num_sample=solver.Mx)
        training_history = []
        start_time = time.time()

        for step in range(self.num_steps):
            if step % solver.update_frequency == 0:
                print('step:', step)
                inputs = solver.sampleX(num_sample=solver.batch_size)
                y_label_out = solver.label(valid_data, self, training=False)
                loss = self.loss_fn(valid_data, y_label_out, training=False).numpy()
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, elapsed_time])
                logging.info('step: %5u,   loss: %.4e,   elapsed time: %3u' % (step, loss, elapsed_time))
                y_label = solver.label(inputs, self, training)
            self.train_step(inputs, y_label)

        return training_history


if __name__ == '__main__':
    solver = NNDirecSolver(d=2, K_z=1.0)
    solver.main()
