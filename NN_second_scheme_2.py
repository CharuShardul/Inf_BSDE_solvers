import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle as pickle
import tensorflow as tf
#from absl import flags
#from absl import logging as absl_logging
import tensorflow.keras.layers as layers
import time
import logging
from datetime import datetime
#from tensorflow.python.ops.numpy_ops import np_config

#np_config.enable_numpy_behavior()

#logging.basicConfig(filename='NN_second_scheme.log', level=logging.INFO)
logging.basicConfig(level=logging.INFO)
tf.keras.backend.set_floatx('float64')
update_frequency = 100
#absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
#absl_logging.set_verbosity('info')

time_now = datetime.now()
logging.info('Time:{}'.format(time_now))


# Parameter initialization
d = 2               # dimension X and BM
dp = 1              # dimension of Y
#num_pic = 10        # number of Picard iterations
Mx = 512            # number of samples of x for logging the loss values while training
M = 3000             # number of samples of E, Eb, W, etc. for each x
M_err = 1000        # number of samples used for error computation
Ntilde = 21         # number of points in each axis of the grid (used for plotting and evaluating errors) = 2n+1
activation = 'relu'
initial_learning_rate = 5e-4
lr_decay = 0.6
n_decays = 10
n_steps = 3000
batch_size = 512    # mini-batch size used while training
epochs = 500

a = 2.0             # >0.5
b = 2.0             # >1.0
theta = 1.5
theta_bar = 1.5
sig_X = 2.0

c = 2.0             # Monotonicity constant mu= (c-1) = 2
K_z = 1.0           # z-Lipschitz constant K_{f,z} = 1


#num_iter = 2000     # number of SGD iterations
#log_freq = 100      # logging frequency within SGD iterations

def coordpoint(x):
    result = np.zeros(d)
    for i in range(d):
        q, r = divmod(x, Ntilde)
        result[i] = r
        x = q
    return result

# Analytical solution for u
def an_u(x):
    # input shape = (num_sample, dim_x); output shape = (num_sample, 1)
    return (1 / d) * tf.reduce_sum(tf.math.atan(x), axis=-1, keepdims=True)


def an_ub(x):
    # input shape = (num_sample, dim_x); output shape = (num_sample, dim_x)
    return (1 / d) * (1 / (1 + x**2))

def an_Delta_u(x):
    # input shape = (num_sample, dim_x); output shape = (num_sample, 1)
    # Returns the value of the Laplacian of u at x (numpy array of coordinates)
    return (-2 / d) * tf.reduce_sum(x / ((1 + x**2)**2), axis=-1, keepdims=True)

'''

alpha = 1.0
beta = 1.0
gamma = 0.0

def an_u(x):
    # input shape = (num_sample, dim_x); output shape = (num_sample, 1)
    return alpha * tf.reduce_sum(x, axis=-1, keepdims=True) + beta * tf.reduce_sum(x**2, axis=-1, keepdims=True) + \
            gamma * tf.reduce_sum(x**3, axis=-1, keepdims=True)

def an_ub(x):
    # input shape = (num_sample, dim_x); output shape = (num_sample, dim_x)
    return alpha * tf.ones(shape=x.shape) + 2 * beta * x + 3 * gamma * x**2

def an_Delta_u(x):
    # input shape = (num_sample, dim_x); output shape = (num_sample, 1)
    # Returns the value of the Laplacian of u at x (numpy array of coordinates)
    return 2 * beta * d + 6 * gamma * tf.reduce_sum(x, axis=-1, keepdims=True)'''

# Choosing the f_0 component of the generator and defining the resulting generator

def f_0(x, y, z):
    # input shape = ((num_sample, dim_x), (num_sample, dim_y=1), (num_sample, dim_z=dim_x))
    # output shape = (num_sample, dim_y=1)
    return -c*y + tf.cos(y + tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))) \
           + K_z * np.sin(np.sqrt(np.sum(np.square(z), axis=-1, keepdims=True)))
           #+ K_z*np.sin(np.sqrt(np.sum(np.square(z), axis=-1, keepdims=True)))
    #return -c * y + np.cos(y + x)
    #return -c * y + K_z * np.sin(np.sqrt(np.sum(np.square(z))))
    #return -c * y + np.cos(y + np.sqrt(np.sum(np.square(x))) + 1) + K_z * np.sin(np.sqrt(np.sum(np.square(z))))


def f(x, y, z):
    # input shape = ((num_sample, dim_x), (num_sample, dim_y=1), (num_sample, dim_z=dim_x))
    # output shape = (num_sample, dim_y=1)
    return f_0(x, y, z) - 0.5*an_Delta_u(x) - f_0(x, an_u(x), an_ub(x))


# M samples of the exponential and gamma distributed times and Brownian motion.

def sampleE(num_sample=Mx):
    # output shape = (num_sample, 1)
    # M samples of exponentially distributed time
    E = np.random.exponential(scale=1/theta, size=[num_sample, 1])
    return E


def sampleEb(num_sample=Mx):
    # output shape = (num_sample, 1)
    # M samples of gamma distributed time
    Ebar = np.random.gamma(shape=0.5, scale=1/theta_bar, size=[num_sample, 1])
    return Ebar


def sampleX(sig=sig_X, num_sample=Mx):
    # output shape = (num_sample, dim_x)
    return np.random.normal(loc=0.0, scale=sig, size=[num_sample, d])


def phi(E, E_bar, x, w_E, w_E_bar, u_E, ub_E, u_E_bar, ub_E_bar, u_x, ub_x):
    # input has all components with their individual samples

    #print("shapes: w ", w_E.shape, w_E_bar.shape)
    #print("shapes: u ", u_E.shape, u_E_bar.shape, u_x.shape)
    #print("shapes: ub ", ub_E.shape, ub_E_bar.shape, ub_x.shape)
    #print("shapes: an_u, an_ub ", u_x.shape, ub_x.shape)

    phi_0 = (1 / theta) * tf.reduce_mean((f(w_E, u_E, ub_E) + a * u_E) * (np.e ** (-E * (a - theta))), axis=1)

    #phi_0 = an_u(x)
    #print("type:", type(u_E), type(ub_E), type(u_E_bar), type(ub_E_bar))

    #phi_0 = (1 / theta) * np.mean((f(w_E, an_u(w_E), an_ub(w_E)) + a * an_u(w_E)) * (np.e ** (-E * (a - theta))), axis=1)

    '''phi_0 = (1/theta) * np.mean(np.array([(f(w_E[:, i, :], u_E[:, i, :], ub_E[:, i, :]) + a * u_E[:, i, :]) *
                                                (np.e ** (-E[:, i, :] * (a - theta)))
                                                for i in range(M)]), axis=0, keepdims=False)'''

    '''phi_0 = (1 / theta) * np.mean(
        np.array([(f(w_E[:, i, :], u_E[:, i, :], an_ub(x)) + a * u_E[:, i, :]) *
                  (np.e ** (-E[:, i, :] * (a - theta)))
                  for i in range(M)]), axis=0, keepdims=False)'''
    #print("u:", u_E, "w_E:", w_E, "f:", f(w_E[:, 1, :], u_E[:, 1, :], an_ub(x)), "exp:", (np.e ** (-E[:, 1, :] * (a - theta))))
    #print("phi:", phi_0, phi_0.shape)

    #print("dtype:", type(tf.sqrt(np.pi/theta_bar)))

    phi_bar_var_red = np.sqrt(np.pi / theta_bar) * \
                      tf.reduce_mean((f(w_E_bar, u_E_bar, ub_E_bar) + b * u_E_bar -
                                      f(x[:, None, :], u_x[:, None, :], ub_x[:, None, :]) - b * u_x[:, None, :]) *
                                     (np.e ** (-E_bar * (b - theta_bar))) *
                                     tf.sqrt(E_bar) *
                                     (w_E_bar - x[:, None, :]) / E_bar, axis=1)

    '''phi_bar_var_red = tf.sqrt(tf.pi / theta_bar).astype('float32') * \
                      tf.reduce_mean((f(w_E_bar, u_E_bar, ub_E_bar) + b * u_E_bar -
                              f(x[:, None, :], u_x[:, None, :], ub_x[:, None, :]) - b * u_x[:, None, :]) *
                              (tf.e ** (-E_bar * (b - theta_bar))) *
                              tf.sqrt(E_bar).astype('float32') *
                              tf.cast((w_E_bar - x[:, None, :]) / E_bar, tf.float32), axis=1)'''

    # an_u(w_E_bar)
    '''phi_bar_var_red = np.sqrt(np.pi / theta_bar).astype('float32') * \
                      np.mean((f(w_E_bar, an_u(w_E_bar), ub_E_bar) + b * u_E_bar -
                              f(x[:, None, :], u_x[:, None, :], ub_x[:, None, :]) - b * u_x[:, None, :]) *
                              (np.e ** (-E_bar * (b - theta_bar))) *
                              np.sqrt(E_bar).astype('float32') *
                              tf.cast((w_E_bar - x[:, None, :]) / E_bar, tf.float32), axis=1)'''


    #phi_bar_var_red = an_ub(x)

    return [phi_0, phi_bar_var_red]


def label(X, model, training):
    E = sampleE(len(X)*M).reshape(len(X), M, 1)
    Eb = sampleEb(len(X)*M).reshape(len(X), M, 1)
    #print("E shape", E.shape)
    #print(len(X))
    #print("shape x", X.shape, X[0, None].shape)
    W = np.random.randn(len(X), M, d)
    W_E = X[:, None, :] + np.sqrt(E) * W
    #W_E = X + tf.sqrt(E) * W
    #print("W_E test:", X[0], W_E[0], np.mean(W_E[0]))
    W_Eb = X[:, None, :] + np.sqrt(Eb) * W
    #print("test W_E:", type(W_E), W_E.shape, W_E[0])
    stacked_W_E = tf.concat([W_E[i, :, :] for i in range(len(X))], axis=0)
    stacked_W_Eb = tf.concat([W_Eb[i, :, :] for i in range(len(X))], axis=0)
    #print("stack test:", type(stacked_W_E), stacked_W_E.shape)

    # stacking samples so that samples 1 to M correspond to X[0] and so on
    prev = tf.reshape(model(stacked_W_E, training=training), shape=[len(X), M, d+1])
    #prev = np.array([model(W_E[:, i, :]) for i in range(M)])
    u_E = prev[:, :, :1]
    #u_E = np.array([an_u(W_E[i]) for i in range(len(W))])
    ub_E = prev[:, :, 1:]
    #u_Eb = np.array([an_u(W_Eb[i]) for i in range(len(W))])
    #print(u_E, u_Eb)

    # stacking samples so that samples 1 to M correspond to X[0] and so on
    prev = tf.reshape(model(stacked_W_Eb, training=training), shape=[len(X), M, d+1])
    u_Eb = prev[:, :, :1]

    ub_Eb = prev[:, :, 1:]

    prev_X = model(X, training=training)
    #u_x = prev_X[:, None, :1]
    #ub_x = prev_X[:, None, 1:]
    u_x = prev_X[:, :1]
    ub_x = prev_X[:, 1:]
    #print("u:", u_x, ub_x, u_x.shape, ub_x.shape)

    label = phi(E, Eb, X, W_E, W_Eb, u_E, ub_E, u_Eb, ub_Eb, u_x, ub_x)

    #label_0 = np.array([label[i][0] for i in range(len(X))])
    label_0 = label[0]

    #label_1 = np.array([phi(E[i], Eb[i], X[i], W_E[i], W_Eb[i], u_E[i], ub_E[i], u_Eb[i], ub_Eb[i])[1]
    #                    for i in range(len(X))])
    label_1 = label[1]

    print("label shapes:", label_0.shape, label_1.shape)
    #print("type:", type(tf.concat([label_0, label_1], axis=1)))
    return tf.concat([label_0, label_1], axis=1)


bn_layers = [
            tf.keras.layers.BatchNormalization(
                axis=1,
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(3)]


NN = tf.keras.Sequential([layers.Dense(20 + d, input_shape=(d,), activation=activation),#activation=tf.keras.layers.LeakyReLu(alpha=0.01)),
                          bn_layers[0],
                          layers.Dense(20 + d, activation=activation),
                          #layers.LeakyReLu(alpha=0.01)),
                          bn_layers[1],
                          #layers.Dense(20 + d, activation=activation),
                          #layers.LeakyReLu(alpha=0.01)),
                          #bn_layers[2],
                          layers.Dense(1 + d, activation=None)])


class NN_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = NN #tf.keras.Sequential([layers.Dense()])
        self.num_steps = n_steps #int(epochs*Mx/batch_size)
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    self.lr_schedule()[0], self.lr_schedule()[1])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-7)

    def lr_schedule(self):
        n_divisions = n_decays
        lr_boundaries = np.linspace(0, self.num_steps, n_divisions + 1,
                                    dtype=int)[1:].tolist()
        lr_values = np.array([])
        temp = np.array([initial_learning_rate*(lr_decay**i) for i in range(n_divisions)])
        lr_values = np.append(lr_values, temp)
        lr_values = np.append(lr_values, 0.4*lr_values[-1]).tolist()
        print("LR boundaries:", lr_boundaries)
        print("LR values:", lr_values)
        return lr_boundaries, lr_values

    def call(self, inputs):
        return self.model(inputs)

    def loss_fn(self, inputs, y_label, training):
        y_pred = self.model(inputs, training=training)
        loss = tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(y_label - y_pred), axis=0, keepdims=True)))
        #print(type(y_label))
        return loss

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

    def train(self, training):
        valid_data = sampleX(num_sample=Mx)
        training_history = []
        start_time = time.time()
        for step in range(self.num_steps):
            if step % update_frequency == 0:
                print("step:", step)
                inputs = sampleX(num_sample=batch_size)
                y_label_out = label(valid_data, self.model, training=False)
                loss = self.loss_fn(valid_data, y_label_out, training=False).numpy()
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, elapsed_time])
                logging.info("step: %5u,   loss: %.4e,   elapsed time: %3u"
                             % (step, loss, elapsed_time))
                y_label = label(inputs, self.model, training)
            self.train_step(inputs, y_label)

        return training_history


model = NN_model()
train_history = model.train(training=True)

start_time = time.time()

if d == 1:
    plt.figure(figsize=(20, 8), dpi=75)
    plt.subplot(121)
    x_axis_0 = np.linspace(-3.0, 3.0, Ntilde).reshape(-1, 1)
    predict = model.predict(x_axis_0)
    plt.plot(x_axis_0, predict[:, 0], 'x', label='predicted u')
    plt.subplot(122)
    plt.plot(x_axis_0, predict[:, 1], 'x', label='predicted u_bar')

    plt.subplot(121)
    x_axis = np.linspace(-3.0, 3.0, 10*Ntilde+1).reshape(-1, 1)
    plt.plot(x_axis, an_u(x_axis), color='green', label='analytical u')
    plt.legend()
    plt.subplot(122)
    plt.plot(x_axis, an_ub(x_axis), color='blue', label='analytical u_bar')
    plt.legend()
    np.save('Numerical_experiments/error_plots/Direct_scheme/u_Kz_{}_{}d.npy'.format(K_z, d), predict[:, 0])
    np.save('Numerical_experiments/error_plots/Direct_scheme/ub_Kz_{}_{}d.npy'.format(K_z, d), predict[:, 1])
    print('shape predict 1= ', predict[:, 0].shape, 'shape predict 1= ', predict[:, 1].shape)
    plt.savefig("Numerical_experiments/NN_second_scheme/test_Kz_{}.png".format(K_z))

    x_axis_err = sampleX(num_sample=M_err)
    predict_err = model.predict(x_axis_err)
    u_err = np.abs(predict_err[:, :1] - an_u(x_axis_err)).reshape(M_err)
    print("shape 1= ", u_err.shape)
    ub_err = np.sqrt(np.mean((predict_err[:, 1:] - an_ub(x_axis_err)) ** 2, axis=-1))
    print("shape 2= ", ub_err.shape)
    np.save('Numerical_experiments/error_plots/Direct_scheme/u_err_Kz_{}_{}d.npy'.format(K_z, d), u_err)
    np.save('Numerical_experiments/error_plots/Direct_scheme/ub_err_Kz_{}_{}d.npy'.format(K_z, d), ub_err)
    print("ave", np.mean(u_err), np.mean(ub_err))
    print("max", np.max(u_err), np.max(ub_err))

    plt.show()



elif d == 2:

    x_axis_0 = np.linspace(-3.0, 3.0, Ntilde)
    x_axis_1 = np.linspace(-3.0, 3.0, Ntilde)
    x_grid_eval_0, x_grid_eval_1 = np.meshgrid(x_axis_0, x_axis_1)

    # print("x grid eval shape:", x_grid_eval_0.shape)
    # x_eval_points = np.transpose(np.array([x_grid_eval_0, x_grid_eval_1]).reshape(2, -1))

    # list of points from the 2-D grid
    x_eval_points = np.array([[[x_axis_0[i], x_axis_1[j]] for j in range(Ntilde)] for i in range(Ntilde)]).reshape(-1,
                                                                                                                   2)

    predict = model.predict(x_eval_points)

    # print("predict shape:", predict.shape)

    fig = plt.figure(figsize=(24, 14), dpi=75)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 0].reshape(Ntilde, Ntilde), label="Predicted u(x)")
    ax.plot_wireframe(x_grid_eval_0, x_grid_eval_1, tf.reshape(an_u(x_eval_points), shape=(Ntilde, Ntilde)), color='red',
                      label="Analytical u(x)")
    ax.set_title("u(x_1, x_2)")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # print("ub_err shape:", np.sqrt(np.mean((predict[:, 1:] - an_ub(x_eval_points))**2, axis=-1)).shape)
    # print("predict ub shape:", predict[:, 1:].shape, "an_ub shape:", an_ub(x_eval_points).shape)
    ub_err = np.sqrt(np.mean((predict[:, 1:] - an_ub(x_eval_points)) ** 2, axis=-1)).reshape(Ntilde, Ntilde)
    ax.plot_surface(x_grid_eval_0, x_grid_eval_1, ub_err, label="l^2 errors for ub(x)")
    ax.set_title("l^2 error for u_b(x_1, x_2)")

    fig.savefig('Numerical_experiments/NN_second_scheme/u_and_l2_err.png')

    fig = plt.figure(figsize=(24, 14), dpi=75)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 1].reshape(Ntilde, Ntilde), label="Predicted ub^1(x)")
    ax.plot_wireframe(x_grid_eval_0, x_grid_eval_1, an_ub(x_eval_points)[:, 0].reshape(Ntilde, Ntilde),
                      color='green',
                      label="Analytical ub^1(x)")
    ax.set_title("u_b^1(x_1, x_2)")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 2].reshape(Ntilde, Ntilde), label="")
    ax.plot_wireframe(x_grid_eval_0, x_grid_eval_1, an_ub(x_eval_points)[:, 1].reshape(Ntilde, Ntilde),
                      color='black',
                      label="Analytical ub^2(x)")
    ax.set_title("u_b^1(x_1, x_2)")

    fig.savefig('Numerical_experiments/NN_second_scheme/ub_1_and_ub_2.png')

    fig = plt.figure(figsize=(24, 7), dpi=75)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 0].reshape(Ntilde, Ntilde), label="Predicted u(x)")
    ax1.plot_wireframe(x_grid_eval_0, x_grid_eval_1, tf.reshape(an_u(x_eval_points), shape=(Ntilde, Ntilde)),
                      color='black',
                      label="Analytical u(x)")
    ax1.set_title(r"$u(x_1, x_2)$")

    ax2.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 1].reshape(Ntilde, Ntilde), color='red', label="Predicted u(x)")
    ax2.plot_wireframe(x_grid_eval_0, x_grid_eval_1, an_ub(x_eval_points)[:, 0].reshape(Ntilde, Ntilde),
                       color='gray',
                       label="Analytical u(x)")
    ax2.set_title(r"$\bar{u}^1(x_1, x_2)$")

    ax3.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 2].reshape(Ntilde, Ntilde), color='red', label="Predicted u(x)")
    ax3.plot_wireframe(x_grid_eval_0, x_grid_eval_1, an_ub(x_eval_points)[:, 1].reshape(Ntilde, Ntilde),
                       color='gray',
                       label="Analytical u(x)")
    ax3.set_title(r"$\bar{u}^2(x_1, x_2)$")
    plt.show()
    #fig.savefig('Numerical_experiments/NN_dir_2d.png', bbox_inches='tight')


elif d>=3:

    X_test = sampleX(sig=0.8 * sig_X, num_sample=M)
    Y_test = model.predict(X_test)

    u_pred = Y_test[:, :1]
    ub_pred = Y_test[:, 1:]

    an_u_test = an_u(X_test)
    an_ub_test = an_ub(X_test)

    L2_err_u = np.sqrt(np.mean(np.mean((u_pred - an_u_test) ** 2, axis=-1)))
    L_inf_err_u = np.max(np.mean(np.abs(u_pred - an_u_test), axis=-1))
    arg_L_inf_err_u = X_test[np.argmax(np.mean(np.abs(u_pred - an_u_test), axis=-1))]
    print("shapes:", L2_err_u.shape, L_inf_err_u.shape, arg_L_inf_err_u.shape)

    print("mean l^2 error for u:", L2_err_u, "\t L^inf error for u:", L_inf_err_u, "\t the error is maximum at:",
          arg_L_inf_err_u)

    L2_err_ub = np.sqrt(np.mean(np.mean((ub_pred - an_ub_test) ** 2, axis=-1)))
    L_inf_err_ub = np.max(np.mean(np.abs(ub_pred - an_ub_test), axis=-1))
    arg_L_inf_err_ub = X_test[np.argmax(np.mean(np.abs(ub_pred - an_ub_test), axis=-1))]

    print("mean L^2 error for u_bar:", L2_err_ub, "\t L^inf error for u_bar:", L_inf_err_ub,
          "\t the error is maximum at:", arg_L_inf_err_ub)

    errors = (L2_err_u, L_inf_err_u, arg_L_inf_err_u, L2_err_ub, L_inf_err_ub, arg_L_inf_err_ub)



    '''fig = plt.figure(figsize=(24, 14), dpi=75)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(np.arange(1, num_pic + 1, 1), [errors[i][0] for i in range(num_pic)])
    ax1.set_title("L^2_mu errors for u", fontweight='bold')
    ax1.set_xlabel("Picard Iterations")
    ax1.set_ylabel("L^2_mu error")

    ax2.plot(np.arange(1, num_pic + 1, 1), [errors[i][1] for i in range(num_pic)])
    ax2.set_title("L^inf_mu errors for u", fontweight='bold')
    ax2.set_xlabel("Picard Iterations")
    ax2.set_ylabel("L^inf_mu error")

    fig.savefig("Numerical_experiments/NN_Picard_multi_2/Pic_errors_u_dim_{}.png".format(d))

    fig = plt.figure(figsize=(24, 16), dpi=75)
    ax3 = fig.add_subplot(1, 2, 1)
    ax4 = fig.add_subplot(1, 2, 2)

    ax3.plot(np.arange(1, num_pic + 1, 1), [errors[i][3] for i in range(num_pic)])
    ax3.set_title("L^2_mu errors for ub", fontweight='bold')
    ax3.set_xlabel("Picard Iterations")
    ax3.set_ylabel("L^2_mu error")

    ax4.plot(np.arange(1, num_pic + 1, 1), [errors[i][4] for i in range(num_pic)])
    ax4.set_title("L^inf_mu errors for ub", fontweight='bold')
    ax4.set_xlabel("Picard Iterations")
    ax4.set_ylabel("L^inf_mu error")'''

    plt.show()
    #fig.savefig("Numerical_experiments/NN_Picard_multi_2/Pic_errors_ub_dim_{}.png".format(d))



'''logging.info('Parameters in this run:'
             '\t num_pic = {},'
             '\t M = {},'
             '\t K_z = {},'
             '\t c = {},\n'
             '\t a = {},'
             '\t b = {},'
             '\t sig_X = {},'
             '\t theta = {},'
             '\t theta_bar = {},\n'
             '\t batch_size={},'
             '\t epochs={}'
             '\t activation={}'
             '\t learning_rate={}'
             .format(num_pic, M, K_z, c, a, b, sig_X, theta, theta_bar, batch_size, epochs, activation, learning_rate))'''