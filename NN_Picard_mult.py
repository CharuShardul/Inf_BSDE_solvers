import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
import time
import logging
from datetime import datetime
#from tensorflow.python.ops.numpy_ops import np_config

#np_config.enable_numpy_behavior()

logging.basicConfig(filename='NN_Picard_mult.log', level=logging.INFO)

time_now = datetime.now()
logging.info('Time:{}'.format(time_now))

# Parameter initialization
d = 1               # dimension X and BM
dp = 1              # dimension of Y
num_pic = 5         # number of Picard iterations
M = 30000           # number of samples
M_err = 1000        # number of samples for error computations
Ntilde = 21         # number of points in each axis of the grid (used for plotting and evaluating errors) = 2n+1
batch_size = 500
epochs = 300
activation = 'relu'
initial_learning_rate = 5e-4
lr_decay = 0.97
decay_steps = 1000
#num_iter = 2000     # number of SGD iterations
#log_freq = 100      # logging frequency within SGD iterations

a = 2.0             # >0.5
b = 2.0             # >1.0
theta = 1.5
theta_bar = 1.5
sig_X = 2.0

c = 2.0             # Monotonicity constant mu= (c-1) = 2
K_z = 1.0           # z-Lipschitz constant K_{f,z} = 1
print("K_z=", K_z)


# Function to convert a 1-d encoding of position vector into d-dimensions
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
    return (1 / d) * np.sum(np.arctan(x), axis=-1, keepdims=True)


def an_ub(x):
    # input shape = (num_sample, dim_x); output shape = (num_sample, dim_x)
    return (1 / d) * (1 / (1 + x**2))

def an_Delta_u(x):
    # input shape = (num_sample, dim_x); output shape = (num_sample, 1)
    # Returns the value of the Laplacian of u at x (numpy array of coordinates)
    return (-2 / d) * np.sum(x / ((1 + x**2)**2), axis=-1, keepdims=True)

'''alpha = 1.0
beta = 1.0
gamma = 0.0

def an_u(x):
    # input shape = (num_sample, dim_x); output shape = (num_sample, 1)
    return alpha * np.sum(x, axis=-1, keepdims=True) + beta * np.sum(x**2, axis=-1, keepdims=True) + \
            gamma * np.sum(x**3, axis=-1, keepdims=True)

def an_ub(x):
    # input shape = (num_sample, dim_x); output shape = (num_sample, dim_x)
    return alpha * np.ones(shape=x.shape) + 2 * beta * x + 3 * gamma * x**2

def an_Delta_u(x):
    # input shape = (num_sample, dim_x); output shape = (num_sample, 1)
    # Returns the value of the Laplacian of u at x (numpy array of coordinates)
    return 2 * beta * d + 6 * gamma * np.sum(x, axis=-1, keepdims=True)'''


# Choosing the f_0 component of the generator and defining the resulting generator

def f_0(x, y, z):
    # input shape = ((num_sample, dim_x), (num_sample, dim_y=1), (num_sample, dim_z=dim_x))
    # output shape = (num_sample, dim_y=1)
    return -c*y + np.cos(y + np.sqrt(np.sum(np.square(x), axis=-1, keepdims=True))) \
           + K_z * np.sin(np.sqrt(np.sum(np.square(z), axis=-1, keepdims=True)))
           #+ K_z * np.sqrt(np.sum(np.square(z), axis=-1, keepdims=True))


    #return -c * y + np.cos(y + x)
    #return -c * y + K_z * np.sin(np.sqrt(np.sum(np.square(z))))
    #return -c * y + np.cos(y + np.sqrt(np.sum(np.square(x))) + 1) + K_z * np.sin(np.sqrt(np.sum(np.square(z))))


def f(x, y, z):
    # input shape = ((num_sample, dim_x), (num_sample, dim_y=1), (num_sample, dim_z=dim_x))
    # output shape = (num_sample, dim_y=1)
    return f_0(x, y, z) - 0.5*an_Delta_u(x) - f_0(x, an_u(x), an_ub(x))


# M samples of the exponential and gamma distributed times and Brownian motion.

def sampleE(num_sample=M):
    # output shape = (num_sample, 1)
    # M samples of exponentially distributed time
    E = np.random.exponential(scale=1/theta, size=[num_sample, 1])
    return E


def sampleEb(num_sample=M):
    # output shape = (num_sample, 1)
    # M samples of gamma distributed time
    Ebar = np.random.gamma(shape=0.5, scale=1/theta_bar, size=[num_sample, 1])
    return Ebar


def sampleX(sig=sig_X, num_sample=M):
    # output shape = (num_sample, dim_x)
    return np.random.normal(loc=0.0, scale=sig, size=[num_sample, d])


def phi(E, E_bar, x, w_E, w_E_bar, u_E, ub_E, u_E_bar, ub_E_bar, u_x, ub_x):
    # input has all components with their individual samples
    phi = (1/theta) * (f(w_E, u_E, ub_E) + a * u_E) * (np.e ** (-E * (a - theta)))
    #print("phi:", phi)
    #print((f(w_E_bar, u_E_bar, ub_E_bar) + b * u_E_bar))
    #print("a", ((w_E_bar - x)/E_bar).astype('float32'))

    '''phi_bar = np.sqrt(np.pi/theta_bar).astype('float32') * \
        (f(w_E_bar, u_E_bar, ub_E_bar) + b * u_E_bar) * \
        (np.e ** (-E_bar * (b - theta_bar))) * \
        np.sqrt(E_bar).astype('float32') * \
        tf.cast((w_E_bar - x)/E_bar, tf.float32)'''

    '''phi_bar_var_red = np.sqrt(np.pi / theta_bar).astype('float32') * \
              (f(w_E_bar, u_E_bar, ub_E_bar) + b * u_E_bar - f(x, an_u(x), an_ub(x)) - b * an_u(x)) * \
              (np.e ** (-E_bar * (b - theta_bar))) * \
              np.sqrt(E_bar).astype('float32') * \
              tf.cast((w_E_bar - x) / E_bar, tf.float32)'''

    phi_bar_var_red = np.sqrt(np.pi / theta_bar).astype('float32') * \
                      (f(w_E_bar, u_E_bar, ub_E_bar) + b * u_E_bar - f(x, u_x, ub_x) - b * u_x) * \
                      (np.e ** (-E_bar * (b - theta_bar))) * \
                      np.sqrt(E_bar).astype('float32') * \
                      tf.cast((w_E_bar - x) / E_bar, tf.float32)

    #phi_bar = (np.sqrt(np.pi / theta_bar).astype('float32') * (f(w_E_bar, an_u(w_E_bar), an_ub(w_E_bar)) + b * an_u(w_E_bar)) * (
    #        np.e ** (-E_bar * (b - theta_bar))) * np.sqrt(E_bar).astype('float32') * tf.cast((w_E_bar - x) / E_bar,
    #                                                                                         tf.float32))

    #phi_bar_var_red = (np.sqrt(np.pi/theta_bar).astype('float32') * (f(x, an_u(x), an_ub(x)) + b * an_u(x)) * (
    #            np.e ** (-E_bar * (b - theta_bar))) * np.sqrt(E_bar).astype('float32') * tf.cast((w_E_bar - x)/E_bar, tf.float32))

    #phi_bar_var_red = (np.sqrt(np.pi / theta_bar).astype('float32') * (f(x, u_x, ub_x) + b * u_x) * (
    #            np.e ** (-E_bar * (b - theta_bar))) * np.sqrt(E_bar).astype('float32') * tf.cast((w_E_bar - x)/E_bar, tf.float32))

    #phi_bar = phi_bar - phi_bar_var_red
    #print("phi_bar:", phi_bar, phi_bar.shape)
    return [phi, phi_bar_var_red]

'''class model_init(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self):
        return 0.0, 0.0'''

def label(X, prev_model):
    # input shape = ((num_sample, dim_X), tensorflow NN)
    # output shape = (num_sample, 1 + dim_X)
    E = sampleE(len(X))
    Eb = sampleEb(len(X))
    #print(len(X))
    #print(X[0])
    W = np.random.randn(len(X), d)
    W_E = X + np.sqrt(E) * (W)
    #print("W_E shape:", W_E.shape)
    W_Eb = X + np.sqrt(Eb) * (W)

    prev = prev_model(W_E)
    u_E = prev[:, :1]
    #u_E = np.array([an_u(W_E[i]) for i in range(len(W))])
    ub_E = prev[:, 1:]
    #u_Eb = np.array([an_u(W_Eb[i]) for i in range(len(W))])
    #print(u_E, u_Eb)

    prev = prev_model(W_Eb)
    u_Eb = prev[:, :1]
    #ub_E = np.array([an_ub(W_E[i]) for i in range(len(W))])
    ub_Eb = prev[:, 1:]
    #ub_Eb = np.array([an_ub(W_Eb[i]) for i in range(len(W))])

    prev_X = prev_model(X)
    u_x = prev_X[:, :1]
    ub_x = prev_X[:, 1:]
    #print("u:", u_x, ub_x, u_x.shape, ub_x.shape)

    #label = np.array([phi(E[i], Eb[i], X[i], W_E[i], W_Eb[i], u_E[i], ub_E[i], u_Eb[i], ub_Eb[i], u_x[i], ub_x[i])
    #                  for i in range(len(X))])

    label = phi(E, Eb, X, W_E, W_Eb, u_E, ub_E, u_Eb, ub_Eb, u_x, ub_x)

    #label_0 = np.array([label[i][0] for i in range(len(X))])
    label_0 = label[0]

    #label_1 = np.array([phi(E[i], Eb[i], X[i], W_E[i], W_Eb[i], u_E[i], ub_E[i], u_Eb[i], ub_Eb[i])[1]
    #                    for i in range(len(X))])
    label_1 = label[1]
    #print(tf.shape(tf.concat([label_0, label_1], axis=1)))

    return tf.concat([label_0, label_1], axis=1)

#model_init.compile()

'''class NN(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, input):
        return tf.keras.Sequential(layers.Dense(20, input_shape=(1,), activation='relu'),
                                   layers.Dense(20, activation='relu'),
                                   layers.Dense(2, activation=None))(input)'''
bn_layers = [
            tf.keras.layers.BatchNormalization(
                axis=1,
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(3)]

#zeros = np.zeros(d+1, dtype='float32')
#model_init = tf.keras.Sequential(layers.Lambda(lambda x: tf.constant([zeros for _ in range(len(x))])))
model_init = tf.keras.Sequential([layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], d+1)))])

NN = tf.keras.Sequential([layers.Dense(20 + d, input_shape=(d,), activation=activation),#activation=tf.keras.layers.LeakyReLu(alpha=0.01)),
                          bn_layers[0],
                          layers.Dense(20 + d, activation=activation),#tf.keras.layers.LeakyReLu(alpha=0.01)),
                          bn_layers[1],
                          #layers.Dense(20 + d, activation=activation),
                          #bn_layers[2],
                          layers.Dense(1 + d, activation=None)])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=lr_decay,
    staircase=True)


def loss_fn(y_label, y_pred):
    # try different weights
    return tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(y_label - y_pred), axis=0, keepdims=True)))

model_0 = model_init
model_0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=loss_fn,
                metrics=['accuracy'])
model_1 = NN
model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=loss_fn,
                metrics=['accuracy'])


# Making the grid and training the NN


start_time = time.time()

if d == 1:
    x_axis = np.linspace(-3.0, 3.0, Ntilde).reshape(-1, 1)
    x_axis_1 = np.linspace(-3.0, 3.0, 10*Ntilde+1).reshape(-1, 1)
    fig = plt.figure(figsize=(12, 5), dpi=75)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("$u(x)$")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title(r"$\bar{u}(x)$")

    yy = an_u(x_axis_1)
    yyb = an_ub(x_axis_1)
    #yy_Delta = np.array([an_Delta_u(x_axis[i]) for i in range(len(x_axis))])
    ax1.plot(x_axis_1, yy, color='black', label=r"Analytical ${u}(x)$")
    ax2.plot(x_axis_1, yyb, color='brown', label=r"Analytical $\bar{u}(x)$")

    for p in range(num_pic):
        print("Picard iteration, p = ", p + 1)

        X_train = sampleX()
        Y_train = label(X_train, model_0)

        model_1.fit(X_train, Y_train,
                    batch_size=batch_size,
                    shuffle=False,
                    epochs=epochs,
                    verbose=1)

        model_0 = model_1

        predict = model_1.predict(x_axis)
        if p % 1 == 0:
            ax1.plot(x_axis, predict[:, 0], 'x', label="Iteration p={}".format(p + 1))
            ax2.plot(x_axis, predict[:, 1], 'x', label="Iteration p={}".format(p + 1))
        # print("predict shape:", predict.shape)

        '''fig = plt.figure(figsize=(24, 14), dpi=75)
        ax = fig.add_subplot()
        ax.plot(x_axis, predict[:, 0], color='red', label="predicted solution p={}".format(p+1))
        ax.plot(x_axis, an_u(x_axis), color='black', label="analytical solution p={}".format(p+1))
        ax.set_title("u(x)")
        ax.legend()

        fig.savefig('Numerical_experiments/NN_Picard_dim1/u_iter_{}.png'.format(p + 1))

        fig = plt.figure(figsize=(24, 14), dpi=75)
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x_axis, predict[:, 1], label="predicted solution p={}".format(p+1))
        ax.plot(x_axis, an_ub(x_axis),
                          color='green',
                          label="analytical solution p={}".format(p + 1))
        ax.set_title("u_b(x)")
        ax.legend()

        ax = fig.add_subplot(1, 2, 2)
        # print("ub_err shape:", np.sqrt(np.mean((predict[:, 1:] - an_ub(x_eval_points))**2, axis=-1)).shape)
        # print("predict ub shape:", predict[:, 1:].shape, "an_ub shape:", an_ub(x_eval_points).shape)
        ub_err = np.sqrt(np.mean((predict[:, 1:] - an_ub(x_axis)) ** 2, axis=-1))
        ax.plot(x_axis, ub_err, label="l^2 error p={}".format(p + 1))
        ax.set_title("l^2 error for u_b(x)")
        ax.legend()'''
    #print("M", predict[:, 0].shape)
    x_axis_err = sampleX(num_sample=M_err)
    predict_err = model_1.predict(x_axis_err)
    u_err = np.abs(predict_err[:, :1] - an_u(x_axis_err)).reshape(M_err)
    print("shape 1= ", u_err.shape)
    ub_err = np.sqrt(np.mean((predict_err[:, 1:] - an_ub(x_axis_err)) ** 2, axis=-1))
    print("shape 2= ", ub_err.shape)
    np.save('Numerical_experiments/error_plots/u_Kz_{}_{}d.npy'.format(K_z, d), u_err)
    np.save('Numerical_experiments/error_plots/ub_Kz_{}_{}d.npy'.format(K_z, d), ub_err)
    print("ave", np.mean(u_err), np.mean(ub_err))
    print("max", np.max(u_err), np.max(ub_err))

    #temp = np.load('Numerical_experiments/NN_Picard_dim1/ub_err_1d.npy')
    #print(temp)

    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$u(x)$")
    ax1.legend(loc='upper left')
    ax2.set_xlabel("$x$")
    ax2.set_ylabel(r"$\bar{u(x)}$")
    ax2.legend(loc='upper left')
    plt.show()
    fig.savefig('Numerical_experiments/NN_Picard_dim1/Pic_iter.png', bbox_inches='tight')


elif d == 2:

    x_axis_0 = np.linspace(-3.0, 3.0, Ntilde)
    x_axis_1 = np.linspace(-3.0, 3.0, Ntilde)
    x_grid_eval_0, x_grid_eval_1 = np.meshgrid(x_axis_0, x_axis_1)

    '''fig = plt.figure(figsize=(12, 5), dpi=75)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("$u(x)$")
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title(r"$\bar{u}(x)$")

    yy = an_u(x_axis_1)
    yyb = an_ub(x_axis_1)
    # yy_Delta = np.array([an_Delta_u(x_axis[i]) for i in range(len(x_axis))])
    ax1.plot(x_axis_1, yy, color='black', label=r"Analytical ${u}(x)$")
    ax2.plot(x_axis_1, yyb, color='brown', label=r"Analytical $\bar{u}(x)$")'''

    # print("x grid eval shape:", x_grid_eval_0.shape)
    # x_eval_points = np.transpose(np.array([x_grid_eval_0, x_grid_eval_1]).reshape(2, -1))

    # list of points from the 2-D grid
    x_eval_points = np.array([[[x_axis_0[i], x_axis_1[j]] for j in range(Ntilde)] for i in range(Ntilde)]).reshape(-1, 2)

    for p in range(num_pic):
        print("Picard iteration, p = ", p+1)

        X_train = sampleX()
        Y_train = label(X_train, model_0)

        #print("Y_train=", Y_train)
        #print("dtype: ", type(Y_train))

        #X_test = sampleX()
        #Y_test = label(X_test, model_0)

        #print("Y_train: ", Y_train, type(Y_train))
        model_1.fit(X_train, Y_train,
                    batch_size=batch_size,
                    shuffle=False,
                    epochs=epochs,
                    verbose=1)

        model_0 = model_1
        #if (p % 3 == 0):
        plt.subplot(121)

        predict = model_1.predict(x_eval_points)
        #print("predict shape:", predict.shape)

        fig = plt.figure(figsize=(24, 7), dpi=75)
        #ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 0].reshape(Ntilde, Ntilde), label=p+1)
        ax.plot_wireframe(x_grid_eval_0, x_grid_eval_1, an_u(x_eval_points).reshape(Ntilde, Ntilde), color='black', label=p+1)
        ax.set_title(r"$u^n(x^1, x^2)$, $n={}$".format(p+1))

        '''ax = fig.add_subplot(1, 2, 2, projection='3d')
        #print("ub_err shape:", np.sqrt(np.mean((predict[:, 1:] - an_ub(x_eval_points))**2, axis=-1)).shape)
        #print("predict ub shape:", predict[:, 1:].shape, "an_ub shape:", an_ub(x_eval_points).shape)
        ub_err = np.sqrt(np.mean((predict[:, 1:] - an_ub(x_eval_points))**2, axis=-1)).reshape(Ntilde, Ntilde)
        ax.plot_surface(x_grid_eval_0, x_grid_eval_1, ub_err, label=p+1)
        ax.set_title("l^2 error for u_b(x_1, x_2)")'''

        #fig.savefig('Numerical_experiments/NN_Picard_mult/u_and_l2_err_iter_{}.png'.format(p + 1), bbox_inches='tight')

        #fig = plt.figure(figsize=(16, 7), dpi=75)
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 1].reshape(Ntilde, Ntilde), color='red', label=p + 1)
        ax.plot_wireframe(x_grid_eval_0, x_grid_eval_1, an_ub(x_eval_points)[:, 0].reshape(Ntilde, Ntilde), color='gray',
                          label=p + 1)
        ax.set_title(r"$\bar u^{1, n}(x^1, x^2)$, $n=%s$" %(p+1))

        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.plot_surface(x_grid_eval_0, x_grid_eval_1, predict[:, 2].reshape(Ntilde, Ntilde), color='red', label=p + 1)
        ax.plot_wireframe(x_grid_eval_0, x_grid_eval_1, an_ub(x_eval_points)[:, 1].reshape(Ntilde, Ntilde), color='gray',
                          label=p + 1)
        ax.set_title(r"$\bar u^{1, n}(x^1, x^2)$, $n=%s$" %(p+1))

        fig.savefig('Numerical_experiments/NN_Picard_mult/ub_1_and_ub_2_iter_{}.png'.format(p + 1), bbox_inches='tight')

    x_axis_err = sampleX(num_sample=M_err)
    predict_err = model_1.predict(x_axis_err)
    u_err = np.abs(predict_err[:, :1] - an_u(x_axis_err)).reshape(M_err)
    print("shape 1= ", u_err.shape)
    ub_err = np.sqrt(np.mean((predict_err[:, 1:] - an_ub(x_axis_err)) ** 2, axis=-1))
    print("shape 2= ", ub_err.shape)
    np.save('Numerical_experiments/error_plots/u_Kz_{}_{}d.npy'.format(K_z, d), u_err)
    np.save('Numerical_experiments/error_plots/ub_Kz_{}_{}d.npy'.format(K_z, d), ub_err)
    print("ave", np.mean(u_err), np.mean(ub_err))
    print("max", np.max(u_err), np.max(ub_err))

    # temp = np.load('Numerical_experiments/NN_Picard_dim1/ub_err_1d.npy')
    # print(temp)

elif d>=3:

    '''x_domain = np.linspace(-3.0, 3.0, Ntilde)
    mesh = np.meshgrid(*[x_domain for _ in range(d)])
    ravelled = list(map(np.ravel, mesh))
    reshaped = [ravelled[i].reshape(-1, 1) for i in range(d)]
    x_grid_domain = np.concatenate(reshaped, axis=-1)
    #print(x_grid_domain.shape)'''

    errors = []

    for p in range(num_pic):

        print("Picard iteration, p = ", p+1)

        X_train = sampleX()
        Y_train = label(X_train, model_0)

        model_1.fit(X_train, Y_train,
                    batch_size=batch_size,
                    shuffle=False,
                    epochs=epochs,
                    verbose=1)

        model_0 = model_1

        X_test = sampleX(sig=0.8*sig_X, num_sample=int(0.1*M))
        Y_test = model_1.predict(X_test)

        u_pred = Y_test[:, :1]
        ub_pred = Y_test[:, 1:]

        an_u_test = an_u(X_test)
        an_ub_test = an_ub(X_test)

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



        '''analytical_u = an_u(x_grid_domain)
        analytical_ub = an_ub(x_grid_domain)

        predict_u = model_1.predict(x_grid_domain)[:, :1]
        predict_ub = model_1.predict(x_grid_domain)[:, 1:]

        print("an_u shape:", analytical_u.shape, "an_ub shape:", analytical_ub.shape)
        print("predict_u shape:", predict_u.shape, "predict_ub shape:", predict_ub.shape)

        L2_err_u = np.sqrt(np.mean(np.mean((predict_u - analytical_u)**2, axis=-1)))
        L_inf_err_u = np.max(np.mean(np.abs(predict_u - analytical_u), axis=-1))
        arg_L_inf_err_u = coordpoint(np.argmax(np.mean(np.abs(predict_u - analytical_u), axis=-1)))

        print("mean L^2 error for u:", L2_err_u, "\t L^inf error for u:", L_inf_err_u, "\t the error is maximum at:",
                                                                                       arg_L_inf_err_u)

        L2_err_ub = np.sqrt(np.mean(np.mean((predict_ub - analytical_ub) ** 2, axis=-1)))
        L_inf_err_ub = np.max(np.mean(np.abs(predict_ub - analytical_ub), axis=-1))
        arg_L_inf_err_ub = coordpoint(np.argmax(np.mean(np.abs(predict_ub - analytical_ub), axis=-1)))

        print("mean L^2 error for u_bar:", L2_err_ub, "\t L^inf error for u_bar:", L_inf_err_ub,
              "\t the error is maximum at:", arg_L_inf_err_ub)'''

    x_axis_err = sampleX(num_sample=M_err)
    predict_err = model_1.predict(x_axis_err)
    u_err = np.abs(predict_err[:, :1] - an_u(x_axis_err)).reshape(M_err)
    print("shape 1= ", u_err.shape)
    ub_err = np.sqrt(np.mean((predict_err[:, 1:] - an_ub(x_axis_err)) ** 2, axis=-1))
    print("shape 2= ", ub_err.shape)
    np.save('Numerical_experiments/error_plots/u_Kz_{}_{}d.npy'.format(K_z, d), u_err)
    np.save('Numerical_experiments/error_plots/ub_Kz_{}_{}d.npy'.format(K_z, d), ub_err)
    print("ave", np.mean(u_err), np.mean(ub_err))
    print("max", np.max(u_err), np.max(ub_err))

elapsed_time = time.time() - start_time
print("elapsed time: ", elapsed_time)

if d>=3:

    for p in range(num_pic):
        print("Picard iteration: {},\n L^2_mu error for u: {}, L^inf_mu error for u: {}, error is max at: {},"
              .format(p+1, errors[p][0], errors[p][1], errors[p][2]))
        print("L^2_mu error for ub: {}, L^inf_mu error for ub: {}, error is max at: {}.".format(errors[p][3], errors[p][4],
                                                                                                errors[p][5]))


    fig = plt.figure(figsize=(24, 14), dpi=75)
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

    ax3.plot(np.arange(1, num_pic+1, 1), [errors[i][3] for i in range(num_pic)])
    ax3.set_title("L^2_mu errors for ub", fontweight='bold')
    ax3.set_xlabel("Picard Iterations")
    ax3.set_ylabel("L^2_mu error")

    ax4.plot(np.arange(1, num_pic+1, 1), [errors[i][4] for i in range(num_pic)])
    ax4.set_title("L^inf_mu errors for ub", fontweight='bold')
    ax4.set_xlabel("Picard Iterations")
    ax4.set_ylabel("L^inf_mu error")

    fig.savefig("Numerical_experiments/NN_Picard_multi_2/Pic_errors_ub_dim_{}.png".format(d))
    plt.show()





#logging.info("Pic_num: %3u,   loss: %.4e,   accuracy: %.4e,   elapsed_time: %4u" %
#             (p+1, loss, accuracy, elapsed_time))
# Plotting the last iterations and analytical solutions

'''x_axis_0 = np.linspace(-3.0, 3.0, 201)
x_axis_1 = x_axis_0
x_eval_points =
y_axis = model_1.predict(x_axis)
y_axis_0 = y_axis[:, 0]
plt.subplot(121)
plt.plot(x_axis, y_axis_0, color='red', label='u prediction')
plt.plot(x_axis, np.array([an_u(x_axis[i]) for i in range(len(x_axis))]), color='green', label='analytical u')
plt.legend()
plt.subplot(122)
y_axis_1 = y_axis[:, 1]
plt.plot(x_axis, y_axis_1, color='black', label='u_bar prediction')
plt.plot(x_axis, np.array([an_ub(np.array([x_axis[i]])) for i in range(len(x_axis))]), color='blue', label='analytical u_bar')
plt.legend()
plt.show()

logging.info('Parameters in this run:'
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
