import numpy as np
import matplotlib.pyplot as plt


RHO_MAX = 1
H = 0.14
K = 0.005
X = np.arange(-0.3, 1.6, H)

NUM_DIVS = len(X)
NUM_TIME_STEPS = 1000


def rho_to_u(rho):
    return 1 - 2 * rho / RHO_MAX


def u_to_rho(u):
    return (1 - u) * RHO_MAX / 2


def f(u):
    return u ** 2 / 2


def df(u):
    return u


def get_s(ui_1, ui):
    s = (f(ui_1) - f(ui)) / (ui_1 - ui)
    return s


def get_init_conditions():
    u0 = np.zeros(NUM_DIVS)
    u0[X <= 0] = -1
    u0[X > 0] = 1

    return u0


def u_star(u_arr, n, i):
    g_u_n_i = df(u_arr[n, i])
    g_u_n_i1 = df(u_arr[n, i + 1])

    if g_u_n_i >= 0 and g_u_n_i1 >= 0:
        u_star_i = u_arr[n, i]

    elif g_u_n_i < 0 and g_u_n_i1 < 0:
        u_star_i = u_arr[n, i+1]

    elif g_u_n_i >= 0 and g_u_n_i1 < 0:
        s = get_s(u_arr[n, i+1], u_arr[n, i])

        if s >= 0:
            u_star_i = u_arr[n, i]
        else:
            u_star_i = u_arr[n, i+1]

    else:
        u_star_i = 0

    return u_star_i


def update_gudonov(u_arr):
    u_arr[:, 0] = u_arr[0, 0] * np.ones(NUM_TIME_STEPS)
    for n in range(NUM_TIME_STEPS-1):
        for i in range(1, NUM_DIVS-1):
            u_arr[n+1, i] = u_arr[n, i] - K / H * (f(u_star(u_arr, n, i)) - f(u_star(u_arr, n, i-1)))

        print(n)

    return u_arr


def plot_u_t(u_arr, t):
    n = int(t / K)
    u_t = u_arr[n, :]
    plt.plot(np.linspace(-0.3, 1.6, NUM_DIVS), u_t)


u_arr = np.zeros((NUM_TIME_STEPS, NUM_DIVS))
u_arr[0, :] = get_init_conditions()

u_arr1 = update_gudonov(u_arr)

plot_u_t(u_arr1, 0)
plot_u_t(u_arr1, 0.005)
plot_u_t(u_arr1, 0.5)
plot_u_t(u_arr1, 1)

plt.show()
