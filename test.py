import numpy as np
import traffic_flow as tf
from matplotlib import pyplot as plt
from matplotlib import animation


def f(u):
    return u ** 2 / 2


def df(u):
    return u


def rho_to_u(rho):
    return 1 - 2 * rho / RHO_MAX


def u_to_rho(u):
    return (1 - u) * RHO_MAX / 2


def get_init_conditions(num_divs):
    # u0 = np.zeros(NUM_DIVS)
    # u0[X <= 0] = -1
    # u0[X > 0] = 1
    u0 = rho_to_u(0.55) * np.ones(num_divs)

    return u0


def plot_u_t(u_arr, t):
    n = int(t / k)
    u_t = u_arr[n, :]
    num_divs = len(u_t)
    plt.plot(np.linspace(-0.3, 1.6, num_divs), u_t)


def plot_rho_t(rho_arr, t):
    n = int(t / k)
    rho_t = rho_arr[n, :]
    num_divs = len(rho_t)
    plt.plot(np.linspace(-0.3, 1.6, num_divs), rho_t)


def show_animation():
    fig = plt.figure()
    ax = plt.axes(xlim=(-1, 2), ylim=(-0.5, 1.5))
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(x, rho_arr[i, :])
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=traffic.num_time_steps, interval=5, blit=True)
    plt.show()


RHO_MAX = 1
h = 0.05
k = 0.005
x = np.arange(-1, 1.6, h)
u0 = get_init_conditions(len(x))

# def __init__(self, h, k, x, total_time, init_cond, f, df):
traffic = tf.TrafficFlow(h, k, x, 10, u0, f, df)
traffic.enable_light(0.4, np.array([2, 5, 7]))
traffic.update_gudonov()
rho_arr = traffic.get_rho()

show_animation()


# plot_u_t(traffic.u, 0)
# plot_u_t(traffic.u, 0.005)
# plot_u_t(traffic.u, 0.5)
# plot_u_t(traffic.u, 1)

# plot_rho_t(traffic.get_rho(), 0)
# plot_rho_t(traffic.get_rho(), 0.5)
# plot_rho_t(traffic.get_rho(), 1)
# plot_rho_t(traffic.get_rho(), 2)
# plt.legend(['0', '0.5', '1', '4'])

# plt.show()

