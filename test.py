import numpy as np
import traffic_flow as tf
from matplotlib import pyplot as plt
from matplotlib import animation


def f(u):
    return u ** 2 / 2


def df(u):
    return u


def get_init_conditions(num_divs):
    # u0 = np.zeros(NUM_DIVS)
    # u0[X <= 0] = -1
    # u0[X > 0] = 1
    u0 = tf.rho_to_u(0.55) * np.ones(num_divs)

    return u0


def plot_u_t(traf, t):
    n = int(t / k)
    u_arr = traf.u
    u_t = u_arr[n, :]
    num_divs = len(u_t)
    plt.plot(np.linspace(-0.3, 1.6, num_divs), u_t)


def plot_rho_t(traf, t):
    n = int(t / k)
    rho_arr = traf.get_rho()
    rho_t = rho_arr[n, :]
    num_divs = len(rho_t)
    plt.plot(np.linspace(-0.3, 1.6, num_divs), rho_t)


h = 0.05
k = 0.005
x = np.arange(-1, 1.6, h)
u0 = get_init_conditions(len(x))

# def __init__(self, h, k, x, total_time, init_cond, f, df):
traffic = tf.TrafficFlow(h=h, k=k, x=x, total_time=20, init_cond=u0, f=f, df=df)
traffic.enable_light(light_position=0.4, toggle_time_list=np.array([2, 5, 7, 10, 15]))
# traffic.enable_speed_breaker(sb_position=0.6)
traffic.update_gudonov()
traffic.show_animation()


