import numpy as np
import traffic_flow as tf
from matplotlib import pyplot as plt


# Functions for Greenshields Model
# def f(u):
#     return u ** 2 / 2
#
#
# def df(u):
#     return u
#
#
# def rho_to_u(rho):
#     return 1 - 2 * rho / tf.RHO_MAX
#
#
# def u_to_rho(u):
#     return (1 - u) * tf.RHO_MAX / 2
#
#
# def df_eq_0():
#     return 0


# Functions for hyperbolic speed model
def f(rho):
    if rho <= tf.RHO_CRITICAL:
        return rho * tf.V_MAX
    elif tf.RHO_CRITICAL < rho < tf.RHO_JAM:
        return tf.V_MAX * (1 - rho/tf.RHO_JAM)
    elif rho >= tf.RHO_JAM:
        return 0


def df(rho):
    if rho <= tf.RHO_CRITICAL:
        return tf.V_MAX
    elif tf.RHO_CRITICAL < rho < tf.RHO_JAM:
        return -tf.V_MAX / tf.RHO_JAM
    elif rho >= tf.RHO_JAM:
        return 0


def u_to_rho(u):
    return u


def rho_to_u(rho):
    return rho


def df_eq_0():
    return 0


def plot_u_t(traf, t):
    n = int(t / k)
    u_arr = traf.u
    u_t = u_arr[n, :]
    num_divs = len(u_t)
    plt.figure()
    plt.plot(np.linspace(-0.3, 1.6, num_divs), u_t)
    plt.show()


def plot_rho_t(traf, t):
    n = int(t / k)
    rho_arr = u_to_rho(traf.u)
    rho_t = rho_arr[n, :]

    del_indices = []
    if traf.is_signal_enabled:
        del_indices.append(traf.signal_position_index)
    if traf.is_sb_enabled:
        del_indices.append(traf.sb_position_index)

    plt.figure()
    plt.axes(xlim=(traf.x[0] - 0.5, traf.x[-1] + 0.5), ylim=(-0.5, 1.5))
    plt.plot(np.delete(traf.x, del_indices), np.delete(rho_t, del_indices))
    plt.show()


h = 0.05
k = 0.005
x = np.arange(-1, 1.6, h)

num_divs = len(x)

# Different initial conditions
# rho0 = 0.55 * np.ones(num_divs)

rho0 = np.linspace(0.1, 0.5, num_divs)

# rho0 = np.zeros(num_divs)
# rho0[x <= 0] = 1
# rho0[x > 0] = 0

traffic = tf.TrafficFlow(
    h=h,
    k=k,
    x=x,
    total_time=10,
    rho0=rho0,
    f=f,
    df=df,
    df_eq_0=df_eq_0,
    u_to_rho=u_to_rho,
    rho_to_u=rho_to_u
)

# Uncomment for traffic signal
# traffic.enable_signal(signal_position=0.4, toggle_time_list=np.array([2, 5, 7]))

# Uncomment for speed breaker
# traffic.enable_speed_breaker(sb_position=0.1)

traffic.update_godunov()
traffic.show_animation()
