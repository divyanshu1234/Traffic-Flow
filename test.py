import numpy as np
import traffic_flow as tf
from matplotlib import pyplot as plt


def f(u):
    return u ** 2 / 2


def df(u):
    return u


def rho_to_u(rho):
    return 1 - 2 * rho / tf.RHO_MAX


def u_to_rho(u):
    return (1 - u) * tf.RHO_MAX / 2


def df_eq_0():
    return 0


# def f(rho):
#     if rho <= tf.RHO_CRITICAL:
#         return rho * tf.V_MAX
#     elif tf.RHO_CRITICAL < rho < tf.RHO_JAM:
#         return tf.V_MAX * (1 - rho/tf.RHO_JAM)
#     elif rho >= tf.RHO_JAM:
#         return 0
#
#
# def df(rho):
#     if rho <= tf.RHO_CRITICAL:
#         return tf.V_MAX
#     elif tf.RHO_CRITICAL < rho < tf.RHO_JAM:
#         return -tf.V_MAX / tf.RHO_JAM
#     elif rho >= tf.RHO_JAM:
#         return 0
#
#
# def u_to_rho(u):
#     return u
#
#
# def rho_to_u(rho):
#     return rho
#
#
# def df_eq_0():
#     return 0


def plot_u_t(traf, t):
    n = int(t / k)
    u_arr = traf.u
    u_t = u_arr[n, :]
    num_divs = len(u_t)
    plt.plot(np.linspace(-0.3, 1.6, num_divs), u_t)


def plot_rho_t(traf, t):
    n = int(t / k)
    rho_arr = u_to_rho(traf.u)
    rho_t = rho_arr[n, :]
    num_divs = len(rho_t)
    plt.plot(np.linspace(-0.3, 1.6, num_divs), rho_t)


h = 0.05
k = 0.005
x = np.arange(-1, 1.6, h)

num_divs = len(x)

rho0 = 0.20 * np.ones(num_divs)
# rho0 = np.linspace(0.0, 0.4, len(x))


traffic = tf.TrafficFlow(h=h, k=k, x=x, total_time=20, rho0=rho0, f=f, df=df, df_eq_0=df_eq_0, u_to_rho=u_to_rho, rho_to_u=rho_to_u)
traffic.enable_light(light_position=0.4, toggle_time_list=np.array([2, 5, 7, 10, 15]))
traffic.enable_speed_breaker(sb_position=0.1)
traffic.update_gudonov()
traffic.show_animation()
