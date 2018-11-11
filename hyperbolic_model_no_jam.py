import numpy as np
import traffic_flow as tf


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


h = 0.05
k = 0.005
x = np.arange(-1, 1.6, h)

num_divs = len(x)

rho0 = np.linspace(0.1, 0.4, num_divs)

traffic = tf.TrafficFlow(
    h=h,
    k=k,
    x=x,
    total_time=5,
    rho0=rho0,
    f=f,
    df=df,
    df_eq_0=df_eq_0,
    u_to_rho=u_to_rho,
    rho_to_u=rho_to_u
)

traffic.update_godunov()
traffic.show_animation()
