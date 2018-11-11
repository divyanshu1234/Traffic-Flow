import numpy as np
import traffic_flow as tf


# Functions for Greenshields Model
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


h = 0.05
k = 0.005
x = np.arange(-1, 1.6, h)

num_divs = len(x)

# Different initial conditions
rho0 = 0.55 * np.ones(num_divs)

# rho0 = np.linspace(0.1, 0.5, num_divs)

# rho0 = np.zeros(num_divs)
# rho0[x <= 0] = 1
# rho0[x > 0] = 0

traffic = tf.TrafficFlow(
    h=h,
    k=k,
    x=x,
    total_time=4,
    rho0=rho0,
    f=f,
    df=df,
    df_eq_0=df_eq_0,
    u_to_rho=u_to_rho,
    rho_to_u=rho_to_u
)

traffic.enable_signal(signal_position=0.4, toggle_time_list=np.array([4]))
traffic.enable_speed_breaker(sb_position=0.8)

traffic.update_godunov()
traffic.show_animation()
