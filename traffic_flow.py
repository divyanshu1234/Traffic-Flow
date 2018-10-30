import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


RHO_MAX = 1
V_SB = 0.1


def rho_to_u(rho):
    return 1 - 2 * rho / RHO_MAX


def u_to_rho(u):
    return (1 - u) * RHO_MAX / 2


class TrafficFlow:

    def __init__(self, h, k, x, total_time, init_cond, f, df):
        self.h = h
        self.k = k
        self.x = x
        self.total_time = total_time
        self.init_cond = init_cond
        self.f = f
        self.df = df
        self.num_divs = len(x)
        self.num_time_steps = int(total_time / k)
        self.u = np.zeros((self.num_time_steps, self.num_divs))
        self.is_light_enabled = False
        self.light_position = None
        self.light_position_index = None
        self.toggle_time_list = None
        self.toggle_time_indies = None
        self.is_sb_enabled = False
        self.sb_position = None
        self.sb_position_index = None

        self.u[0, :] = self.init_cond
        self.u[:, 0] = self.u[0, 0] * np.ones(self.num_time_steps)

    def enable_light(self, light_position, toggle_time_list):
        # Initially it is red
        # Change red to green first
        self.is_light_enabled = True
        self.light_position = light_position
        self.light_position_index = int((light_position - self.x[0]) / self.h)
        self.toggle_time_list = np.append(np.insert(toggle_time_list, 0, 0), self.total_time)
        self.toggle_time_indies = (self.toggle_time_list / self.k).astype(int)

        self.apply_boundary_conditions()

    # Todo - fix speed breaker conditions
    def enable_speed_breaker(self, sb_position):
        self.is_sb_enabled = True
        self.sb_position = sb_position
        self.sb_position_index = int((sb_position - self.x[0]) / self.h)

    def apply_boundary_conditions(self):
        if self.is_light_enabled:
            for i in np.arange(0, len(self.toggle_time_indies) - 1, 2):
                self.u[self.toggle_time_indies[i]: self.toggle_time_indies[i + 1], self.light_position_index] = -1
                self.u[self.toggle_time_indies[i]: self.toggle_time_indies[i + 1], self.light_position_index + 1] = 1

        if self.is_sb_enabled:
            self.u[:, self.sb_position_index] = V_SB
            if self.is_light_enabled:
                if self.sb_position_index > self.light_position_index:
                    self.u[:, self.sb_position_index + 1] = 1

    def get_s(self, ui_1, ui):
        s = (self.f(ui_1) - self.f(ui)) / (ui_1 - ui)
        return s

    def u_star(self, n, i):
        g_u_n_i = self.df(self.u[n, i])
        g_u_n_i1 = self.df(self.u[n, i + 1])

        if g_u_n_i >= 0 and g_u_n_i1 >= 0:
            u_star_i = self.u[n, i]

        elif g_u_n_i < 0 and g_u_n_i1 < 0:
            u_star_i = self.u[n, i + 1]

        elif g_u_n_i >= 0 and g_u_n_i1 < 0:
            s = self.get_s(self.u[n, i + 1], self.u[n, i])
            if s >= 0:
                u_star_i = self.u[n, i]
            else:
                u_star_i = self.u[n, i + 1]

        else:
            u_star_i = 0

        return u_star_i

    def update_gudonov(self):
        self.u[:, 0] = self.u[0, 0] * np.ones(self.num_time_steps)
        for n in range(self.num_time_steps - 1):
            self.apply_boundary_conditions()

            for i in range(1, self.num_divs - 1):
                self.u[n + 1, i] = self.u[n, i] - self.k / self.h * \
                                   (self.f(self.u_star(n, i)) - self.f(self.u_star(n, i - 1)))

            if n % (self.num_time_steps / 100) == 0:
                print(n, self.num_time_steps - 2)

        self.u[:, 0] = self.u[:, 1]
        self.u[:, -1] = self.u[:, -2]

    def get_rho(self):
        return (1 - self.u) * RHO_MAX / 2

    def show_animation(self):
        fig = plt.figure()
        ax = plt.axes(xlim=(self.x[0] - 0.5, self.x[-1] + 0.5), ylim=(-0.5, 1.5))
        line, = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            line.set_data(self.x, self.get_rho()[i, :])
            return line,

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=self.num_time_steps, interval=5,
                                       blit=True)
        plt.show()
