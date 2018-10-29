import numpy as np

RHO_MAX = 1


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
        self.change_time_list = None
        self.change_time_indies = None
        self.boundary_conditions = None
        self.self = self

        self.u[0, :] = self.init_cond
        self.u[:, 0] = self.u[0, 0] * np.ones(self.num_time_steps)

    def enable_light(self, light_position, change_time_list):
        # Initially it is red
        # Change red to green first
        self.is_light_enabled = True
        self.light_position = light_position
        self.light_position_index = int((light_position - self.x[0]) / self.h)
        self.change_time_list = np.append(np.insert(change_time_list, 0, 0), self.total_time)
        self.change_time_indies = (self.change_time_list / self.k).astype(int)

        self.apply_boundary_conditions()

    def apply_boundary_conditions(self):
        if self.is_light_enabled:
            for i in np.arange(0, len(self.change_time_indies) - 1, 2):
                self.u[self.change_time_indies[i]: self.change_time_indies[i + 1], self.light_position_index] = -1
                self.u[self.change_time_indies[i]: self.change_time_indies[i + 1], self.light_position_index + 1] = 1

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
