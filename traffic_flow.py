import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

RHO_MAX = 1
RHO_MIN = 0
RHO_CRITICAL = 0.4
RHO_JAM = 1

V_MAX = 0.95
V_SB = 0.1


class TrafficFlow:
    """
    Notes
    -----
    Only __init__, enable_signal, enable_speed_breaker, update_gudonov and show_animation
    are supposed to be called from outside the class

    """
    def __init__(self, h, k, x, total_time, rho0, f, df, df_eq_0, u_to_rho, rho_to_u):
        """
        Parameters
        ----------
        h : float
            Length step
        k : float
            Time step
        x : float array
            Discrete space domain
        total_time : float
            Total time for the simulation
        rho0 : float array
            Initial condition of density on x
        f : function
            Flow function
        df : function
            Derivative of flow function
        df_eq_0 : function
            Returns value of u for which df = 0, for Gudonov scheme
        u_to_rho : function
            Converts u to rho
        rho_to_u : function
            Converts rho to u
        """

        self.h = h
        self.k = k
        self.x = x
        self.total_time = total_time
        self.init_cond = rho_to_u(rho0)
        self.f = f
        self.df = df
        self.df_eq_0 = df_eq_0
        self.u_to_rho = u_to_rho
        self.rho_to_u = rho_to_u

        self.num_divs = len(x)
        self.num_time_steps = int(total_time / k)
        self.u = np.zeros((self.num_time_steps, self.num_divs))
        self.is_signal_enabled = False
        self.signal_position = None
        self.signal_position_index = None
        self.toggle_time_list = None
        self.toggle_time_indies = None
        self.is_sb_enabled = False
        self.sb_position = None
        self.sb_position_index = None

        self.u[0, :] = self.init_cond
        self.u[:, 0] = self.u[0, 0] * np.ones(self.num_time_steps)

    def enable_signal(self, signal_position, toggle_time_list):
        """
        Used to enable the traffic signal

        Parameters
        ----------
        signal_position : float
            Position of the traffic signal on the road, should belong to x
        toggle_time_list : float list
            List of time instances when the traffic signal changes
            Initially it is red, the first number in the list changes red to green first
        """

        self.is_signal_enabled = True
        self.signal_position = signal_position
        self.signal_position_index = int((signal_position - self.x[0]) / self.h)
        self.toggle_time_list = np.append(np.insert(toggle_time_list, 0, 0), self.total_time)
        self.toggle_time_indies = (self.toggle_time_list / self.k).astype(int)

        self.apply_boundary_conditions()

    def enable_speed_breaker(self, sb_position):
        """
        Used to enable the speed breaker

        Parameters
        ----------
        sb_position : float
            Position of the speed breaker on the road, should belong to x
        """

        self.is_sb_enabled = True
        self.sb_position = sb_position
        self.sb_position_index = int((sb_position - self.x[0]) / self.h)

    def update_godunov(self):
        """Call this function to do the calculations"""

        self.u[:, 0] = self.u[0, 0] * np.ones(self.num_time_steps)
        for n in range(self.num_time_steps - 1):
            self.apply_boundary_conditions()

            for i in range(1, self.num_divs - 1):
                self.u[n + 1, i] = self.u[n, i] - self.k / self.h * \
                                   (self.f(self.u_star(n, i)) - self.f(self.u_star(n, i - 1)))

                if self.is_sb_enabled:
                    if i == self.sb_position_index:
                        self.u[n + 1, i] = V_SB * self.u[n + 1, i]

            if n % (self.num_time_steps / 100) == 0:
                print(n, self.num_time_steps - 2)

        self.u[:, 0] = self.u[:, 1]
        self.u[:, -1] = self.u[:, -2]

    def show_animation(self):
        """Call this function to show the animation of the traffic flow"""

        fig = plt.figure()
        ax = plt.axes(xlim=(self.x[0] - 0.5, self.x[-1] + 0.5), ylim=(-0.5, 1.5))
        line, = ax.plot([], [], lw=2)

        def init():
            plt.xlabel('Road')
            plt.ylabel('Density')
            line.set_data([], [])
            return line,

        def animate(n):
            x_n = self.x
            rho_n = self.u_to_rho(self.u)[n, :]

            del_indices = []
            if self.is_signal_enabled:
                del_indices.append(self.signal_position_index)
            if self.is_sb_enabled:
                del_indices.append(self.sb_position_index)

            line.set_data(np.delete(x_n, del_indices), np.delete(rho_n, del_indices))
            return line,

        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=self.num_time_steps,
            interval=10,
            blit=True)

        plt.show()

    def apply_boundary_conditions(self):
        if self.is_signal_enabled:
            for i in np.arange(0, len(self.toggle_time_indies) - 1, 2):
                self.u[self.toggle_time_indies[i]:
                       self.toggle_time_indies[i + 1], self.signal_position_index] \
                    = self.rho_to_u(RHO_MAX)

                self.u[self.toggle_time_indies[i]:
                       self.toggle_time_indies[i + 1], self.signal_position_index + 1] \
                    = self.rho_to_u(RHO_MIN)

        if self.is_sb_enabled:
            if self.is_signal_enabled:
                if self.sb_position_index > self.signal_position_index:
                    self.u[:, self.sb_position_index + 1] = self.rho_to_u(RHO_MIN)
                else:
                    self.u[:, self.sb_position_index] = self.rho_to_u(RHO_MAX)
            else:
                self.u[:, self.sb_position_index + 1] = self.rho_to_u(RHO_MIN)

    def get_s(self, ui_1, ui, i):
        s = (self.f(ui_1) - self.f(ui)) / (ui_1 - ui)

        if self.is_sb_enabled:
            if i == self.sb_position_index:
                s = V_SB * s

        return s

    def u_star(self, n, i):
        g_u_n_i = self.df(self.u[n, i])
        g_u_n_i1 = self.df(self.u[n, i + 1])

        if self.is_sb_enabled:
            if i == self.sb_position_index:
                g_u_n_i = V_SB * g_u_n_i
                g_u_n_i1 = V_SB * g_u_n_i1

        if g_u_n_i >= 0 and g_u_n_i1 >= 0:
            u_star_i = self.u[n, i]

        elif g_u_n_i < 0 and g_u_n_i1 < 0:
            u_star_i = self.u[n, i + 1]

        elif g_u_n_i >= 0 and g_u_n_i1 < 0:
            s = self.get_s(self.u[n, i + 1], self.u[n, i], i)
            if s >= 0:
                u_star_i = self.u[n, i]
            else:
                u_star_i = self.u[n, i + 1]

        else:
            u_star_i = self.df_eq_0()

        return u_star_i
