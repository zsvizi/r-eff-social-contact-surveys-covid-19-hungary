import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Plotter:
    def __init__(self, sim_obj):
        self.sim_obj = sim_obj
        self.filename_base = self.sim_obj.data.get_contact_data_filename()

    def plot_r_eff(self, r_eff: np.ndarray) -> None:
        """
        Creates R_eff plot
        :param r_eff: np.ndarray, calculated effective reproduction numbers
        :return: None
        """
        # Plot effective reproduction number
        b_s = self.sim_obj.bin_size
        t0 = np.linspace(0, self.sim_obj.start_date_delta, 1 + self.sim_obj.start_date_delta * b_s)
        t1 = np.linspace(self.sim_obj.start_date_delta,
                         (self.sim_obj.time_plot - 1) * self.sim_obj.time_step + self.sim_obj.start_date_delta,
                         1 + ((self.sim_obj.time_plot - 1) * self.sim_obj.time_step) * b_s)
        t = np.append(t0, t1[1:], axis=0)
        fig = plt.figure(figsize=(6, 6))
        plt.plot(t, r_eff)
        plt.plot(t[b_s:-b_s:b_s], self.sim_obj.data.contact_num_data["outside"].to_numpy() +
                 self.sim_obj.data.contact_num_data["family"].to_numpy())
        self._generate_date(fig)
        fig.savefig(os.path.join("./plots", self.filename_base + '_r_eff.pdf'))
        plt.show()

    def plot_dynamics(self, sol: np.ndarray) -> None:
        """
        Create plots about model dynamics
        :param sol: np.ndarray, calculated piecewise solution
        :return: None
        """
        # Plot daily incidence
        t = np.linspace(0, self.sim_obj.time_plot, self.sim_obj.time_plot * self.sim_obj.bin_size)
        fig = plt.figure(figsize=(6, 6))
        plt.plot(t, np.diff(self.sim_obj.model.get_cumulative(sol)))
        self._generate_date(fig)
        fig.savefig(os.path.join("./plots", self.filename_base + '_daily_incidence.pdf'))
        plt.show()

        # Plot cumulative cases
        t = np.linspace(0, self.sim_obj.time_plot, 1 + self.sim_obj.time_plot * self.sim_obj.bin_size)
        fig = plt.figure(figsize=(6, 6))
        plt.plot(t, self.sim_obj.model.get_cumulative(sol))
        self._generate_date(fig)
        fig.savefig(os.path.join("./plots", self.filename_base + '_cumulative.pdf'))
        plt.show()

    def plot_dominant_eigenvalues(self) -> None:
        """
        Creates plot about dominant eigenvalues
        :return: None
        """
        t = range(len(self.sim_obj.r0_generator.debug_list))
        fig = plt.figure(figsize=(6, 6))
        plt.plot(t, np.array(self.sim_obj.r0_generator.debug_list))
        self._generate_date(fig)
        fig.savefig(os.path.join("./plots", self.filename_base + '_debug.pdf'))
        plt.show()

    def _generate_date(self, fig):
        """
        Generate dates along x axis on the input figure object
        :param fig: figure object
        :return: None
        """
        date_bin = 14
        list_of_dates = np.array([
            d.strftime('%m-%d') for d in pd.date_range(start=self.sim_obj.start_date,
                                                       periods=self.sim_obj.time_plot * self.sim_obj.time_step + 1)
        ])
        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(np.arange(len(list_of_dates[::date_bin])) * date_bin, list_of_dates[::date_bin])
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
