import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataloader import DataLoader, transform_matrix
from model import RostModelHungary
from r0 import R0Generator


class Simulation:
    def __init__(self):
        """
        Constructor initializing class by
        - loading data (contact matrices, model parameters, age distribution)
        - creates model and r0generator objects
        - calculates initial transmission rate
        """
        # Debug variable
        self.debug = True
        # Instantiate DataLoader object to load model parameters, age distributions and contact matrices
        self.data = DataLoader()
        # Instantiate dynamical system
        self.model = RostModelHungary(model_data=self.data)

        self.r0 = 2.2
        # Get model parameters from DataLoader and append susceptibility age vector to the dictionary
        self.parameters = self.data.model_parameters_data
        self.parameters.update({"susc": np.array([0.5, 0.5, 1, 1, 1, 1, 1, 1])})

        # Instantiate R0generator object for calculating effective reproduction numbers
        self.r0_generator = R0Generator(param=self.parameters, n_age=self.model.n_age,
                                        debug=self.debug)
        # Calculate initial transmission rate (beta) based on reference matrix and self.r0
        self.parameters.update({"beta": self._get_initial_beta()})

        # Number of points evaluated for a time unit in odeint
        self.bin_size = 10
        # Number of contact matrices used for the simulation
        self.n_cm = 210
        # Number of days, where one contact matrix is valid
        n_days = 1
        # Number of time points plotted
        self.time_plot = n_days * self.n_cm

    def run(self) -> None:
        """
        Run simulation, see details below
        :return: None
        """
        # Get transformed contact matrix (here, we have the reference matrix)
        # Transform means: multiply by age distribution as a column,
        # then take average of result and transpose of result
        # then divide by the age distribution as a column
        cm_tr = self._get_transformed_cm(cm=self.data.reference_contact_data.iloc[0].to_numpy())

        # Get solution for the first time interval (here, we have the reference matrix)
        solution = self._get_solution(contact_mtx=cm_tr)
        sol_plot = copy.deepcopy(solution)

        # Get effective reproduction numbers for the first time interval
        # R_eff is calculated at each points for which odeint gives values ('bin_size' number of values for one day
        r_eff = self._get_r_eff(cm=cm_tr, solution=solution)
        r_eff_plot = copy.deepcopy(r_eff)

        # Time variable (mainly for debugging purposes)
        t = 1
        # Piecewise solution of the dynamical model: change contact matrix on basis of n_days (see in constructor)
        for cm in self.data.contact_data.iloc[1:self.n_cm].to_numpy():
            # Transform actual contact matrix data
            cm_tr = self._get_transformed_cm(cm=cm)

            # Get solution for the actual time interval
            solution = self._get_solution(contact_mtx=cm_tr,
                                          iv=solution[-1])
            # Append this solution piece
            sol_plot = np.append(sol_plot, solution[1:], axis=0)

            # Get effective reproduction number for the actual time interval
            r_eff = self._get_r_eff(cm=cm_tr, solution=solution)
            r_eff_plot = np.append(r_eff_plot, r_eff[1:], axis=0)

            t += 1
        if self.debug:
            fig = plt.figure(figsize=(6, 6))
            plt.plot(range(len(self.r0_generator.debug_list)), np.array(self.r0_generator.debug_list))
            self._generate_date(fig)

            fig.savefig(os.path.join("./plots", 'debug.pdf'))
            plt.show()

        # Create plots about dynamics and R_eff values
        if not self.debug:
            self._plot_dynamics(sol_plot)
        self._plot_r_eff(r_eff_plot, case='2')

    def _get_initial_beta(self) -> float:
        """
        Calculates transmission rate used in the dynamical model based on the reference matrix
        :return: float, transmission rate for reference matrix
        """
        # Get transformed reference matrix
        cm = self._get_transformed_cm(cm=self.data.reference_contact_data.iloc[0].to_numpy())

        # Get initial values for susceptibles and population
        population = self.model.population
        initial_values = self.model.get_initial_values().reshape(1, -1)
        susceptibles = self.model.get_comp(initial_values, self.model.c_idx["s"])

        # Get largest eigenvalue of the NGM
        eig_value_0 = self.r0_generator.get_eig_val(contact_mtx=cm,
                                                    population=population,
                                                    susceptibles=susceptibles)[0]
        # Get initial beta from baseline R0
        beta = self.r0 / eig_value_0
        return beta

    def _get_transformed_cm(self, cm: np.ndarray) -> np.ndarray:
        """
        Symmetrizes input contact matrix
        :param cm: np.ndarray, input contact matrix as a row vector
        :return: np.ndarray, symmetrized contact matrix in a matrix form
        """
        return transform_matrix(age_data=self.data.age_data,
                                matrix=cm.reshape((self.model.n_age, self.model.n_age)))

    def _get_r_eff(self, cm: np.ndarray, solution: np.ndarray) -> np.ndarray:
        """
        Calculates r_eff values for actual time interval
        :param cm: np.ndarray, actual contact matrix
        :param solution: np.ndarray, solution piece of the model for the actual time interval
        :return: np.ndarray, r_eff values
        """
        susceptibles = self.model.get_comp(solution, self.model.c_idx["s"])
        r_eff = self.parameters["beta"] * self.r0_generator.get_eig_val(contact_mtx=cm,
                                                                        population=self.model.population,
                                                                        susceptibles=susceptibles)
        return r_eff

    def _get_solution(self, contact_mtx: np.ndarray, iv: np.ndarray = None) -> np.ndarray:
        """
        Solves dynamical model for actual time interval (assumes uniformly divided intervals!)
        :param contact_mtx: np.ndarray, actual contact matrix
        :param iv: np.ndarray, initial value for odeint, mostly end point of the previous solution piece
        :return: np.ndarray, solution piece for this interval
        """
        # Get time interval
        t = np.linspace(0, self.time_plot / self.n_cm, 1 + int(self.time_plot * self.bin_size / self.n_cm))
        # For first time interval, get initial values from model class method
        if iv is None:
            initial_value = self.model.get_initial_values()
        else:
            initial_value = iv
        solution = self.model.get_solution(t=t, initial_values=initial_value, parameters=self.parameters,
                                           contact_matrix=contact_mtx)
        return solution

    def _plot_r_eff(self, r_eff: np.ndarray, case: str) -> None:
        # Plot effective reproduction number
        t = np.linspace(0, self.time_plot, 1 + self.time_plot * self.bin_size)
        fig = plt.figure(figsize=(6, 6))
        plt.plot(t, r_eff)
        self._generate_date(fig)
        fig.savefig(os.path.join("./plots", 'r_eff_' + case + '.pdf'))
        plt.show()

    def _plot_dynamics(self, sol: np.ndarray) -> None:
        # Plot daily incidence
        t = np.linspace(0, self.time_plot, self.time_plot * self.bin_size)
        fig = plt.figure(figsize=(6, 6))
        plt.plot(t, np.diff(self.model.get_cumulative(sol)))
        self._generate_date(fig)
        fig.savefig(os.path.join("./plots", 'daily_incidence.pdf'))
        plt.show()

        # Plot cumulative cases
        t = np.linspace(0, self.time_plot, 1 + self.time_plot * self.bin_size)
        fig = plt.figure(figsize=(6, 6))
        plt.plot(t, self.model.get_cumulative(sol))
        self._generate_date(fig)
        fig.savefig(os.path.join("./plots", 'cumulative.pdf'))
        plt.show()

    def _generate_date(self, fig):
        date_bin = 14
        list_of_dates = np.array([
            d.strftime('%m-%d') for d in pd.date_range(start='2020-03-31', periods=self.n_cm + 1)
        ])
        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(np.arange(len(list_of_dates[::date_bin])) * date_bin, list_of_dates[::date_bin])
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)


if __name__ == '__main__':
    os.makedirs("./plots", exist_ok=True)
    simulation = Simulation()
    simulation.run()
