import copy
import os

import matplotlib.pyplot as plt
import numpy as np

from dataloader import DataLoader, transform_matrix
from model import RostModelHungary
from r0 import R0Generator


class Simulation:
    def __init__(self):
        self.data = DataLoader()
        self.model = RostModelHungary(model_data=self.data)

        self.r0 = 2.2
        self.parameters = self.data.model_parameters_data
        self.parameters.update({"susc": np.array([0.5, 0.5, 1, 1, 1, 1, 1, 1])})

        self.r0_generator = R0Generator(param=self.parameters, n_age=self.model.n_age)
        self.parameters.update({"beta": self._get_initial_beta()})

        self.bin_size = 100
        self.n_cm = 13
        self.time_plot = 7 * self.n_cm

    def run(self):
        # Get transformed contact matrix
        cm = self._get_transformed_cm(cm=self.data.contact_data.iloc[0].to_numpy())

        # Get solution for the first time interval
        solution = self._get_solution(contact_mtx=cm)
        sol_plot = copy.deepcopy(solution)

        # Get effective reproduction numbers for the first time interval
        r_eff = self._get_r_eff(cm=cm, solution=solution)
        r_eff_plot = copy.deepcopy(r_eff)

        for cm in self.data.contact_data.iloc[1:self.n_cm].to_numpy():
            # Transform actual contact matrix data
            cm = self._get_transformed_cm(cm=cm)

            # Get solution for the actual time interval
            solution = self._get_solution(contact_mtx=cm,
                                          iv=solution[-1])
            sol_plot = np.append(sol_plot, solution[1:], axis=0)

            # Get effective reproduction number for the actual time interval
            r_eff = self._get_r_eff(cm=cm, solution=solution)
            r_eff_plot = np.append(r_eff_plot, r_eff[1:], axis=0)

        self._plot_dynamics(sol_plot)
        self._plot_r_eff(r_eff_plot)

    def _get_initial_beta(self) -> float:
        # Get transformed contact matrix
        cm = self._get_transformed_cm(cm=self.data.contact_data.iloc[0].to_numpy())

        # Get initial values for susceptibles and population
        population = self.model.population
        initial_values = self.model.get_initial_values().reshape(1, -1)
        susceptibles = self.model.get_comp(initial_values, self.model.c_idx["s"])

        # Get initial eigenvalue of the NGM
        eig_value_0 = self.r0_generator.get_eig_val(contact_mtx=cm,
                                                    population=population,
                                                    susceptibles=susceptibles)[0]
        # Get initial beta from R0
        beta = self.r0 / eig_value_0
        return beta

    def _get_transformed_cm(self, cm):
        return transform_matrix(age_data=self.data.age_data,
                                matrix=cm.reshape((self.model.n_age, self.model.n_age)))

    def _get_r_eff(self, cm, solution):
        susceptibles = self.model.get_comp(solution, self.model.c_idx["s"])
        r_eff = self.parameters["beta"] * self.r0_generator.get_eig_val(contact_mtx=cm,
                                                                        population=self.model.population,
                                                                        susceptibles=susceptibles)
        return r_eff

    def _get_solution(self, contact_mtx: np.ndarray, iv: np.ndarray = None) -> np.ndarray:
        # Get time interval
        t = np.linspace(0, self.time_plot / self.n_cm, 1 + int(self.time_plot * self.bin_size / self.n_cm))
        if iv is None:
            initial_value = self.model.get_initial_values()
        else:
            initial_value = iv
        solution = self.model.get_solution(t=t, initial_values=initial_value, parameters=self.parameters,
                                           contact_matrix=contact_mtx)
        return solution

    def _plot_r_eff(self, r_eff):
        # Plot effective reproduction number
        t = np.linspace(0, self.time_plot, 1 + self.time_plot * self.bin_size)
        fig = plt.figure(figsize=(6, 6))
        plt.plot(t, r_eff)
        fig.savefig(os.path.join("./plots", 'r_eff.pdf'))
        plt.show()

    def _plot_dynamics(self, sol):
        # Plot daily incidence
        t = np.linspace(0, self.time_plot, self.time_plot * self.bin_size)
        fig = plt.figure(figsize=(6, 6))
        plt.plot(t, np.diff(self.model.get_cumulative(sol)))
        fig.savefig(os.path.join("./plots", 'daily_incidence.pdf'))
        plt.show()

        # Plot cumulative cases
        t = np.linspace(0, self.time_plot, 1 + self.time_plot * self.bin_size)
        fig = plt.figure(figsize=(6, 6))
        plt.plot(t, self.model.get_cumulative(sol))
        fig.savefig(os.path.join("./plots", 'cumulative.pdf'))
        plt.show()


if __name__ == '__main__':
    os.makedirs("./plots", exist_ok=True)
    simulation = Simulation()
    simulation.run()
