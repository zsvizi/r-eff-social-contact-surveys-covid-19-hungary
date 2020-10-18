import copy

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
        self.parameters.update({"beta": self.get_initial_beta()})

        self.bin_size = 100
        self.n_cm = 10
        self.time_plot = 7 * self.n_cm

    def get_initial_beta(self) -> float:
        cm = self.data.contact_data.iloc[0].to_numpy()
        initial_values = self.model.get_initial_values().reshape(1, -1)

        population = self.model.population
        susceptibles = self.model.get_comp(initial_values, self.model.c_idx["s"])

        contact_matrix = transform_matrix(age_data=self.data.age_data,
                                          matrix=cm.reshape((self.model.n_age, self.model.n_age)))
        eig_value_0 = self.r0_generator.get_eig_val(contact_mtx=contact_matrix,
                                                    population=population,
                                                    susceptibles=susceptibles)[0]
        beta = self.r0 / eig_value_0
        print(beta)
        return beta

    def run(self):
        cm = self.data.contact_data.iloc[0].to_numpy()
        solution = self.get_solution(contact_mtx=cm)
        sol_plot = copy.deepcopy(solution)

        for cm in self.data.contact_data.iloc[1:self.n_cm].to_numpy():
            solution = self.get_solution(contact_mtx=cm, iv=solution[-1])
            sol_plot = np.append(sol_plot, solution[1:], axis=0)

        t = np.linspace(0, self.time_plot, self.time_plot * self.bin_size)
        plt.plot(t, np.diff(self.model.get_cumulative(sol_plot)))
        plt.show()

        t = np.linspace(0, self.time_plot, 1 + self.time_plot * self.bin_size)
        plt.plot(t, self.model.get_cumulative(sol_plot))
        plt.show()

    def get_solution(self, contact_mtx: np.ndarray, iv: np.ndarray = None) -> np.ndarray:
        t = np.linspace(0, self.time_plot / self.n_cm, 1 + int(self.time_plot * self.bin_size / self.n_cm))
        if iv is None:
            initial_value = self.model.get_initial_values()
        else:
            initial_value = iv
        contact_matrix = transform_matrix(age_data=self.data.age_data,
                                          matrix=contact_mtx.reshape((self.model.n_age, self.model.n_age)))
        solution = self.model.get_solution(t=t, initial_values=initial_value, parameters=self.parameters,
                                           contact_matrix=contact_matrix)
        return solution


if __name__ == '__main__':
    simulation = Simulation()
    simulation.run()
