import copy

import matplotlib.pyplot as plt
import numpy as np

from dataloader import DataLoader, transform_matrix
from model import RostModelHungary


class Simulation:
    def __init__(self):
        self.data = DataLoader()
        self.model = RostModelHungary(model_data=self.data)

        self.parameters = self.data.model_parameters_data
        self.parameters.update({"beta": 0.31975})
        self.parameters.update({"susc": np.array([0.5, 0.5, 1, 1, 1, 1, 1, 1])})

        self.time_plot = 100
        self.bin_size = 100
        self.n_cm = 10

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

    def get_solution(self, contact_mtx, iv=None):
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
