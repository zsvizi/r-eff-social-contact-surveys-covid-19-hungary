import copy
import datetime

import numpy as np

from dataloader import DataLoader, transform_matrix
from model import RostModelHungary
from r0 import R0Generator


class Simulation:
    def __init__(self, **config):
        """
        Constructor initializing class by
        - loading data (contact matrices, model parameters, age distribution)
        - creates model and r0generator objects
        - calculates initial transmission rate
        """
        # ------------- USER-DEFINED PARAMETERS -------------
        # Debug variable
        self.debug = False
        # Time step in contact data
        self.time_step = 1
        # Baseline R0 for uncontrolled epidemic
        self.r0 = 2.2
        # Variable for clarifying contact matrix for baseline beta calculation
        # - None: reference matrix from reference_contact_data
        # - specified tuple of date strings (e.g. ('2020-08-30', '2020-09-06')): specified matrix from contact data
        self.baseline_cm_date = ('2020-08-30', '2020-09-06')  # None / ('2020-08-30', '2020-09-06')
        # Are effective R values calculated?
        self.is_r_eff_calc = False
        # ------------- USER-DEFINED PARAMETERS END -------------

        # Instantiate DataLoader object to load model parameters, age distributions and contact matrices
        self.data = DataLoader(**config)

        # Instantiate dynamical system
        self.model = RostModelHungary(model_data=self.data)

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
        # Number of contact matrices used in the plotting
        self.time_plot = None

        # Member variables for plotting
        self.r_eff_plot = None
        self.sol_plot = None
        self.timestamps = None
        self.repi_r0_list = None

    def run(self) -> None:
        """
        Run simulation and plot results
        :return: None
        """
        # Run simulation
        self.simulate()

    def simulate(self, start_time: str = "2020-03-31", end_time: str = "2021-01-26"
                 # , reference_matrix: str = None,
                 ) -> None:
        """
        Simulate epidemic model and calculates reproduction number
        :return: None
        """
        start_date_delta = 1
        start_date = datetime.datetime.strptime(start_time, '%Y-%m-%d') \
            - datetime.timedelta(days=start_date_delta)
        valid_dates = [date
                       for date in self.data.contact_data.index
                       if start_date <=
                       datetime.datetime.strptime(date[0], "%Y-%m-%d") <=
                       datetime.datetime.strptime(end_time, "%Y-%m-%d")]
        # Define time_plot
        self.time_plot = 1 + len(valid_dates)
        # Get transformed contact matrix (here, we have the reference matrix)
        # Transform means: multiply by age distribution as a row (based on concept of contact matrices from data),
        # then take average of result and transpose of result
        # then divide by the age distribution as a column
        cm_tr = self._get_transformed_cm(cm=self.data.reference_contact_data.iloc[0].to_numpy())
        # Get solution for the first time interval (here, we have the reference matrix)
        solution = self._get_solution(contact_mtx=cm_tr, is_start=True)
        sol_plot = copy.deepcopy(solution)
        # Get effective reproduction numbers for the first time interval
        # R_eff is calculated at each points for which odeint gives values ('bin_size' amount of values for one day)
        r_eff = self._get_r_eff(cm=cm_tr, solution=solution)
        r_eff_plot = copy.deepcopy(r_eff)
        # Variables for handling missing dates
        previous_day = start_date + datetime.timedelta(days=start_date_delta - self.time_step)
        no_missing_dates = 0
        # Piecewise solution of the dynamical model: change contact matrix on basis of n_days (see in constructor)
        for date in valid_dates:
            # Get contact matrix for current date
            cm = self.data.contact_data.loc[date].to_numpy()

            # Transform actual contact matrix data
            cm_tr = self._get_transformed_cm(cm=cm)

            # Get solution for the actual time interval
            solution = self._get_solution(contact_mtx=cm_tr,
                                          iv=solution[-1])
            # Append this solution piece
            sol_plot = np.append(sol_plot, solution[1:], axis=0)

            # Get effective reproduction number for the actual time interval
            r_eff = self._get_r_eff(cm=cm_tr, solution=solution, date=date)
            r_eff_plot = np.append(r_eff_plot, r_eff[1:], axis=0)

            # Handle missing data
            # In the following, we use date[0], since contact matrices are indexed by tuple (start_date, end_date)
            one_day_back = datetime.datetime.strptime(date[0], '%Y-%m-%d') - datetime.timedelta(days=self.time_step)
            if one_day_back != previous_day:
                diff_days = (datetime.datetime.strptime(date[0], '%Y-%m-%d') - previous_day).days
                # Append zeros for missing dates
                for _ in range(1, diff_days):
                    no_missing_dates += 1
                    sol_plot = np.append(sol_plot, np.zeros(solution[1:].shape), axis=0)
                    r_eff_plot = np.append(r_eff_plot, np.zeros(r_eff[1:].shape), axis=0)
                    self.r0_generator.debug_list.append(-1)
            # Update previous day to actual one
            previous_day = datetime.datetime.strptime(date[0], '%Y-%m-%d')

        # Correct self.time_plot by the number of missing dates
        self.time_plot += no_missing_dates

        # Store results
        self.r_eff_plot = r_eff_plot
        # added timestamps for simulation data points, first timestamp 1 day before data timestamps for reference matrix
        self.timestamps = np.concatenate([[self.data.start_ts - 24 * 3600],
                                          np.linspace(self.data.start_ts, self.data.end_ts, len(self.r_eff_plot)-1)])
        self.sol_plot = sol_plot

    def get_repi_r0_list(self) -> None:
        """
        Calculate eigenvalues for matrices from representative query
        :return:
        """
        print("-------- Representative matrices --------")
        print("Baseline beta:", self.parameters["beta"])
        print("For matrix BASELINE eig. val =", self.r0 / self.parameters["beta"],
              "-> baseline r0 =", self.r0)
        print("-----------------------------------------")
        repi_cm_df = self.data.representative_contact_data
        repi_r0_list = []
        for indx in repi_cm_df.index:
            self.is_r_eff_calc = False
            cm = repi_cm_df.loc[indx].to_numpy()
            cm_tr = self._get_transformed_cm(cm=cm)
            solution = self._get_solution(contact_mtx=cm_tr, is_start=True)
            r_eff = self._get_r_eff(cm=cm_tr, solution=solution)[0]
            print("For matrix", indx, "eig. val =", r_eff / self.parameters["beta"], "-> r0 =", r_eff)
            repi_r0_list.append(r_eff)
        repi_r0_list.insert(3, repi_r0_list[3])
        # Store result
        self.repi_r0_list = repi_r0_list

    def _get_initial_beta(self) -> float:
        """
        Calculates transmission rate used in the dynamical model based on the reference matrix
        :return: float, transmission rate for reference matrix
        """
        # Get transformed reference matrix
        cm = self._get_transformed_cm(cm=self._get_baseline_cm())

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

    def _get_baseline_cm(self) -> np.ndarray:
        """
        Returns contact matrix for baseline beta calculation based on user-defined date
        :return: np.ndarray contact matrix from data
        """
        if self.baseline_cm_date is None:
            baseline_cm = self.data.reference_contact_data.iloc[0].to_numpy()
        else:
            baseline_cm = self.data.contact_data.loc[self.baseline_cm_date].to_numpy()
        return baseline_cm

    def _get_transformed_cm(self, cm: np.ndarray) -> np.ndarray:
        """
        Symmetrizes input contact matrix
        :param cm: np.ndarray, input contact matrix as a row vector
        :return: np.ndarray, symmetrized contact matrix in a matrix form
        """
        return transform_matrix(age_data=self.data.age_data,
                                matrix=cm.reshape((self.model.n_age, self.model.n_age)))

    def _get_r_eff(self, cm: np.ndarray, solution: np.ndarray, date: str = None) -> np.ndarray:
        """
        Calculates r_eff values for actual time interval
        :param cm: np.ndarray, actual contact matrix
        :param solution: np.ndarray, solution piece of the model for the actual time interval
        :param date: str date of calculation
        :return: np.ndarray, r_eff values
        """
        susceptibles = self.model.get_comp(solution, self.model.c_idx["s"])
        r_eff = self.parameters["beta"] * self.r0_generator.get_eig_val(contact_mtx=cm,
                                                                        population=self.model.population,
                                                                        susceptibles=susceptibles,
                                                                        date=date,
                                                                        is_effective_calculated=self.is_r_eff_calc)
        return r_eff

    def _get_solution(self, contact_mtx: np.ndarray, iv: np.ndarray = None, is_start: bool = False) -> np.ndarray:
        """
        Solves dynamical model for actual time interval (assumes uniformly divided intervals!)
        :param contact_mtx: np.ndarray, actual contact matrix
        :param iv: np.ndarray, initial value for odeint, mostly end point of the previous solution piece
        :return: np.ndarray, solution piece for this interval
        """
        if is_start:
            time_step = 1
        else:
            time_step = self.time_step
        # Get time interval
        t = np.linspace(0, time_step, 1 + time_step * self.bin_size)
        # For first time interval, get initial values from model class method
        if iv is None:
            initial_value = self.model.get_initial_values()
        else:
            initial_value = iv
        solution = self.model.get_solution(t=t, initial_values=initial_value, parameters=self.parameters,
                                           contact_matrix=contact_mtx)
        return solution
