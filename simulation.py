import copy
import datetime

import numpy as np

from dataloader import DataLoader, transform_matrix
from model import RostModelHungary
from r0 import R0Generator


class Simulation:
    def __init__(self, **config) -> None:
        """
        Constructor initializing class by
        - loading data (contact matrices, model parameters, age distribution)
        - creates model and r0generator objects
        - calculates initial transmission rate
        :return: None
        """
        # ------------- USER-DEFINED PARAMETERS -------------
        # Time step in contact data
        self.time_step = 1
        # Baseline R0 for uncontrolled epidemic
        self.r0 = 1.3
        # Variables for clarifying contact matrix for baseline beta calculation
        # - date for calibration is the selected date (for element of the tuple baseline cm date)
        # - ('2020-01-01', '2020-01-01'): reference matrix from reference_contact_data
        # - other tuple of date strings (e.g. ('2020-08-30', '2020-09-06')): specified matrix from contact data
        self.date_for_calibration = '2020-09-13'
        self.baseline_cm_date = (self.date_for_calibration, '2020-09-20')
        # Are effective R values calculated?
        self.is_r_eff_calc = False
        # Date from effective reproduction number is calculated if is_r_eff_calc = True
        # Currently it is the the date_for_calibration as a timestamp
        self.date_r_eff_calc = datetime.datetime.strptime(self.date_for_calibration, "%Y-%m-%d").timestamp()

        # TEST: added for tesing initial values
        # Is initial value test running?
        self.is_init_value_tested = True
        # Initial R0 for testing initial values
        self.initial_r0 = 2.0
        # Initial ratio of recovereds for testing initial values
        self.ratio_recovered_first_wave = 0.01
        self.init_ratio_recovered = 0.02
        # Date for the initial contact matrix
        self.date_init_cm = '2020-08-30'
        # Flag for choosing between seasonality functions
        self.is_piecewise_linear_used = False
        # ------------- USER-DEFINED PARAMETERS END -------------

        # Instantiate DataLoader object to load model parameters, age distributions and contact matrices
        self.data = DataLoader(**config)

        # Instantiate dynamical system
        self.model = RostModelHungary(population_data=self.data.age_data.flatten())

        # Get model parameters from DataLoader and append susceptibility age vector to the dictionary
        self.parameters = self.data.model_parameters_data
        self.parameters.update({"susc": np.array([0.5, 0.5, 1, 1, 1, 1, 1, 1])})

        # Instantiate R0generator object for calculating effective reproduction numbers
        self.r0_generator = R0Generator(param=self.parameters, n_age=self.model.n_age)

        # Number of points evaluated for a time unit in odeint
        self.bin_size = 10

        # Is R_eff calculated for the current date?
        self.is_r_eff_calc_current_date = False

        # Member variables for plotting
        self.r_eff_plot = None
        self.sol_plot = None
        self.timestamps = None
        self.repi_r0_list = None

        self.init_latent = None
        self.init_infected = None

    def run(self) -> None:
        """
        Run simulation and plot results
        :return: None
        """
        # Run simulation
        self.simulate()

    def simulate(self, start_time: str = "2020-03-31",
                 end_time: str = "2021-01-26",
                 c: float = 0.3,
                 **config) -> None:
        """
        Simulate epidemic model and calculates reproduction number
        Assumptions:
        - contact matrix is available for all days between start_time and end_time
        - parameter 'beta' does NOT contain seasonality effect (it has to be adjusted, if it is used)
        :param start_time: str, start date given in "%Y-%m-%d" format
        :param end_time: str, end date given in "%Y-%m-%d" format
        :param c: float, seasonality scale
        :return: None
        """
        # Local wrapper for seasonality cos function
        def seasonality_cos_wrap(t: float):
            return seasonality_cos(t=t, c0=c, origin='2020-02-01')

        # Local wrapper for seasonality piecewise linear function
        def seasonality_piecewise_linear_wrap(t: float):
            param_dict = dict()
            param_dict['low_seasonality'] = 0.6
            param_dict['high_seasonality'] = 1.1
            param_dict['lin_increase_duration'] = 31
            param_dict['lin_decrease_duration'] = 30
            param_dict['date_max_last'] = '2019-03-01'
            param_dict['date_min_last'] = '2019-09-05'
            return seasonality_piecewise_linear(t=t, param_dict=param_dict)

        # Choose seasonality function for the simulation
        seasonality_func = seasonality_piecewise_linear_wrap \
            if self.is_piecewise_linear_used \
            else seasonality_cos_wrap

        # Transform start and end time to timestamp
        start_ts = datetime.datetime.strptime(start_time, '%Y-%m-%d').timestamp()
        end_ts = datetime.datetime.strptime(end_time, '%Y-%m-%d').timestamp()

        # Update data if config argument is used
        if len(list(config.keys())) > 0:
            self.data = DataLoader(**config)

        # Calculate initial transmission rate (beta) based on reference matrix and self.r0
        # Important: since R0 = beta * spectral_radius(NGM) * seasonality
        # and method calculating initial beta returns R0 / spectral_radius(NGM)
        # here we have to divide by the seasonality factor
        baseline_date_ts = datetime.datetime.strptime(self.date_for_calibration, '%Y-%m-%d').timestamp()
        self.parameters.update(
            {"beta": self._get_initial_beta() / seasonality_func(t=baseline_date_ts)})

        # Add one day for reference
        start_date_delta = 1
        start_date = datetime.datetime.strptime(start_time, '%Y-%m-%d') \
            - datetime.timedelta(days=start_date_delta)
        start_date_ts = start_date.timestamp()

        # Generate valid dates between start and end date
        valid_dates = [date
                       for date in self.data.contact_data.index
                       if start_date <=
                       datetime.datetime.strptime(date[0], "%Y-%m-%d") <=
                       datetime.datetime.strptime(end_time, "%Y-%m-%d")]

        # Get 0th day matrix
        # - matrix from reference file, if simulation starting time (=start_date) = date of first measured matrix
        # - matrix from day before start_date, if start_date is later
        zeroth_day_index = (start_date.strftime("%Y-%m-%d"),
                            (start_date + datetime.timedelta(days=7)).strftime("%Y-%m-%d"))
        zeroth_day_matrix = \
            self.data.reference_contact_data.iloc[0].to_numpy() \
            if start_time == '2020-03-31' \
            else self.data.contact_data.loc[zeroth_day_index].to_numpy()

        # Get transformed contact matrix (here, we have the reference matrix)
        cm_tr = self.get_transformed_cm(cm=zeroth_day_matrix)

        # Get solution for the first time interval (here, we have the reference matrix)
        solution = self._get_solution(contact_mtx=cm_tr, is_start=True,
                                      season_factor=seasonality_func(t=start_date_ts))
        sol_plot = copy.deepcopy(solution)

        # Get effective reproduction numbers for the first time interval
        # R_eff is calculated at each points for which odeint gives values ('bin_size' amount of values for one day)
        r_eff = self._get_r_eff(cm=cm_tr, solution=solution,
                                season_factor=seasonality_func(t=start_date_ts))
        r_eff_plot = copy.deepcopy(r_eff)

        # Piecewise solution of the dynamical model
        for date in valid_dates:
            # Convert date to timestamp
            date_ts = datetime.datetime.strptime(date[0], '%Y-%m-%d').timestamp()

            # Get contact matrix for current date
            cm = self.data.contact_data.loc[date].to_numpy()

            # Transform actual contact matrix data
            cm_tr = self.get_transformed_cm(cm=cm)

            # Get solution for the actual time interval
            init_val = \
                solution[-1] \
                if date[0] != self.date_for_calibration \
                else None  # TEST: added for testing init values
            solution = self._get_solution(contact_mtx=cm_tr, iv=init_val,
                                          season_factor=seasonality_func(t=date_ts))

            # Append this piece of solution
            sol_plot = np.append(sol_plot, solution[1:], axis=0)

            # Set flag for calculating R_eff for the current date, if is_r_eff_calc = True
            if self.is_r_eff_calc:
                if date_ts < self.date_r_eff_calc:
                    self.is_r_eff_calc_current_date = False
                else:
                    self.is_r_eff_calc_current_date = True

            # Get effective reproduction number for the actual time interval
            r_eff = self._get_r_eff(cm=cm_tr, solution=solution,
                                    season_factor=seasonality_func(t=date_ts))
            r_eff_plot = np.append(r_eff_plot, r_eff[1:], axis=0)

        # Store results
        self.r_eff_plot = r_eff_plot
        self.sol_plot = sol_plot
        # Add timestamps for simulation data points,
        # first timestamp 1 day before data timestamps for reference matrix
        self.timestamps = np.concatenate([[start_date.timestamp()],
                                          np.linspace(start_ts, end_ts, len(self.r_eff_plot) - 1)])

    def _get_initial_beta(self) -> float:
        """
        Calculates transmission rate used in the dynamical model based on the reference matrix
        Assumption: there is no seasonality effect, i.e. R0 = beta * spectral_radius(NGM)
        :return: float, transmission rate for reference matrix
        """
        # Get transformed reference matrix
        cm = self.get_transformed_cm(cm=self._get_baseline_cm())

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
        if self.baseline_cm_date == ('2020-01-01', '2020-01-01'):
            baseline_cm = self.data.reference_contact_data.iloc[0].to_numpy()
        else:
            baseline_cm = self.data.contact_data.loc[self.baseline_cm_date].to_numpy()
        return baseline_cm

    def get_transformed_cm(self, cm: np.ndarray) -> np.ndarray:
        """
        Transforms input contact matrix:
        - multiply by age distribution as a row (based on concept of contact matrices from data),
        - then take average of result and transpose of result
        - then divide by the age distribution as a column
        :param cm: np.ndarray, input contact matrix as a row vector
        :return: np.ndarray, symmetrized contact matrix in a matrix form
        """
        return transform_matrix(age_data=self.data.age_data,
                                matrix=cm.reshape((self.model.n_age, self.model.n_age)))

    def _get_r_eff(self, cm: np.ndarray,
                   solution: np.ndarray,
                   season_factor: float = 1.0) -> np.ndarray:
        """
        Calculates r_eff values for actual time interval
        :param cm: np.ndarray, actual contact matrix
        :param solution: np.ndarray, solution piece of the model for the actual time interval
        :param season_factor: float, seasonality factor
        :return: np.ndarray, r_eff values
        """
        susceptibles = self.model.get_comp(solution, self.model.c_idx["s"])
        r_eff = self.parameters["beta"] * \
            self.r0_generator.get_eig_val(contact_mtx=cm,
                                          population=self.model.population,
                                          susceptibles=susceptibles,
                                          is_effective_calculated=self.is_r_eff_calc_current_date)
        # Result is adjusted by the seasonality factor (since beta does not contain this effect)
        r_eff *= season_factor
        return r_eff

    def _get_solution(self, contact_mtx: np.ndarray,
                      iv: np.ndarray = None,
                      is_start: bool = False,
                      season_factor: float = 1.0) -> np.ndarray:
        """
        Solves dynamical model for actual time interval (assumes uniformly divided intervals!)
        :param contact_mtx: np.ndarray, actual contact matrix
        :param iv: np.ndarray, initial value for odeint, mostly end point of the previous solution piece
        :param is_start: bool, flag for calculating solution on the first time interval
        :param season_factor: float, seasonality factor
        :return: np.ndarray, solution piece for this interval
        """
        if is_start:
            time_step = 1
        else:
            time_step = self.time_step
        # Get time interval
        t = np.linspace(0, time_step, 1 + time_step * self.bin_size)
        initial_value = self.get_initial_value(iv=iv, season_factor=season_factor)
        # Beta is adjusted by the seasonality factor here
        self.parameters["beta"] *= season_factor
        solution = self.model.get_solution(t=t, initial_values=initial_value, parameters=self.parameters,
                                           contact_matrix=contact_mtx)
        self.parameters["beta"] /= season_factor
        return solution

    def get_initial_value(self, iv: np.ndarray, season_factor: float) -> np.ndarray:
        """
        Get initial value for solving model (from the start/piecewise)
        :param iv: np.ndarray the initial value for solving model
        :param season_factor: float, seasonality factor at the actual time
        :return: np.ndarray initial value
        """
        # TEST: added for testing init values (whole TRUE branch)
        if self.is_init_value_tested:
            if iv is None:
                # Scale and rescale beta by seasonality, since beta does not contain this effect
                self.parameters["beta"] *= season_factor * (self.initial_r0 / self.r0)
                initial_value = calculate_initial_value(obj=self)
                self.init_latent = np.sum(initial_value[self.model.n_age:3 * self.model.n_age])
                self.init_infected = np.sum(initial_value[3 * self.model.n_age:-3 * self.model.n_age])
                self.parameters["beta"] /= season_factor * (self.initial_r0 / self.r0)

                # Save the calculated initial vector
                np.savetxt("./data/initial_value_" +
                           self.date_for_calibration + "_" +
                           str(self.initial_r0) + "_" +
                           str(self.init_ratio_recovered) +
                           ".csv",
                           X=np.asarray(initial_value),
                           delimiter=";")
            else:
                initial_value = iv
        else:
            # For first time interval, get initial values from model class method
            if iv is None:
                initial_value = self.data.initial_value
            else:
                initial_value = iv
        return initial_value


def seasonality_cos(t: float, c0: float = 0.3, origin: str = '2020-02-01') -> float:
    """
    Theoretical cosine function for simulating seasonality of epidemic spread.
    It is assumed, that efficiency of the spread is larger during winter time, less during summer time.
    The period of the function is 366 days.
    :param origin: str, date from elapsed time is measured
    :param t: float, timestamp of the date
    :param c0: float, magnitude of the seasonality effect
    :return: float, seasonality factor
    """
    d = (t - datetime.datetime.strptime(origin, '%Y-%m-%d').timestamp()) / (24 * 3600)
    return 0.5 * c0 * np.cos(2 * (np.pi * d / 366)) + 1 - 0.5 * c0


def seasonality_piecewise_linear(t: float,
                                 param_dict: dict) -> float:
    """
    Theoretical piecewise linear function for simulating seasonality of epidemic spread.
    It is assumed, that efficiency of the spread is larger during winter time, less during summer time.
    The period of the function is 366 days.
    :param param_dict: dict, contains parameters of the seasonality function
    :param t: float, actual date in timestamp
    :return: float, seasonality value
    """
    low_seasonality = param_dict['low_seasonality']
    high_seasonality = param_dict['high_seasonality']
    lin_increase_duration = param_dict['lin_increase_duration']
    lin_decrease_duration = param_dict['lin_decrease_duration']

    # date when seasonality starts dropping from high value (end of wintertime)
    date_min_last = param_dict['date_min_last']
    # date when seasonality starts dropping from high value (end of wintertime)
    date_max_last = param_dict['date_max_last']

    date_max_last_ts = datetime.datetime.strptime(date_max_last, '%Y-%m-%d').timestamp()
    date_min_last_ts = datetime.datetime.strptime(date_min_last, '%Y-%m-%d').timestamp()
    # Number of days passed between last day of max and min
    diff_max_min = (date_min_last_ts - date_max_last_ts) // (24 * 3600)
    # Number of days passed since initial day of dropping started
    days_from_drop = (t - date_max_last_ts) // (24 * 3600)
    act_time = days_from_drop % 366

    # Piecewise linear, 366-periodic seasonality function
    # - linear decrease on [0, t1]
    # - constant low value on [t1, t2]
    # - linear increase on [t2, t3]
    # - constant high value on [t3, 366]
    if act_time < lin_increase_duration:
        m = (low_seasonality - high_seasonality) / lin_increase_duration
        seas_value = high_seasonality + m * act_time
    elif lin_increase_duration <= act_time < diff_max_min:
        seas_value = low_seasonality
    elif diff_max_min <= act_time < (diff_max_min + lin_decrease_duration):
        m = (high_seasonality - low_seasonality) / lin_decrease_duration
        seas_value = low_seasonality + m * (act_time - diff_max_min)
    else:
        seas_value = high_seasonality

    return seas_value


def calculate_initial_value(obj: Simulation) -> np.ndarray:
    """
    Calculate initial value (for testing purposes)
    :param obj: Simulation, actual Simulation object
    :return: np.ndarray initial value array
    """
    # Get initial values with almost fully susceptible population
    init_val = obj.model.get_initial_values()
    # Put specified ratio of susceptibles to recovered,
    # where ratio comes from the first epidemic wave
    idx_s_age_struct = obj.model.c_idx["s"] * obj.model.n_age
    idx_r_age_struct = obj.model.c_idx["r"] * obj.model.n_age
    init_val[idx_s_age_struct:(idx_s_age_struct + obj.model.n_age)] -= \
        obj.ratio_recovered_first_wave * obj.data.age_data
    init_val[idx_r_age_struct:(idx_r_age_struct + obj.model.n_age)] += \
        obj.ratio_recovered_first_wave * obj.data.age_data
    # Time vector for the calculations
    tt = np.linspace(0, 400, 1 + 400 * obj.bin_size)
    # Get contact matrix for current date
    cm = obj.data.contact_data.loc[obj.date_init_cm].to_numpy()
    cm_tr = obj.get_transformed_cm(cm=cm)
    # Get solution starting from almost fully susceptible population
    sol = obj.model.get_solution(t=tt, initial_values=init_val,
                                 parameters=obj.parameters,
                                 contact_matrix=cm_tr)
    # Get time series of aggregated recovered population
    sol_rec = obj.model.aggregate_by_age(sol, obj.model.c_idx["r"])
    # Get time point, where sol_rec / original_population reaches a threshold ratio
    normalized_recovered = sol_rec / np.sum(obj.data.age_data)
    is_rec_ratio_less_than_init_ratio = \
        normalized_recovered > obj.init_ratio_recovered
    # Get the state from the solution vector at time point,
    # where sol_rec / original_population reached a threshold ratio
    init_value = sol[is_rec_ratio_less_than_init_ratio][0].flatten()
    return init_value
