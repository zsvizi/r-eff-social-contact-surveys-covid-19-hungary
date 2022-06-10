import datetime

import numpy as np


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
    return 0.5 * c0 * np.cos(2 * np.pi * d / 366) + (1.0 - 0.5 * c0)


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
    if act_time < lin_decrease_duration:
        m = (low_seasonality - high_seasonality) / lin_decrease_duration
        seas_value = high_seasonality + m * act_time
    elif lin_decrease_duration <= act_time < diff_max_min:
        seas_value = low_seasonality
    elif diff_max_min <= act_time < (diff_max_min + lin_increase_duration):
        m = (high_seasonality - low_seasonality) / lin_increase_duration
        seas_value = low_seasonality + m * (act_time - diff_max_min)
    else:
        seas_value = high_seasonality

    return seas_value


def seasonality_truncated_cos(t: float, c0: float, origin: str,
                              trunc_val: float) -> float:
    """
    Theoretical truncated cosine seasonality function for simulating seasonality of epidemic spread.
    It is assumed, that efficiency of the spread is larger during winter time, less during summer time.
    The period of the function is 366 days.
    :param t: float, actual date in timestamp
    :param c0: float, seasonality factor for original cosine seasonality
    :param origin: str, date string for original cosine seasonality
    :param trunc_val: float, truncation value
    :return: float, seasonality value
    """
    # Calculate elapsed time from beginning of the year in dates
    jan_1 = '2019-01-01'
    d = (t - datetime.datetime.strptime(jan_1, '%Y-%m-%d').timestamp()) / (24 * 3600)
    # Calculate elapsed time of seasonality peak from beginning of the year
    d_peak = (datetime.datetime.strptime(origin, '%Y-%m-%d').timestamp() -
              datetime.datetime.strptime(jan_1, '%Y-%m-%d').timestamp()
              ) / (24 * 3600) % 366

    # Local function for calculating cosine seasonality based in input day values
    def seasonality_cos_d(c: float, day: float, day_peak: float):
        return 0.5 * c0 * np.cos(2 * np.pi * (day - day_peak) / 366) + (1.0 - 0.5 * c)

    # Transform actual day to [0, 365] and apply mirroring for proper truncation
    d_mod = d % 366
    if d_mod > 366 / 2:
        d_mod = 366 - d_mod
    # Calculate normed seasonality value (norming by peak size)
    normed_value = seasonality_cos_d(c=c0, day=d_mod, day_peak=d_peak) / \
        seasonality_cos_d(c=c0, day=d_peak, day_peak=d_peak)
    # Apply truncation
    calc_val = min(max(normed_value, trunc_val), 1.0)
    # In interval [0, d_peak], the seasonality value is constant and it is at its maximum
    if d_mod < d_peak:
        output = 1.0
    else:
        output = calc_val
    return output
