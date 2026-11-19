import pandas as pd

from src.simulation import Simulation


sim_1 = Simulation(contact_data_json='dynmatrix_step_1d_window_7d_v15_kid_masked_all.json')
contact_data = pd.DataFrame(sim_1.data.contact_data_json)

model_storage = {}

sim = Simulation(contact_data_json='dynmatrix_step_1d_window_7d_v15_kid_reduced_all.json')
methods = sim.data.reference_r_eff_data["method"].sort_values().unique()
method_mask = sim.data.reference_r_eff_data["method"] == methods[0]

contact_data["avg_contactnum"] = contact_data.avg_actual_outside_proxy + contact_data.avg_family

sim.date_for_calibration = '2020-09-13'
sim.baseline_cm_date = (sim.date_for_calibration, '2020-09-20')

daterange = pd.date_range(start='2020-03-01', end='2021-03-01', freq='D').map(
    lambda d: str(d.date())).tolist()
