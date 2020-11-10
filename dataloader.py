import json
import os

import numpy as np
import pandas as pd
import xlrd


PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))


def transform_matrix(age_data, matrix: np.ndarray):
    age_distribution = age_data.reshape((-1, 1))

    matrix_1 = matrix * age_distribution
    output = (matrix_1 + matrix_1.T) / (2 * age_distribution)
    return output


class DataLoader:
    def __init__(self):
        self._model_parameters_data_file = os.path.join(PROJECT_PATH, "data", "model_parameters.json")
        self._contact_data_file = os.path.join(PROJECT_PATH,
                                               "contact_matrix", "results", "dynmatrix_step_1d_window_7d.csv")
        self._reference_contact_file = os.path.join(PROJECT_PATH,
                                                    "contact_matrix", "results",
                                                    "RefWeekFMtxDyn_t_fmtx8x8_wds30_wsh1.csv")
        self._age_data_file = os.path.join(PROJECT_PATH, "data", "age_distribution.xls")

        self._get_age_data()
        self._get_model_parameters_data()
        self._get_contact_mtx()
        # self._get_reference_contact_mtx()

    def _get_age_data(self):
        wb = xlrd.open_workbook(self._age_data_file)
        sheet = wb.sheet_by_index(0)
        datalist = np.array([sheet.row_values(i) for i in range(0, sheet.nrows)])
        wb.unload_sheet(0)
        ages = [0, 5, 15, 30, 45, 60, 70, 80, len(datalist)]
        self.age_data = np.array(list(map(sum, [[datalist[x] for x in range(ages[idx], ages[idx+1])]
                                                for idx in range(len(ages)-1)]))).flatten()

    def _get_model_parameters_data(self):
        # Load model parameters
        with open(self._model_parameters_data_file) as f:
            parameters = json.load(f)
        self.model_parameters_data = dict()
        for param in parameters.keys():
            param_value = parameters[param]["value"]
            if isinstance(param_value, list):
                self.model_parameters_data.update({param: np.array(param_value)})
            else:
                self.model_parameters_data.update({param: param_value})

    def _get_contact_mtx(self):
        data = pd.read_csv(self._contact_data_file, delimiter=',|:',
                           names=['c_' + str(i) + str(j) for i in range(8) for j in range(8)], index_col=0)

        self.contact_data = data

    def _get_reference_contact_mtx(self):
        data = pd.read_csv(self._reference_contact_file, delimiter=',|:',
                           names=['c_' + str(i) + str(j) for i in range(8) for j in range(8)], index_col=0)

        self.reference_contact_data = data
