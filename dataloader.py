from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
import xlrd


PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))


def transform_matrix(age_data, matrix: np.ndarray):
    age_distribution = age_data.reshape((1, -1))

    matrix_1 = matrix * age_distribution
    output = (matrix_1 + matrix_1.T) / (2 * age_distribution.reshape((-1, 1)))
    return output


class DataLoader:
    def __init__(self, **config):
        # Model related data file paths
        self._model_parameters_data_file = os.path.join(PROJECT_PATH, "data", "model_parameters.json")
        self._age_data_file = os.path.join(PROJECT_PATH, "data", "age_distribution.xls")
        self._reference_r0_data_file = os.path.join(PROJECT_PATH, "data", "ferenci_r0.csv")

        # Contact matrices
        self._contact_data_json = os.path.join(PROJECT_PATH,
                                               "contact_matrix", "results",
                                               "dynmatrix_step_1d_window_7d_v10_kid_reduced_all.json")
        self._reference_contact_file = os.path.join(PROJECT_PATH,
                                                    "contact_matrix", "results",
                                                    "online_reference.csv")
        self._representative_contact_file = os.path.join(PROJECT_PATH,
                                                         "contact_matrix", "results",
                                                         "Repr_SumWDKFMtx_weightnorm.csv")

        # Load model parameters
        self._get_model_parameters_data()
        self._get_age_data()
        # Load contact data JSON
        self._get_contact_data_json()
        # Load contact matrices
        self._get_online_survey_data()
        self._get_representative_contact_mtx()
        self._get_reference_contact_mtx()
        # Load reference R0 data
        self._get_reference_r0_data()

        # Overload specified data members, if optional arguments in constructor are used
        self._contact_data_file = None
        if "contact_data_file" in config:
            contact_data_file = str(config.get("contact_data_file"))
            self._contact_data_file = os.path.join(PROJECT_PATH,
                                                   "contact_matrix", "results",
                                                   contact_data_file)
            self._get_contact_mtx()

        if "contact_num_data_file" in config:
            contact_num_data_file = str(config.get("contact_num_data_file"))
            self._contact_num_data_file = os.path.join(PROJECT_PATH,
                                                       "contact_matrix", "results",
                                                       contact_num_data_file)
            self._get_contact_num_data()

    def get_contact_data_filename(self):
        return self._contact_data_file.split('/')[-1].split('.')[0]

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

    def _get_contact_data_json(self):
        with open(self._contact_data_json) as f:
            content = json.load(f)
        self.contact_data_json = content

    def _get_online_survey_data(self):
        contact_mtx_data = []
        date_list = []
        timestamps = []
        contact_num_data = []
        for day_data in self.contact_data_json:
            contact_mtx_data.append(np.array(day_data['contact_matrix']).flatten())
            date_list.append((day_data['start_date'], day_data['end_date']))
            timestamps.append((day_data['start_ts'], day_data['end_ts']))
            contact_num_data.append(np.array([day_data['start_ts'],
                                              day_data['end_ts'],
                                              day_data['avg_actual_outside_proxy'],
                                              day_data['avg_actual_inside_proxy'],
                                              day_data['avg_family'],
                                              day_data['avg_masking']
                                              ]))

        contact_matrices = pd.DataFrame(data=np.array(contact_mtx_data))
        contact_matrices.index = pd.MultiIndex.from_tuples(date_list)
        contact_num = pd.DataFrame(data=np.array(contact_num_data),
                                   columns=['start', 'end', 'outside', 'inside', 'family', 'mask_percentage']
                                   ).fillna(value=np.nan)
        self.contact_data = contact_matrices
        self.start_ts = timestamps[0][0]
        self.end_ts = timestamps[-1][-1]
        self.contact_num_data = contact_num

    def _get_reference_contact_mtx(self):
        data = pd.read_csv(self._reference_contact_file, delimiter=',|:', engine='python',
                           names=['c_' + str(i) + str(j) for i in range(8) for j in range(8)], index_col=0)

        self.reference_contact_data = data

    def _get_representative_contact_mtx(self):
        data = pd.read_csv(self._representative_contact_file, delimiter=',|:', engine='python',
                           names=['c_' + str(i) + str(j) for i in range(8) for j in range(8)], index_col=0)
        data.fillna(0, inplace=True)
        self.representative_contact_data = data

    def _get_reference_r0_data(self):
        # data from the webpage of Ferenci Tamas
        df = pd.read_csv(self._reference_r0_data_file, header=None)

        df.columns = ['method', 'date', 'r0', 'ci']

        df['ci_lower'] = df['ci'].map(lambda s: float(s.split('-')[0]))
        df['ci_upper'] = df['ci'].map(lambda s: float(s.split('-')[1]))

        df['datetime'] = df['date'].map(lambda d: datetime.strptime(d, '%m/%d/%Y'))
        df['ts'] = df['datetime'].map(lambda d: d.timestamp())
        self.reference_r0_data = df

    def _get_contact_mtx(self):
        data = pd.read_csv(self._contact_data_file, delimiter=',|:', engine='python',
                           names=['c_' + str(i) + str(j) for i in range(8) for j in range(8)], index_col=0)

        def start_date(x):
            return datetime.utcfromtimestamp(int(str(x).split('-')[0])).strftime('%Y-%m-%d')

        def end_date(x):
            return datetime.utcfromtimestamp(int(str(x).split('-')[1])).strftime('%Y-%m-%d')

        data.index = pd.MultiIndex.from_tuples([(start_date(x), end_date(x)) for x in data.index])
        self.contact_data = data
        self.start_ts = datetime.strptime(data.index[0][0], '%Y-%m-%d').timestamp()
        self.end_ts = datetime.strptime(data.index[-1][0], '%Y-%m-%d').timestamp()

    def _get_contact_num_data(self):
        data = pd.read_csv(self._contact_num_data_file, header=None, sep="-|:|,", engine='python')\
            .rename({0: 'start', 1: 'end', 2: 'outside', 3: 'inside', 4: 'family', 5: 'mask_percentage'}, axis=1)
        self.contact_num_data = data
