import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv
import re as re
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

"""
Categorical Data Features:
1. Form Name - total of 9 unique values -> form_name
2. Hospital - total of four unique values (float)
3. User Name - 154 values
4. Basic Stage-אבחנה - C P R or NULL
5. Diagnosis data-אבחנה - dates
6. Her2-אבחנה - various data representing positive and negative values.
7. dאבחנ-Histological diagnosis - 41 values const english CAPS
8. Histopatological degree-אבנחה - total of 6 values
9. אבחנה-Ivi -Lymphovascular invasion
10. lymphatic penetration - 5 values
11. M -metastases mark (NTM)-אבחנה - total 6 values
15. Side - Hebrew consts 3 values
16. Surgery date1 - out of 3 dates
17. Surgery date2 - out of 3 dates
18. Surgery date3 - out of 3 dates
19. Surgery name1 - out of 23 names
19. Surgery name2 - out of 18 names
19. Surgery name3 - out of 6 names
20 Surgery sum - no. of surgeries 
21. Surgery before of after-Activity date
22. Surgery before of after-Actual activity
23. id_hushed_internalpatientid
24. T -Tumor mark (TNM) - 22 values
Ordinal Data Features:
1. Age-אבחנה - Age of the patient
2. KI67 protein-אבחנה - percentage
3. Stage - 17 values
4. Tumor depth - 6 values
5. Tumor width - 31 values
6. er - various data representing positive and negative values.
6. pr - various data representing positive and negative values.
REQ
3. Positive nodes - 28 values
12. Margin type-אבחנה - 3 values in hebrew
13. M -lymph nodes mark (NTM)-אבחנה total 21 values
14. Nodes exam - 42 values
"""

HEBREW_AVCHANA = "אבחנה"
HER2POS_KW = ['+', 'חיובי', '1', '2', '0']


class DataProcess:

    def __init__(self):
        self.one_hot = OneHotEncoder()
        self.ordinal_encoder = OrdinalEncoder()
        self.categories = ['her2', 'histopatological_degree',
                           'ivi_lymphovascular_invasion', 'id_hushed_internalpatientid'
            , 'margin_type', 'stage', 'basic_stage']

        self.ordinal = ['age', 'ki67_protein', 'pr', 'er', 'tumor_depth', 'tumor_width', 'surgery_sum',
                        'positive_nodes', 'nodes_exam', 'm_metastases_mark_(tnm)', 't_tumor_mark_(tnm)',
                        'n_lymph_nodes_mark_(tnm)']

        self.feat_to_drop = ['user_name', "surgery_date1", 'surgery_date2', 'surgery_date3', 'surgery_name1',
                             'surgery_name2',
                             'surgery_name3', 'diagnosis_date', 'form_name', 'hospital']
        # lymphatic_penetration, t_tumor_mark_(tnm) m_metastases_mark_(tnm) n_lymph_nodes_mark_(tnm)

    def load_data(self, filename_x: str, filename_y1: str, filename_y2: str) -> (pd.DataFrame, pd.DataFrame):
        X = read_csv(filename_x)
        y1 = read_csv(filename_y1)
        y2 = read_csv(filename_y2)

        X = self.refactor_feature_names(X)
        y1 = self.refactor_feature_names(y1)
        y2 = self.refactor_feature_names(y2)

        print(X.dtypes)
        unified = pd.DataFrame(X)
        unified['labels1'], unified['labels2'] = y1, y2
        unified.drop(columns=self.feat_to_drop)
        unified = self.encode_data(data=unified)

        # for col in self.categories:
        #     if col != 'id_hushed_internalpatientid' and col != 'diagnosis_date':
        #         print(f"column {col} containing unique values: {set(unified[col])}\n\n\n")
        # for col in self.ordinal:
        #     if col != 'id_hushed_internalpatientid' and col != 'diagnosis_date':
        #         print(f"column {col} containing unique values: {set(unified[col])}\n\n\n")

        return unified

    def data_preprocessing(self, data: pd.DataFrame):
        # Clean The Data
        for categorical in self.categories:
            encoder_df = pd.DataFrame(self.one_hot.fit_transform(data[categorical]))
            data.join(encoder_df)
        data.drop(columns=categorical, axis=1, inplace=True)

    def clean_her2(self,df):
        df['her2'] = df['her2'].str.lower()
        # df = df[df['her2'].notna()]
        df = df[df.her2.notnull()]
        df.loc[df['her2'].str.contains('0'), 'her2'] = '0'
        df.loc[df['her2'].str.contains('neg'), 'her2'] = '0'
        df.loc[df['her2'].str.contains('pos'), 'her2'] = '1'
        df.loc[df['her2'].str.contains('-'), 'her2'] = '0'
        df.loc[df['her2'].str.contains('\\+'), 'her2'] = '1'
        # df.loc[df['her2'].str.contains('+'), 'her2'] = '1'
        # df.loc[df['her2'].str.contains('ative'), 'her2'] = '1'
        df.loc[df['her2'].str.contains('eg'), 'her2'] = '0'
        df.loc[df['her2'].str.contains('no'), 'her2'] = '0'
        # df['her2'] = df['her2'].str.replace('eg', '0', regex=True)
        # df['her2'] = df['her2'].str.replace('no', '0', regex=True)
        df.loc[df['her2'].str.contains('^o'), 'her2'] = '0'
        df['her2'] = df['her2'].str.replace('akhah', '0', regex=True)
        df['her2'] = df['her2'].str.replace('akhkh', '0', regex=True)
        df.loc[df['her2'].str.contains('^n'), 'her2'] = '0'
        df.loc[df['her2'].str.contains('^in'), 'her2'] = '0.5'
        df.loc[df['her2'].str.contains('border'), 'her2'] = '0.5'
        df.loc[df['her2'].str.contains('equivocal'), 'her2'] = '0.5'
        df.loc[df['her2'].str.contains('pending'), 'her2'] = '0.5'
        df.loc[df['her2'].str.contains('1lified'), 'her2'] = '1'
        df['her2'] = df['her2'].str.replace('בינוני', '0.5', regex=True)
        df['her2'] = df['her2'].str.replace('שלילי', '0', regex=True)
        df['her2'] = df['her2'].str.replace('חיובי', '1', regex=True)
        df['her2'] = df['her2'].str.replace('3', '1', regex=True)
        # df.loc[df['her2'].str.contains('+'), 'her2'] = 1
        # df.loc[df['her2'].str.contains('1'), 'her2'] = '0'
        df['her2'] = df['her2'].str.replace('amp', '1', regex=True)
        df.loc[df['her2'].str.contains('2'), 'her2'] = '0.5'
        df = df[(df.her2 == '0') | (df.her2 == '1') | (df.her2 == '0.5')]
        her2 = pd.unique(df['her2'])
        return df

    def refactor_feature_names(self, data: pd.DataFrame):
        """
        rename all feature names to be snake_case_names for sake of uniformity
        :return:
        """
        s = '_'
        for name in data.columns:
            # print(f"before {name}")
            original = name
            name = name.replace(HEBREW_AVCHANA, "")
            new = re.split(f'-| |{HEBREW_AVCHANA}', name)
            new = [word.lower() for word in new if len(word) > 0]
            new = s.join(new)
            data.rename(columns={original: new}, inplace=True)
        return data

    def encode_data(self, data: pd.DataFrame):
        # data['her2']=data.iloc["neg" in data['her2']]
        data.replace('נקיים', 0, inplace=True)
        data.replace('ללא', 0, inplace=True)
        data.replace('נגועים', 1, inplace=True)

        data.replace('G4 - Undifferentiated', value=0, inplace=True)
        data.replace('G1 - Well Differentiated', value=3, inplace=True)
        data.replace('G2 - Modereately well differentiated', value=2, inplace=True)
        data.replace('G3 - Poorly differentiated', value=1, inplace=True)
        data = data.loc[data['margin_type'] != 'GX - Grade cannot be assessed']
        data = data.loc[data['margin_type'] != 'Null']

        data['ki67_protein'] = data['ki67_protein'].str.replace(r'\D+-\d+|\d+-\D+', ' ', regex=True)
        data['ki67_protein'] = data['ki67_protein'].str.replace(r'-\D+ |-\D+', ' ', regex=True)
        data['ki67_protein'] = data['ki67_protein'].str.replace(r'\D+', ' ', regex=True)

        data.dropna(subset=['ki67_protein'], inplace=True)

        def ki_split(a):
            return a.split(' ')

        data['ki67_protein'] = list(map(ki_split, np.asarray(data['ki67_protein'].values)))

        def tnm(a):
            return [char for char in a]

        data.dropna(subset=['n_lymph_nodes_mark_(tnm)'], inplace=True)
        data.dropna(subset=['m_metastases_mark_(tnm)'], inplace=True)
        data['m_metastases_mark_(tnm)'] = list(map(tnm, np.asarray(data['m_metastases_mark_(tnm)'].values, dtype=str)))
        #data['n_lymph_nodes_mark_(tnm)'] = list(
            #map(tnm, np.asarray(data['n_lymph_nodes_mark_(tnm)'].values, dtype=str)))

        return data

    def clean_lymph(self, df):
        df['n_lymph_nodes mark_(tnm)'] = df['n_lymph_nodes_mark_(tnm)'].str.lower()
        df = df[df['n_lymph_nodes_mark_(tnm)'].notnull()]
        df = df[df['n_lymph_nodes_mark_(tnm)'] != 'N4']
        df = df[df['n_lymph_nodes_mark_(tnm)'] != 'N1mic']
        df['N1'] = df['n_lymph_nodes_mark_(tnm)'].str.contains('N1')
        df['N2'] = df['n_lymph_nodes_mark_(tnm)'].str.contains('N2')
        df['N3'] = df['n_lymph_nodes_mark_(tnm)'].str.contains('N3')
        uni = df['n_lymph_nodes_mark_(tnm)'].unique()
        return df

    def drop_noisy_data(self, data: pd.DataFrame):
        """
        drop null or corrupted samplings
        :return:
        """
        print(data.isnull().sum())

        data.dropna(inplace=True)



if __name__ == "__main__":
    dp = DataProcess()
    df= dp.load_data('train.feats.csv', 'train.labels.0.csv','train.labels.1.csv')
    df = dp.refactor_feature_names(df)
    df = dp.clean_her2(df)
    df = dp.clean_lymph(df)
    print(df.head)
