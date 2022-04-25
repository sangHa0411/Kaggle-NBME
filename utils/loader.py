import os
import ast 
import pandas as pd

class Loader :
    def __init__(self, dir_path) :
        self.df = self.load(dir_path)

    def load(self, dir_path) :
        train_df = pd.read_csv(os.path.join(dir_path, 'train.csv'))
        patients_df = pd.read_csv(os.path.join(dir_path, 'patient_notes.csv'))
        features_df = pd.read_csv(os.path.join(dir_path, 'features.csv'))

        train_df = train_df.merge(features_df, on=['feature_num', 'case_num'], how='left')
        train_df = train_df.merge(patients_df, on=['pn_num', 'case_num'], how='left')
        
        train_df = self.parsing(train_df)
        return train_df

    def parsing(self, df) :
        df['annotation'] = df['annotation'].apply(ast.literal_eval)
        df['location'] = df['location'].apply(ast.literal_eval)
        df['annotation_length'] = df['annotation'].apply(len)
        return df

