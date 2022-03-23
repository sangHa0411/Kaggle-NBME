import os
import re
import ast 
import pandas as pd
from datasets import Dataset

class Loader :
    def __init__(self, dir_path, seed) :
        self.df = self.load(dir_path)
        self.seed = seed

    def process_feature_text(self, text):
        text = re.sub('I-year', '1-year', text)
        text = re.sub('-OR-', " or ", text)
        text = re.sub('-', ' ', text)
        return text

    def clean_spaces(self, txt):
        txt = re.sub('\n', ' ', txt)
        txt = re.sub('\t', ' ', txt)
        txt = re.sub('\r', ' ', txt)
        return txt

    def load(self, dir_path) :
        train_df = pd.read_csv(os.path.join(dir_path, 'train.csv'))
        patients_df = pd.read_csv(os.path.join(dir_path, 'patient_notes.csv'))
        features_df = pd.read_csv(os.path.join(dir_path, 'features.csv'))

        train_df = train_df.merge(features_df, on=['feature_num', 'case_num'], how='left')
        train_df = train_df.merge(patients_df, on=['pn_num', 'case_num'], how='left')
        
        train_df = self.preprocess(train_df)
        train_df = self.parsing(train_df)
        return train_df

    def parsing(self, df) :
        df['annotation'] = df['annotation'].apply(ast.literal_eval)
        df['location'] = df['location'].apply(ast.literal_eval)
        df['annotation_length'] = df['annotation'].apply(len)
        return df

    def preprocess(self, df) :
        df['clean_text'] = df['pn_history'].apply(lambda x: x.strip())
        df['feature_text'] = df['feature_text'].apply(self.process_feature_text)

        df['feature_text'] = df['feature_text'].apply(self.clean_spaces)
        df['clean_text'] = df['clean_text'].apply(self.clean_spaces)
        return df

    def convert(self, df) :
        pn_history, feature_text, annotations, locations = [], [], [], []

        for i in range(len(df)) :
            row = df.iloc[i]
            pn_history.append(row['clean_text'])
            feature_text.append(row['feature_text'])
            annotations.append(row['annotation_length'])
            locations.append(row['location'])
        dset = Dataset.from_dict({'locations' : locations, 'history' : pn_history, 'feature' : feature_text, 'annotation_length' : annotations})
        return dset

    def get(self, ) :
        dset = self.convert(self.df).shuffle(self.seed)
        return dset
