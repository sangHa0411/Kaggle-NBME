import os
import re
import ast 
import random
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm

class Loader :
    def __init__(self, dir_path, seed) :
        self.df = self.load(dir_path)
        self.seed = seed

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

    def get(self,) :
        dset = self.convert(self.df).shuffle(self.seed)
        return dset

    def split(self, eval_ratio=0.2) :
        df = self.df
        docs = list(df['clean_text'].unique())

        eval_ids = []
        for doc in tqdm(docs) :
            sub_df = df[df['clean_text'] == doc]
            id_list = list(sub_df['id'])

            size = len(sub_df)
            eval_size = int(size * eval_ratio)
            eval_ids.extend(random.sample(id_list, eval_size))

        flags = []
        for i in tqdm(range(len(df))) :
            if df.iloc[i]['id'] in eval_ids :
                flags.append(True)
            else :
                flags.append(False)

        eval_df = df[flags]
        eval_dset = self.convert(eval_df.reset_index(drop=True)).shuffle(self.seed)

        flags = list(np.array(flags) == False)
        train_df = df[flags]
        train_dset = self.convert(train_df.reset_index(drop=True)).shuffle(self.seed)

        dset = DatasetDict({'train' : train_dset, 'validation' : eval_dset})
        return dset
