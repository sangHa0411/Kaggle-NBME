import os
import random
import pandas as pd
from datasets import Dataset, DatasetDict

class Loader :
    def __init__(self, dir_path, validation_ratio, seed) :
        self.df = self.load(dir_path)
        self.dataset = self.parsing(self.df)
        self.validation_ratio = validation_ratio
        self.seed = seed

    def load(self, dir_path) :
        train_df = pd.read_csv(os.path.join(dir_path, 'train.csv'))
        patients_df = pd.read_csv(os.path.join(dir_path, 'patient_notes.csv'))
        features_df = pd.read_csv(os.path.join(dir_path, 'features.csv'))

        train_df = train_df.merge(features_df, on=['feature_num', 'case_num'], how='left')
        train_df = train_df.merge(patients_df, on=['pn_num', 'case_num'], how='left')
        return train_df

    def parsing(self, df) :
        dataset = []
        for i in range(len(df)) :
            row = df.iloc[i]
            pn_history = row['pn_history']
            feature_text = row['feature_text']

            location = row['location']

            if len(location) == 2 :
                info = {'history' : pn_history, 'feature' : feature_text, 'locations' : []}
            else :
                location = location[1:-1]
                locs = location.split(', ')

                loc_parsed = []
                for loc in locs :
                    if ';' in loc :
                        loc_str = loc[1:-1]
                        loc_list = loc_str.split(';')

                        for l in loc_list :
                            start, end = l.split(' ')
                            loc_parsed.append([int(start), int(end)])
                    else :
                        loc_str = loc[1:-1]
                        start, end = loc_str.split(' ')
                        loc_parsed.append([int(start), int(end)])
                
                info = {'history' : pn_history, 'feature' : feature_text, 'locations' : loc_parsed}
            dataset.append(info)
        return dataset

    def get(self,) :
        dset = self.convert_dataset(self.dataset)
        
        ids_list = list(range(len(dset)))
        train_size = int(len(ids_list) * (1-self.validation_ratio))

        train_ids = random.sample(ids_list, train_size)
        validation_ids = list(set(ids_list) - set(train_ids))

        train_dset = dset.select(train_ids)
        validation_dset = dset.select(validation_ids)
        datasets = DatasetDict({'train' : train_dset, 'validation' : validation_dset})
        datasets = datasets.shuffle(self.seed)

        return datasets

    def convert_dataset(self, dataset) :
        l_list, h_list, f_list = [], [], []

        for data in dataset : 
            l_list.append(data['locations'])
            h_list.append(data['history'])
            f_list.append(data['feature'])

        dset = {'locations' : l_list, 'history' : h_list, 'feature' : f_list}
        dset = Dataset.from_dict(dset)
        return dset