import numpy as np

class Preprocessor :
    def __init__(self, tokenizer, max_length) :
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, dataset) :
        size = len(dataset['history'])

        history_list = dataset['history']
        feature_list = dataset['feature']

        inputs = [history_list[i] + self.tokenizer.sep_token + feature_list[i] for i in range(size)]       
        model_inputs = self.tokenizer(inputs,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            max_length=self.max_length,
            truncation=True,
        )

        label_dset = []
        mapping = model_inputs.pop("offset_mapping")
        for i in range(size) :
            locations = dataset['locations'][i]
            input_ids = model_inputs['input_ids'][i]

            if len(locations) == 0 :
                labels = np.zeros(len(input_ids)).astype('int')
            else :
                labels = []
                for loc in locations :
                    token_start_index = 1
                    token_end_index = input_ids.index(self.tokenizer.sep_token_id) - 1

                    org_start, org_end = loc
                    if mapping[i][token_start_index][0] <= org_start and org_end <= mapping[i][token_end_index][1] :

                        while(token_start_index < len(mapping[i]) and mapping[i][token_start_index][0] <= org_start) :
                            token_start_index += 1

                        while(token_end_index >= token_start_index and org_end < mapping[i][token_end_index][1]) :
                            token_end_index -= 1

                    label_vector = np.zeros(len(input_ids))
                    label_vector[token_start_index-1:token_end_index+1] = 1
                    labels.append(label_vector)

                labels = np.sum(labels, axis=0)
                labels = list(np.where(labels>0.0, 1, 0))
            label_dset.append(labels)
            
        model_inputs['labels'] = label_dset
        return model_inputs
