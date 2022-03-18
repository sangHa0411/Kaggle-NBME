import numpy as np

class Encoder :
    def __init__(self, tokenizer, max_length, label_pad_token_id=-100) :
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

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

            sep_token_index = input_ids.index(self.tokenizer.sep_token_id)
            labels = np.zeros(len(input_ids)).astype('int')
            labels[sep_token_index:] = self.label_pad_token_id

            if len(locations) > 0 :                
                for loc in locations :
                    token_start_index = 1
                    token_end_index = sep_token_index - 1

                    org_start, org_end = loc
                    if mapping[i][token_start_index][0] <= org_start and org_end <= mapping[i][token_end_index][1] :

                        while(token_start_index < len(mapping[i]) and mapping[i][token_start_index][0] <= org_start) :
                            token_start_index += 1

                        while(token_end_index >= token_start_index and org_end < mapping[i][token_end_index][1]) :
                            token_end_index -= 1

                        labels[token_start_index-1:token_end_index+1] = 1
            
            label_dset.append(labels)
            
        model_inputs['labels'] = label_dset
        return model_inputs