import numpy as np

class Encoder :
    def __init__(self, tokenizer, max_length, label_pad_token_id=-100) :
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, dataset) :
        model_inputs = self.tokenizer(dataset['feature'],
            dataset['history'],
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            max_length=self.max_length,
            truncation='only_second',
        )

        label_dset = []

        size = len(dataset['feature'])
        mapping = model_inputs.pop("offset_mapping")
        for i in range(size) :
            locations = dataset['locations'][i]
            input_ids = model_inputs['input_ids'][i]
            sequence_ids = model_inputs.sequence_ids(i)

            start_token, end_token = self.get_positions(sequence_ids)
            labels = np.zeros(len(input_ids)).astype('int')
            labels[end_token:] = self.label_pad_token_id
            labels[:start_token] = self.label_pad_token_id
            
            if len(locations) > 0 :
                for loc in locations :
                    token_start_index = start_token
                    token_end_index = end_token - 1
                    
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

    def get_positions(self, vector) :
        start_token = 1
        end_token = len(vector) - 1

        while (start_token < len(vector) and vector[start_token] != 1)  :
            start_token += 1

        while (end_token > 0 and vector[end_token] != 1) :
            end_token -= 1

        return start_token, end_token+1