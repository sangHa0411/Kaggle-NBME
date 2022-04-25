import numpy as np

class Encoder :

    def __init__(self, tokenizer, max_length=512, label_pad_token_id=-100) :
        self.tokenizer = tokenizer 
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, dataset) :
        model_inputs = self.tokenizer(dataset['pn_history'],
            dataset['feature_text'],
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            max_length=self.max_length,
            truncation='only_second',
        )
       
        model_labels = []
        mappings = model_inputs.pop("offset_mapping")

        for i in range(len(dataset['case_num'])) :
            mapping = mappings[i]
            locations = dataset['location'][i]
            annotation_length = dataset['annotation_length'][i]
            sequence_ids = model_inputs.sequence_ids(i)

            start_token, end_token = self.get_positions(sequence_ids)

            labels = np.zeros(len(sequence_ids)).astype('float')
            labels[end_token+1:] = self.label_pad_token_id
            labels[:start_token] = self.label_pad_token_id
            
            if annotation_length > 0 :
                for loc in locations :
                    for l in [s.split() for s in loc.split(';')] :
                        org_start, org_end = int(l[0]), int(l[1])

                        token_start_index = start_token
                        token_end_index = end_token

                        if mapping[token_start_index][0] <= org_start and org_end <= mapping[token_end_index][1] :
                            while(token_start_index < len(sequence_ids) and mapping[token_start_index][0] + 1 < org_start) :
                                token_start_index += 1

                            while(token_end_index >= token_start_index and mapping[token_end_index][1] > org_end) :
                                token_end_index -= 1

                            labels[token_start_index:token_end_index+1] = 1.0
            
            model_labels.append(labels)
            
        model_inputs['labels'] = model_labels
        return model_inputs

    def get_positions(self, vector, value=0) :
        start_token = 1
        end_token = len(vector) - 1

        while (start_token < len(vector) and vector[start_token] != value)  :
            start_token += 1

        while (end_token > 0 and vector[end_token] != value) :
            end_token -= 1

        return start_token, end_token