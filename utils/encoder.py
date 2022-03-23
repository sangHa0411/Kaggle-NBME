import copy
import numpy as np

class Encoder :
    def __init__(self, plm, tokenizer, max_length, label_pad_token_id=-100) :
        self.plm = plm
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, dataset) :
        model_inputs = self.tokenizer(dataset['feature'],
            dataset['history'],
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False if 'roberta' in self.plm else True,
            max_length=self.max_length,
            truncation='only_second',
        )

        model_labels = []

        batch_size = len(dataset['feature'])
        mappings = model_inputs.pop("offset_mapping")

        for i in range(batch_size) :
            mapping = copy.deepcopy(mappings[i])
            input_ids = model_inputs['input_ids'][i]
            locations = dataset['locations'][i]
            annotation_length = dataset['annotation_length'][i]
            sequence_ids = model_inputs.sequence_ids(i)

            start_token, end_token = self.get_positions(sequence_ids)

            if 'deberta' in self.plm :
                for i in range(start_token, end_token+1) : 
                    if self.tokenizer.convert_ids_to_tokens(input_ids[i])[0] == 'Ġ' :
                        mapping[i] = (mapping[i][0]+1, mapping[i][1])

            labels = np.zeros(len(sequence_ids)).astype('int')
            labels[end_token+1:] = self.label_pad_token_id
            labels[:start_token] = self.label_pad_token_id
            
            if annotation_length > 0 :
                for loc in locations :
                    for l in [s.split() for s in loc.split(';')] :
                        org_start, org_end = int(l[0]), int(l[1])

                        token_start_index = start_token
                        token_end_index = end_token

                        if mapping[token_start_index][0] <= org_start and org_end <= mapping[token_end_index][1] :
                            while(token_start_index < len(sequence_ids) and mapping[token_start_index][0] < org_start) :
                                token_start_index += 1

                            while(token_end_index >= token_start_index and mapping[token_end_index][1] > org_end) :
                                token_end_index -= 1

                            labels[token_start_index:token_end_index+1] = 1
            
            model_labels.append(labels)
            
        model_inputs['labels'] = model_labels
        return model_inputs

    def get_positions(self, vector) :
        start_token = 1
        end_token = len(vector) - 1

        while (start_token < len(vector) and vector[start_token] != 1)  :
            start_token += 1

        while (end_token > 0 and vector[end_token] != 1) :
            end_token -= 1

        return start_token, end_token