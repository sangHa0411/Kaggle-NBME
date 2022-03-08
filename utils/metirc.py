import numpy as np

def true_positive(label_ids, prediction_ids) :
    ids_set = set([pred for pred in prediction_ids if pred in label_ids])
    return len(ids_set)

def false_negative(label_ids, prediction_ids) :
    ids_set = set([label for label in label_ids if label not in prediction_ids])
    return len(ids_set)

def false_positive(label_ids, prediction_ids) :
    ids_set = set([pred for pred in prediction_ids if pred not in label_ids])
    return len(ids_set)

def compute_metrics(pred):
    labels = pred.label_ids 
    predictions = np.argmax(pred.predictions, axis=-1)

    batch_size = len(labels)

    f1_scores = []
    for i in range(batch_size) :
        label_ids = (labels[i] == 1).astype('int')
        predictions_ids = (predictions[i] == 1).astype('int')

        label_ids = list(label_ids.nonzero()[0])
        prediction_ids = list(predictions_ids.nonzero()[0])

        if len(label_ids) == 0 and len(predictions_ids) == 0 :
            f1 = 1.0
        else :
            try :
                tp = true_positive(label_ids, prediction_ids)
                fp = false_positive(label_ids, prediction_ids)
                fn = false_negative(label_ids, prediction_ids)
                
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)

                if precision == 0 and recall == 0 :
                    breakpoint

                f1 = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError as e :
                f1 = 0.0
                
        f1_scores.append(f1)

    f1_score = np.mean(f1_scores)
    return {'F1 Score' : f1_score}