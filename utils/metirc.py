import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids

    predictions = np.where(pred.predictions > 0.5, 1.0, 0.0)
    predictions = predictions.squeeze(-1)

    label_list = []
    pred_list = []

    batch_size, seq_size = labels.shape
    
    for i in range(batch_size) :
        label_ids = labels[i]
        prediction_ids = predictions[i]

        for j in range(seq_size) :
            label = label_ids[j]
            pred = prediction_ids[j]

            if label == -100 :
                continue

            if label == 1 or pred == 1:
                label_list.append(label)
                pred_list.append(pred)

    label_indices = list(range(2))
    acc = accuracy_score(label_list, pred_list)
    f1 = f1_score(label_list, pred_list, average="micro", labels=label_indices) * 100.0
    return {'F1' : f1, 'Accuracy' : acc}