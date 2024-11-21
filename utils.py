import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def create_sequences_fun( X, y, sequence_length):
        num_sequences = X.shape[0] - sequence_length + 1
        X_sequences = []
        y_sequences = []
        for i in range(num_sequences):
            X_sequences.append(X[i:i+sequence_length])
            y_sequences.append(y[i+sequence_length-1])
        X_sequences = torch.stack(X_sequences)
        y_sequences = torch.stack(y_sequences).squeeze(-1)
        return X_sequences, y_sequences

def metrics_fun(y_test_trimmed, y_pred_classes):
    # Calculate metrics
    accuracy = accuracy_score(y_test_trimmed, y_pred_classes)
    f1 = f1_score(y_test_trimmed, y_pred_classes, average='weighted')
    precision = precision_score(y_test_trimmed, y_pred_classes, average='weighted')
    recall = recall_score(y_test_trimmed, y_pred_classes, average='weighted')
    conf_mat = confusion_matrix(y_test_trimmed, y_pred_classes)
                    
    # Calculate false positive rate for each class and take the average
    fp_rate_per_class = []
    for i in range(conf_mat.shape[0]):
        fp = conf_mat[:, i].sum() - conf_mat[i, i]
        tn = conf_mat.sum() - (conf_mat[i, :].sum() + conf_mat[:, i].sum() - conf_mat[i, i])
        if fp + tn > 0:
            fp_rate = fp / (fp + tn)
            fp_rate_per_class.append(fp_rate)
    false_positive_rate = np.mean(fp_rate_per_class) if fp_rate_per_class else 0.0
    return accuracy, f1, precision, recall, false_positive_rate