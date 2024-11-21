import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import create_sequences_fun, metrics_fun
from models import SimpleGRU, SimpleLSTM, SimpleMLP, SimpleTCN


class FeatureClassifier:
    def __init__(self, X_train, X_test, Y_train, Y_test, feature_set, model_type='MLP', epochs=10, 
                 sequence_length=10, batch_size=50, learning_rate=0.1, expts=3):
        """
        Initializes the FeatureClassifier class.

        Parameters:
        - X_train (np.ndarray or torch.Tensor): Training feature matrix of shape (num_train_samples, num_features).
        - X_test (np.ndarray or torch.Tensor): Testing feature matrix of shape (num_test_samples, num_features).
        - Y_train (np.ndarray or torch.Tensor): Training target vector of shape (num_train_samples,).
        - Y_test (np.ndarray or torch.Tensor): Testing target vector of shape (num_test_samples,).
        - feature_set (list or array-like): List of feature indices to use for classification.
        - model_type (str): Type of model to use ('LSTM', 'GRU', 'TCN', 'MLP').
        - epochs (int): Number of training epochs.
        - sequence_length (int): Length of input sequences for sequence models.
        - batch_size (int): Batch size for training.
        - learning_rate (float): Learning rate for the optimizer.
        - expts (int): Number of experiments to run.
        """
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('initializing')
        # Select feature_set from train and test data
        if isinstance(X_train, np.ndarray):
            self.X_train = torch.tensor(X_train[:, feature_set], dtype=torch.float32).to(self.device)
        elif isinstance(X_train, torch.Tensor):
            self.X_train = X_train[:, feature_set].clone().float().to(self.device)
        else:
            raise TypeError("X_train must be a numpy.ndarray or torch.Tensor")
        
        if isinstance(X_test, np.ndarray):
            self.X_test = torch.tensor(X_test[:, feature_set], dtype=torch.float32).to(self.device)
        elif isinstance(X_test, torch.Tensor):
            self.X_test = X_test[:, feature_set].clone().float().to(self.device)
        else:
            raise TypeError("X_test must be a numpy.ndarray or torch.Tensor")
        
        # Targets
        if isinstance(Y_train, np.ndarray):
            self.Y_train = torch.tensor(Y_train, dtype=torch.long).to(self.device)
        elif isinstance(Y_train, torch.Tensor):
            self.Y_train = Y_train.clone().long().to(self.device)
        else:
            raise TypeError("Y_train must be a numpy.ndarray or torch.Tensor")
        
        if isinstance(Y_test, np.ndarray):
            self.Y_test = torch.tensor(Y_test, dtype=torch.long).to(self.device)
        elif isinstance(Y_test, torch.Tensor):
            self.Y_test = Y_test.clone().long().to(self.device)
        else:
            raise TypeError("Y_test must be a numpy.ndarray or torch.Tensor")

        self.feature_set = feature_set
        self.model_type = model_type
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.expts = expts

        # Determine output dimension
        self.output_dim = len(torch.unique(self.Y_train))

        # Initialize metrics lists
        self.accuracy_list1, self.fp_list1, self.f1_score1, self.precision_list1, self.recall_list1 = [], [], [], [], []

    def initialize_model(self):
        """
        Initializes the classification model based on the specified model type.

        Returns:
        - nn.Module: The initialized model.
        """
        input_size = self.X_train.shape[1]
        if self.model_type == 'LSTM':
            model = SimpleLSTM(input_size, 250, self.output_dim).to(self.device)
        elif self.model_type == 'TCN':
            model = SimpleTCN(input_size, 250, self.output_dim).to(self.device)
        elif self.model_type == 'GRU':
            model = SimpleGRU(input_size, 250, self.output_dim).to(self.device)
        elif self.model_type == 'MLP':
            hidden_sizes = [250, 250]
            model = SimpleMLP(input_size, hidden_sizes, self.output_dim).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Conducts a single experiment: training and evaluating the model.

        Appends the computed metrics to the respective lists.

        Parameters:
        - X_train (torch.Tensor): Training feature tensor.
        - y_train (torch.Tensor): Training target tensor.
        - X_test (torch.Tensor): Testing feature tensor.
        - y_test (torch.Tensor): Testing target tensor.
        """
        print('train and eval')
        # Initialize metrics for this experiment
        accuracy_list, fp_list, f1_score_list, precision_list, recall_list = [], [], [], [], []

        # Initialize model
        model = self.initialize_model()

        # Initialize optimizer and scheduler
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 500, 750], gamma=0.1)

        # Define loss function
        criterion = nn.NLLLoss()

        # Training loop
        for epoch in range(self.epochs):
            model.train()
            loss_list = []

            num_train_samples = X_train.size(0)
            for i in range(0, num_train_samples, self.batch_size):
                end_idx = min(i + self.batch_size, num_train_samples)
                indices = list(range(i, end_idx))
                current_batch_size = end_idx - i
                if current_batch_size < self.batch_size:
                    continue  # Skip batch if it's not full size (optional based on model requirements)
                X_batch = X_train[indices]
                y_batch = y_train[indices].squeeze()

                if self.model_type in ['LSTM', 'GRU', 'TCN']:
                    try:
                        X_batch_seq, y_batch_seq = create_sequences_fun(X_batch, y_batch, self.sequence_length)
                        # Ensure batch size remains the same after sequence creation
                        # If not, skip or handle appropriately
                        if X_batch_seq.size(0) < 1:
                            continue
                        X_batch = X_batch_seq
                        y_batch = y_batch_seq
                    except ValueError:
                        # Sequence length longer than dataset, skip this batch
                        continue

                # Forward pass
                y_pred = model(X_batch)

                # Compute loss
                loss = criterion(y_pred, y_batch)
                loss_list.append(loss.item())

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Step the scheduler
            scheduler.step()

            # Compute average loss
            avg_loss = np.array(loss_list).mean() if loss_list else 0
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss}")

            # Evaluation
            model.eval()
            with torch.no_grad():
                if self.model_type in ['LSTM', 'GRU', 'TCN']:
                    try:
                        X_test_seq, y_test_seq = create_sequences_fun(X_test, y_test, self.sequence_length)
                        y_pred_test = model(X_test_seq)
                        y_test_trimmed = y_test[self.sequence_length-1:].to(self.device)
                        # Align the lengths
                        y_test_trimmed = y_test_trimmed[:len(y_pred_test)]
                    except ValueError:
                        # Sequence length longer than test set, skip evaluation
                        y_pred_test = torch.tensor([]).to(self.device)
                        y_test_trimmed = torch.tensor([]).to(self.device)
                else:
                    y_pred_test = model(X_test)
                    y_test_trimmed = y_test
                
                if y_pred_test.size(0) > 0:
                    y_pred_classes = torch.argmax(y_pred_test, dim=1)

                    # Ensure both tensors are on CPU and have matching dimensions
                    y_test_trimmed = y_test_trimmed.cpu()[:len(y_pred_classes)]
                    y_pred_classes = y_pred_classes.cpu()

                    # Calculate metrics
                    accuracy, f1, precision, recall, false_positive_rate = metrics_fun(y_test_trimmed, y_pred_classes)

                    # Append metrics
                    accuracy_list.append(accuracy)
                    f1_score_list.append(f1)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    fp_list.append(false_positive_rate)

                    # Print metrics
                    print(f"Training Evaluation Report:\n"
                          f" - Accuracy: {accuracy*100:.2f}%\n"
                          f" - F1 Score (Weighted): {f1:.2f}\n"
                          f" - Precision (Weighted): {precision:.2f}\n"
                          f" - Recall (Weighted): {recall:.2f}\n"
                          f" - Average False Positive Rate: {false_positive_rate:.2f}\n")
                else:
                    # No evaluation performed
                    print("No evaluation performed for this epoch due to sequence length constraints.")
                    continue

        # After training, collect metrics
        self.accuracy_list1.append(accuracy_list)
        self.f1_score1.append(f1_score_list)
        self.precision_list1.append(precision_list)
        self.recall_list1.append(recall_list)
        self.fp_list1.append(fp_list)

    def run_main(self):
        """
        Runs the classification process across multiple experiments and returns the metrics.

        Returns:
        - dict: A dictionary containing lists of metrics across experiments.
        """
        print('running main')
        for expt in range(self.expts):
            print(f"Experiment {expt+1}/{self.expts}")
            self.train_and_evaluate(self.X_train, self.Y_train, self.X_test, self.Y_test)
        
        # Compile metrics into a dictionary
        metrics = {
            'accuracy': self.accuracy_list1,
            'f1_score': self.f1_score1,
            'precision': self.precision_list1,
            'recall': self.recall_list1,
            'false_positive_rate': self.fp_list1
        }
        return metrics


