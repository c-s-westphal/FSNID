import numpy as np
import os
import torch
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from evaluate_features import FeatureClassifier
from feature_selection_methods.fsnid_fs import fsnid_selection
from feature_selection_methods.brown_fs import brown_selection
from feature_selection_methods.firefly_fs import firefly_selection
from feature_selection_methods.lasso_fs import lasso_selection
from feature_selection_methods.pi_fs import pi_selection
import time
import json


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        args: Parsed arguments containing 'nme' and 'selection_method'.
    """
    parser = argparse.ArgumentParser(description='Feature Selection and Classification Script')
    parser.add_argument('--nme', type=str, default='BOTIOT',
                        help='Name of the dataset (default: BOTIOT)')
    parser.add_argument('--selection_method', type=str, default='fsnid',
                        choices=['fsnid', 'brown', 'firefly', 'lasso', 'pi'],
                        help='Feature selection method to use (default: fsnid)')
    parser.add_argument('--model_type', type=str, default='MLP',
                        choices=['MLP', 'LSTM', 'TCN', 'GRU'],
                        help='Type of model to evaluate the features with (default: MLP)')
    args = parser.parse_args()
    return args

def get_selection_class(method):
    """
    Returns the corresponding feature selection class based on the method name.

    Args:
        method (str): The name of the selection method.

    Returns:
        selection_class: The corresponding feature selection class.

    Raises:
        ValueError: If the selection method is not recognized.
    """
    selection_mapping = {
        'fsnid': fsnid_selection,
        'brown': brown_selection,
        'firefly': firefly_selection,
        'lasso': lasso_selection,
        'pi': pi_selection
    }

    if method not in selection_mapping:
        raise ValueError(f"Invalid selection method '{method}'. Choose from {list(selection_mapping.keys())}.")

    return selection_mapping[method]

def save_metrics(metrics, data_dir, nme, selection_method):
    """
    Saves the metrics dictionary to a JSON file.

    Args:
        metrics (dict): The metrics to save.
        data_dir (str): The directory where the metrics file will be saved.
        nme (str): The dataset name.
        selection_method (str): The feature selection method used.
    """
    metrics_filename = f'{selection_method}_{nme}_metrics.json'
    metrics_path = os.path.join(data_dir, metrics_filename)

    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
    except TypeError as te:
        print(f"Failed to serialize metrics to JSON: {te}")
        # Optionally, save as a pickle file
        import pickle
        pickle_filename = f'{selection_method}_{nme}_metrics.pkl'
        pickle_path = os.path.join(data_dir, pickle_filename)
        with open(pickle_path, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"Metrics saved as pickle to {pickle_path}")
    except Exception as e:
        print(f"An unexpected error occurred while saving metrics: {e}")

def main():
    # Parse command-line arguments
    args = parse_arguments()
    nme = args.nme
    selection_method = args.selection_method
    model_type = args.model_type

    print(f"Dataset Name (nme): {nme}")
    print(f"Feature Selection Method: {selection_method}")
    print(f"Model Type: {model_type}")

    # Define filenames
    X_filename = f'{nme}_X.npy'
    Y_filename = f'{nme}_Y.npy'

    # Get current working directory
    current_dir = os.getcwd()

    # Define the data directory path
    data_dir = os.path.join(current_dir, 'data')
    X_path = os.path.join(data_dir, X_filename)
    Y_path = os.path.join(data_dir, Y_filename)

    # Check if files exist
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"File not found: {X_path}")
    if not os.path.exists(Y_path):
        raise FileNotFoundError(f"File not found: {Y_path}")

    print("Loading data...")
    # Load the data
    X = np.load(X_path)
    y = np.load(Y_path)

    print(f"Original Shapes -> X: {X.shape}, y: {y.shape}")

    # Convert y to a PyTorch tensor
    y_tensor = torch.tensor(y).long()

    # Convert X to a DataFrame for preprocessing
    df = pd.DataFrame(X)

    print("Applying preprocessing...")
    # Apply normalization
    df = (df - df.min()) / (df.max() - df.min())
    # Fill NaNs with 0
    df = df.fillna(0)
    # Convert back to NumPy array
    X_preprocessed = df.to_numpy()

    print("Preprocessing completed.")
    print(f"Preprocessed X shape: {X_preprocessed.shape}")

    # Convert y_tensor back to NumPy array for consistency
    y_preprocessed = y_tensor.numpy()

    # Initialize the feature selection class based on the selection method
    selection_class = get_selection_class(selection_method)
    print(f"Initializing feature selection using '{selection_method}' method...")
    if selection_method == 'fsnid':
        feature_selector = selection_class(X_preprocessed, y_preprocessed, model_type=model_type)
    else:
        feature_selector = selection_class(X_preprocessed, y_preprocessed)
    
    # Perform feature selection
    start_time = time.time()
    feats = feature_selector.run_main()
    end_time = time.time()
    print(f"Selected Features: {feats}")

    metrics = FeatureClassifier(X_preprocessed, X_preprocessed, y_preprocessed, y_preprocessed, feats, model_type=model_type).run_main()

    # Optional: Save selected features and processing time
    print("Saving classification metrics...")
    save_metrics(metrics, data_dir, nme, selection_method)
    
    print(f'Finished, and the following features were selected: {feats}')
    np.save(os.path.join(data_dir, f'fsnid_fs_{nme}.npy'), np.array(feats))
    print(f"Duration of the process: {end_time - start_time} seconds")
    np.save(os.path.join(data_dir, f'fsnid_{nme}_time.npy'), np.array([end_time - start_time]))
    

if __name__ == "__main__":
    main()
