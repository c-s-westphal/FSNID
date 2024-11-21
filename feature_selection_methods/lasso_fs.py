import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class lasso_selection:
    def __init__(self, X, y, alpha=0.01, max_iterations=10000):
        """
        Initializes the LassoFeatureSelector.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target vector.
        - alpha (float): Regularization strength. Default is 0.01.
        - max_iterations (int): Maximum number of iterations for the solver. Default is 5000.
        """
        self.X = X
        self.y = y
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.selected_features = []

    def run_main(self):
        """
        Executes the feature selection process using Lasso regression.

        Returns:
        - selected_features (list): List of indices of selected important features.
        """
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # Initialize and fit the Lasso model
        lasso = Lasso(alpha=self.alpha, max_iter=self.max_iterations, random_state=42)
        lasso.fit(X_scaled, self.y)

        # Print the coefficients
        print("LASSO coefficients:", lasso.coef_)

        # Feature selection based on non-zero coefficients
        self.selected_features = list(np.where(lasso.coef_ != 0)[0])

        return self.selected_features


