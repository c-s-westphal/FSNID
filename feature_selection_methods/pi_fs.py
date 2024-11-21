import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

class pi_selection:
    def __init__(self, X, y, n_estimators=100, random_state=42, n_repeats=20, n_trials=3):
        """
        Initializes the RandomForestFeatureSelector.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target vector.
        - n_estimators (int): Number of trees in the forest. Default is 100.
        - random_state (int): Seed for random number generator. Default is 42.
        - n_repeats (int): Number of times to permute a feature. Default is 20.
        - n_trials (int): Number of trials to perform feature importance calculation. Default is 3.
        """
        self.X = X
        self.y = y.ravel()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_repeats = n_repeats
        self.n_trials = n_trials

    def run_main(self):
        """
        Executes the feature selection process using Random Forest and permutation importance.

        Returns:
        - important_features (list): List of indices of selected important features.
        """
        # Initialize arrays to store importances
        feature_importances = np.zeros((self.X.shape[1], self.n_trials))
        random_feature_importances = np.zeros(self.n_trials)

        # Perform feature importance calculation over multiple trials
        for trial in range(self.n_trials):
            # Randomly sample 30% of the data without replacement
            indices = np.random.choice(self.X.shape[0], int(self.X.shape[0] * 0.3), replace=False)
            X_new = self.X[indices, :]
            y_new = self.y[indices]

            # Add a random feature to the dataset
            random_feature = np.random.rand(X_new.shape[0], 1)
            X_with_random = np.hstack((X_new, random_feature))

            # Initialize and train the Random Forest model
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
            model.fit(X_with_random, y_new)

            # Index of the random feature
            random_feature_index = X_with_random.shape[1] - 1

            # Compute permutation importance
            perm_importance = permutation_importance(
                model,
                X_with_random,
                y_new,
                n_repeats=self.n_repeats,
                random_state=self.random_state + trial,
                n_jobs=-1
            )

            # Store the mean importance of the random feature
            random_feature_importances[trial] = perm_importance.importances_mean[random_feature_index]

            # Store the mean importances of the actual features
            feature_importances[:, trial] = perm_importance.importances_mean[:random_feature_index]

        # Determine the 95th percentile threshold from random feature importances
        threshold = np.percentile(random_feature_importances, 95)

        # Select features that are consistently more important than the threshold across all trials
        important_features = [
            feature_idx
            for feature_idx in range(self.X.shape[1])
            if np.all(feature_importances[feature_idx, :] > threshold)
        ]

        return important_features
