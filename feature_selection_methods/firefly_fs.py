import numpy as np
import torch
from mine import mine_fa, mine  # Ensure that the 'mine' module is available

class firefly_selection:
    """
    Firefly Algorithm for Feature Selection.

    This class implements the Firefly Algorithm to select an optimal subset of features
    based on mutual information and classifier performance.
    """
    
    def __init__(self, L, C, n_fireflies=25, k=15, Tmax=500, gamma=1.0, alpha=0.05, 
                  device=None):
        """
        Initialize the FireflyFeatureSelector.

        Parameters:
        - n_fireflies (int): Number of fireflies in the population.
        - n_features (int): Total number of features in the dataset.
        - k (int): Number of features to select.
        - Tmax (int): Maximum number of iterations.
        - gamma (float): Attractiveness coefficient.
        - alpha (float): Randomness coefficient.
        - L (np.ndarray): Feature dataset (samples x features).
        - C (np.ndarray): Class labels corresponding to the dataset.
        - device (str, optional): Computation device ('cuda' or 'cpu'). Auto-detects if not provided.
        """
        self.n_fireflies = n_fireflies
        self.n_features = L.shape[1]
        self.k = k
        self.Tmax = Tmax
        self.gamma = gamma
        self.alpha = alpha
        self.L = L
        self.C = C
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize firefly population with unique binary arrays
        self.fireflies = self.generate_unique_binary_arrays(self.n_fireflies, 
                                                             self.n_features, 
                                                             self.k)
        # Evaluate initial fitness of all fireflies
        self.fitness = np.array([self.evaluate_classifier(firefly) for firefly in self.fireflies])
        
        # Sort fireflies based on fitness (ascending order)
        sorted_indices = np.argsort(self.fitness)
        self.fireflies = np.array(self.fireflies)[sorted_indices]
        self.fitness = self.fitness[sorted_indices]
    
    def generate_unique_binary_arrays(self, x, max_length, target_sum):
        """
        Generate a list of unique binary arrays with a fixed number of ones.

        Parameters:
        - x (int): Number of unique arrays to generate.
        - max_length (int): Length of each binary array.
        - target_sum (int): Number of ones in each binary array.

        Returns:
        - List[np.ndarray]: List of unique binary arrays.
        """
        arrays = set()
        attempts = 0
        max_attempts = x * 10  # Prevent infinite loop
        
        while len(arrays) < x and attempts < max_attempts:
            arr = np.zeros(max_length, dtype=int)
            indices = np.random.choice(max_length, target_sum, replace=False)
            arr[indices] = 1
            arrays.add(tuple(arr))
            attempts += 1
        
        if len(arrays) < x:
            raise ValueError("Unable to generate the required number of unique binary arrays.")
        
        return [np.array(arr) for arr in arrays]
    
    def evaluate_classifier(self, feature_subset):
        """
        Evaluate the classifier performance based on the selected features.

        Parameters:
        - feature_subset (np.ndarray): Binary array indicating selected features.

        Returns:
        - float: Mean classifier performance metric.
        """
        selected_features = self.L[:, feature_subset.astype(bool)]
        # Compute mutual information-based feature assessment
        mi_scores = mine(selected_features,self.C, 5).run()
        return np.mean(mi_scores, axis=1)[-1]
    
    def move_firefly(self, i, j):
        """
        Move firefly i towards firefly j based on attractiveness and randomness.

        Parameters:
        - i (int): Index of the current firefly.
        - j (int): Index of the attracting firefly.
        """
        # Calculate Euclidean distance between firefly i and j
        distance = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
        beta = np.exp(-self.gamma * distance ** 2)#+0.3  # Attractiveness
        
        for idx in range(self.n_features):
            # Calculate movement probability for each feature
            probability = beta * (self.fireflies[j][idx] - self.fireflies[i][idx]) + \
                          self.alpha * (np.random.rand() - 0.5)
            if np.random.rand() < probability:
                if self.fireflies[i][idx] == 0 and np.sum(self.fireflies[i]) < self.k:
                    self.fireflies[i][idx] = 1  # Add feature
                elif self.fireflies[i][idx] == 1:
                    self.fireflies[i][idx] = 0  # Remove feature
    
    def run_main(self):
        """
        Execute the Firefly Algorithm to perform feature selection.

        Returns:
        - Tuple[np.ndarray, float]: The best feature subset and its fitness score.
        """
        for t in range(1, self.Tmax + 1):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if i != j and self.fitness[i] < self.fitness[j]:
                        # Move firefly i towards firefly j
                        self.move_firefly(i, j)
                        # Re-evaluate the moved firefly
                        self.fitness[i] = self.evaluate_classifier(self.fireflies[i])
            
            # Sort fireflies based on updated fitness
            sorted_indices = np.argsort(self.fitness)
            self.fireflies = self.fireflies[sorted_indices]
            self.fitness = self.fitness[sorted_indices]
            
            print(f"Iteration {t} completed. Best features: {self.fireflies[-1]}")
        
        # Return the best feature subset and its fitness
        return self.fireflies[-1]




