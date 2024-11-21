import numpy as np
import math
from mine import mine_fa, mine_fa_td, mine, mine_td



    
class fsnid_selection:
    def __init__(self, features, targets, num_iterations=10000, model_type=None, mi_ordering_bool = True):
        """
        Initializes the FsnidSelection class.

        Parameters:
        - features (np.ndarray or torch.Tensor): Feature matrix.
        - targets (np.ndarray or torch.Tensor): Target vector.
        - num_iterations (int): Number of iterations for mining.
        - model_type (str, optional): Type of model to use ('LSTM', 'GRU', 'TCN'). 
                                      If None or unsupported, defaults to mine_fa.
        """
        self.features = features
        self.targets = targets
        self.num_iterations = num_iterations
        self.model_type = model_type
        self.mi_ordering_bool = mi_ordering_bool
        self.nm_upper_bound = self.null_model()

    def run_main(self):
        """
        Runs the main feature selection process.

        Returns:
        - feats (list): Selected feature indices.
        """
        feats = list(range(self.features.shape[1]))
        if self.mi_ordering_bool:
            iterable_feats = self.mi_ordering()
        else:
            iterable_feats = range(self.features.shape[1])
        for feat in iterable_feats:  
            temp_indexes = feats.copy()
            temp_indexes = [item for item in temp_indexes if item != feat]
            if len(feats) > 1:
                arr = self.mine(self.features[:, feats], self.targets) - self.mine(self.features[:, temp_indexes], self.targets)
            else:
                arr = self.mine(self.features[:, feats], self.targets)

            if arr.mean(axis=0)[-1] - 2 * (arr.std(axis=0)[-1] / math.sqrt(3)) < self.nm_upper_bound:
                feats = temp_indexes
                print(f'Feature {feat} excluded: lower bound {arr.mean(axis=0)[-1] - 2*(arr.std(axis=0)[-1]/math.sqrt(3))} < nm {self.nm_upper_bound}')
            else:
                print(f'Feature {feat} included: lower bound {arr.mean(axis=0)[-1] - 2*(arr.std(axis=0)[-1]/math.sqrt(3))} >= nm {self.nm_upper_bound}')
        return feats

    def null_model(self):
        """
        Computes the null model upper bound.

        Returns:
        - float: Null model upper bound value.
        """
        nm_arr = self.mine(np.random.random(self.targets.shape), self.targets)
        return nm_arr.mean(axis=0)[-1] + 2 * (nm_arr.std(axis=0)[-1] / math.sqrt(3))

    def mine(self, features, targets):
        """
        Internal method to instantiate and run the appropriate mining class.

        Parameters:
        - features (np.ndarray or torch.Tensor): Feature matrix.
        - targets (np.ndarray or torch.Tensor): Target vector.

        Returns:
        - np.ndarray: Results from the mining process.
        """
        if self.model_type in ['LSTM', 'GRU', 'TCN']:
            # Use mine_fa_td with the specified model_type
            miner = mine_td(
                p_dis=features,
                q_dis=targets,
                num_iterations=self.num_iterations,
                model_type=self.model_type
            )
        else:
            # Use mine_fa as the default
            miner = mine(
                p_dis=features,
                q_dis=targets,
                num_iterations=self.num_iterations
            )
        
        return miner.run()

    def mi_ordering(self):
        mis = [self.mine(self.features[:, feat], self.targets).mean(axis=0)[-1] for feat in range(self.features.shape[1])]
        sorted_indices = sorted(range(len(mis)), key=lambda i: mis[i])
        return sorted_indices
    


   
    


