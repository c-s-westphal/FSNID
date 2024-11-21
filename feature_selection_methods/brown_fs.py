import numpy as np
import math
from mine import mine_fa



    

class brown_selection:
    def __init__(self, features, targets, num_iterations=10000):
        '''
        Here we introduce our conditional likelihood maximization code
        '''
        self.features = features
        self.targets = targets
        self.num_iterations = num_iterations
        print('b4 null')
        
        self.chosen_indexes = []
        print('initialized')

    def get_updated_mis(self):
        '''
        Here, we complete an individual round of selection.
        This involves claculatinf the conditional MI of the 
        non selected features and adding that which scored the max.
        '''
        mis = []
        feats = []
        stds = []
        for feat in range(self.features.shape[1]):
            if feat not in self.chosen_indexes:
                feats.append(feat)
                temp_indexes = self.chosen_indexes.copy()
                temp_indexes.append(feat)
                if len(self.chosen_indexes) > 0:
                    arr = (mine_fa(self.features[:, temp_indexes], self.targets, self.num_iterations).run()-mine_fa(self.features[:, self.chosen_indexes], self.targets, self.num_iterations).run())
                    mis.append(arr.mean(axis=0)[-1])
                    stds.append(arr.std(axis=0)[-1])                
                else:
                    arr =(mine_fa(self.features[:, temp_indexes], self.targets, self.num_iterations).run())
                    mis.append(arr.mean(axis=0)[-1])
                    stds.append(arr.std(axis=0)[-1])         
        max_val = max(mis)
        max_index = np.argmax(np.array(mis))
        print(f'here are the calculated mis {mis}')
        is_done = False if max_val - 1*(stds[max_index]/math.sqrt(3)) > self.nm_upper_bound else True
        return feats[mis.index(max_val)], is_done

    def null_model(self):
        '''
        In this function we introduce the null model used to calculate when 
        enough features have been selected.
        '''
        print('right before')
        #mine_fa(selected_features,self.C, 5).run()
        data = np.random.random(self.targets.shape)
        nm_arr = mine_fa(data, self.targets, self.num_iterations).run()
        return nm_arr.mean(axis=1)[-1] + 1*(nm_arr.std(axis=1)[-1]/math.sqrt(3))

    def run_main(self):
        '''
        Here we run the entire selection process.
        '''
        self.nm_upper_bound = self.null_model()
        selected_feat, is_done = self.get_updated_mis()
        self.chosen_indexes.append(selected_feat)
        print(f'feature selected at this stage is {selected_feat}, all chosen are {self.chosen_indexes}')
        
        while not is_done:
            selected_feat, is_done = self.get_updated_mis()
            self.chosen_indexes.append(selected_feat)
            print(f'feature selected at this stage is {selected_feat}, all chosen are {self.chosen_indexes}')
            if len(self.chosen_indexes) == self.features.shape[1]:
                is_done = True
        return self.chosen_indexes
        



