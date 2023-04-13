import pandas as pd
import numpy as np

class SimpleHeuristicModel():
    def __init__(self):
        self.means = pd.read_csv('means_heuristic_model.csv').to_numpy()
        self.input = None


    def predict(self, input):

        predictions = []

        self.input = np.asarray(input)
        if self.check_input():
            if len(self.input.shape) == 2:
                for arr in self.input:
                    predictions.append(self.categorize(arr))
            else:
                predictions.append(self.categorize(input))

        return predictions

    def check_input(self):
        
        if len(self.input.shape) == 1:
            if len(self.input) == 3:
                return True
        
        elif len(self.input.shape) == 2:
            if self.input.shape[1] == 3:
                return True

        raise Exception('Provide (x, 3) dim array. Use elevation, slope and road distance variables')
    
    def categorize(self, features):

        lowest_mean_diff = float('inf')
        category = None

        for i in range(0, 7):
            abs_diff = 0
            working_means = self.means[i]

            for i2 in range(0, 3):
                if i2 == 1:
                    abs_diff += 10 * abs(working_means[i2] - features[i2])
                else:
                    abs_diff += abs(working_means[i2] - features[i2])

            if abs_diff < lowest_mean_diff:
                lowest_mean_diff = abs_diff
                category = i + 1
                
        return category