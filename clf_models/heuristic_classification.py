import pandas as pd
import numpy as np
from helpers import open_covtype_sample
from random import randint


class SimpleHeuristicModel():
    def __init__(self):
        self.means = self.open_means()
        self.input = None

        
    def open_means(self):
        for loop in range(2):
            try:
                means = pd.read_csv('means_heuristic_model.csv').to_numpy()
                return means
            
            except FileNotFoundError:
                print('File with means not found, generating means based on sample.')
                self.generate_means()
                print('Means generated')


    def generate_means(self):

        df = open_covtype_sample()
        means_tuples = []

        for label in range(1, 8):
            working_df = df[df[54] == label]
            
            means_tuples.append((working_df[0].mean(),
                                working_df[2].mean(),
                                working_df[5].mean()))
        df_means = pd.DataFrame(means_tuples)
        df_means.to_csv('means_heuristic_model.csv')


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
            if len(self.input) == 54:
                return True
        
        elif len(self.input.shape) == 2:
            if self.input.shape[1] == 54:
                return True

        raise Exception('Provide (x, 54) dim array.')
    

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
                
                # Adding some variance to results
                abs_diff += randint(-1500, 1500)
                
            if abs_diff < lowest_mean_diff:
                lowest_mean_diff = abs_diff
                category = i + 1
                
        return category