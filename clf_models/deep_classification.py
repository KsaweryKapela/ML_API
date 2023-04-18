from helpers import open_covtype_sample, X_y_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = open_covtype_sample()
X, y = X_y_split(df)