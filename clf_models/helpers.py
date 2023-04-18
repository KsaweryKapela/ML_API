import pandas as pd
import matplotlib.pyplot as plt

# Working on sample to save computing power
def open_covtype_sample(purpose='Train'):

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
    df = df.sample(frac = 1) # Shuffling dataframe
    df = pd.read_csv(url, names=range(55))

    if purpose == 'Train':
        df = df[:10000]
    elif purpose == 'Eval':
        df = df[10000:12000]
    print('Sample opened')
    return df

def X_y_split(df):
    X = df[range(54)].to_numpy()
    y = df[54].to_numpy()
    return X, y