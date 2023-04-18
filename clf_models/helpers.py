import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# Working on sample to save computing power
def open_covtype_sample(purpose='Train'):

    if purpose not in ['Train', 'Eval', 'Eval_2', 'Test']:
        return 'Choose correct purpouse keyword'

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
    df = pd.read_csv(url, names=range(55))
    df = df.sample(frac = 1) # Shuffling dataframe

    # Limiting test set due to computing power limitations
    if purpose == 'Train':
        df = df[:10000]

    elif purpose == 'Eval':
        df = df[10000:12000]
    
    elif purpose == 'Eval_2':
        df = df[12000:14000]
        
    elif purpose == 'Test':
        df = df[12000:20000]

    print(f'{purpose} Sample opened')
    return df

def X_y_split(df):
    X = df[range(54)].to_numpy()
    y = df[54].to_numpy()
    return X, y

def one_hot_encode(arr):
    encoder = OneHotEncoder()
    encoded_labels = encoder.fit_transform(arr.reshape(-1, 1))
    arr = encoded_labels.toarray()
    return arr
