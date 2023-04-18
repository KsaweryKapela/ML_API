import numpy as np
import matplotlib.pyplot as plt
from helpers import open_covtype_sample, X_y_split, one_hot_encode
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle


def create_model(neurons=64, activation_1='linear', activation_2='linear', optimizer='Adam'):
    model = Sequential()
    model.add(Dense(neurons, input_dim=54, activation=activation_1))
    model.add(Dense(neurons/2, activation=activation_2))
    model.add(Dense(round(neurons/3), activation=activation_2))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def grid_1(X, y):
    # Reduce grid or epochs to save computing power
    neurons = [100, 64, 32]
    activation_1 = ['relu', 'tanh', 'sigmoid','linear']
    activation_2 = ['relu', 'tanh', 'sigmoid','linear']

    param_grid = dict(neurons=neurons,
                      activation_1=activation_1,
                      activation_2=activation_2)


    model = KerasClassifier(model=create_model, batch_size=20, epochs=100, verbose=0,
                            neurons=neurons, activation_1=activation_1, activation_2=activation_2)
                            
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

    print('Fitting grid_1, can take some time')
    grid_result = grid.fit(X, y)

    print(f'Grid_1 best acc: {grid_result.best_score_}, best params: {grid_result.best_params_}')

    return grid_result.best_params_


def grid_2(X, y, params):

    # Splitting grid into two functions, so it saves computing power
    batch_size = [10, 25, 50]
    epochs = [50, 100, 200]
    param_grid = dict(batch_size=batch_size, epochs=epochs)

    model = KerasClassifier(model=create_model, activation_1=params['activation_1'],
                                                  activation_2=params['activation_2'],
                                                  neurons=params['neurons'],
                                                  verbose=0)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

    print('Fitting grid_2')
    grid_result = grid.fit(X, y)

    print(f'Grid 2 best acc: {grid_result.best_score_}, best params: {grid_result.best_params_}')

    return grid_result.best_params_


def visualize_training(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def evaluate_best_model(X, y, X_eval, y_eval, params_1, params_2):

    print('Evaluating best model')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = create_model(activation_1=params_1['activation_1'],
                              activation_2=params_1['activation_2'],
                              neurons=params_1['neurons'])
                            # Deleted optimizer, can make accuraccy static
                             

    history = best_model.fit(X_train, y_train, validation_data=(X_eval, y_eval),
                             verbose=2, epochs=params_2['epochs'], batch_size=params_2['batch_size'])

    pred = best_model.predict(X_test)
    print(f'Accuraccy score of best model: {accuracy_score(np.round(pred), y_test)}')
    visualize_training(history)


def save_model(name, params_1, params_2):
    # Creating model again so it's trained on a whole 10 000 sample as other models.
    model = create_model(activation_1=params_1['activation_1'],
                         activation_2=params_1['activation_2'],
                         neurons=params_1['neurons'])
    print('Fitting model to save')
    model.fit(X, y, verbose=0, epochs=params_2['epochs'], batch_size=params_2['batch_size'])
    path = 'clf_models/models/'
    model.save(f'{path}{name}')
    print(f'Model saved in {path}{name}')


if __name__ == "__main__":

    X, y = X_y_split(open_covtype_sample())
    y = one_hot_encode(y)

    X_eval, y_eval = X_y_split(open_covtype_sample('Eval'))
    y_eval = one_hot_encode(y_eval)

    best_params_1 = grid_1(X, y)
    best_params_2 = grid_2(X, y, best_params_1)
    evaluate_best_model(X, y, X_eval, y_eval, best_params_1, best_params_2)
    save_model('NN_clf', best_params_1, best_params_2)
