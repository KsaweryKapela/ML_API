from clf_models.helpers import open_covtype_sample, X_y_split, one_hot_encode
from clf_models.heuristic_classification import SimpleHeuristicModel
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def eval_shallow_model(name, X, y, path):
    print(f'Evaluating {name}')
    model = pickle.load(open(f'{path}{name}', 'rb'))
    score = model.score(X, y)
    confusion = confusion_matrix(model.predict(X), y)
    print(f'{name} score: {score}')
    return score, confusion


def eval_deep_model(name, X, y, path):
    print(f'Evaluating {name}')
    y = one_hot_encode(y)
    model = keras.models.load_model(f'{path}{name}')
    pred = model.predict(X)
    score = accuracy_score(np.round(pred), y)
    print(f'{name} score: {score}')
    confusion = confusion_matrix(np.round(pred).argmax(axis=1), y.argmax(axis=1))
    return score, confusion


def eval_heuristic_model(X, y):
    print(f'Evaluating heuristic model')
    model = SimpleHeuristicModel()
    pred = model.predict(X)
    score = accuracy_score(pred, y)
    print(f'HEU score: {score}')
    confusion = confusion_matrix(pred, y)
    return score, confusion


def plot_scores(scores, labels):

    fig, ax = plt.subplots()
    ax.bar(labels, scores)

    ax.set_title('Accuraccy scores of models')

    plt.show()


def plot_confusion_matrices(matrices):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axs = axs.ravel()

    for i, (model_name, confusion_matrix) in enumerate(matrices.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=range(1, 8))
        disp.plot(ax=axs[i])
        axs[i].set_title(f"{model_name} confusion matrix")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    PATH = 'clf_models/models/'
    NN_MODEL = 'NN_clf'
    SVC_MODEL = 'RF_clf.sav'
    RF_MODEL = 'SVC_clf.sav'

    X, y = X_y_split(open_covtype_sample('Eval_2')) # Using eval_2 since eval was used for nn eval

    score_SVC, confusion_SVC = eval_shallow_model(SVC_MODEL, X, y, PATH)
    score_RF, confusion_RF = eval_shallow_model(RF_MODEL, X, y, PATH)
    score_NN, confusion_NN = eval_deep_model(NN_MODEL, X, y, PATH)
    score_HEU, confusion_HEU = eval_heuristic_model(X, y)

    scores = [score_SVC, score_RF, score_NN, score_HEU]
    labels = ['Score SVC', 'Score RF', 'Score NN', 'Score HEU']

    plot_scores(scores, labels)

    matrices = [confusion_SVC, confusion_RF, confusion_NN, confusion_HEU]
    matrices_dict = dict(zip(labels, matrices))

    plot_confusion_matrices(matrices_dict)

