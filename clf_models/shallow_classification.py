from helpers import open_covtype_sample, X_y_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle

def tweak_param(clf, clf_param, X, y, opening_value, method='*', increase=10, loops=7):
    param_value = opening_value
    best_acc = 0
    best_param_value = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for i in range(loops):
        current_clf = clf()
        current_params = {clf_param: param_value}
        current_clf.set_params(**current_params)
        current_clf.fit(X_train, y_train)
        pred = current_clf.predict(X_test)
        acc = accuracy_score(pred, y_test)
        print(acc)

        if acc > best_acc:
            best_acc = acc
            best_param_value = param_value

        if method == '*':
            param_value *= increase

        elif method == '+':
            param_value += increase
        
        print(f'Loop {i + 1}/{loops}')

    print(f'{clf()} best acc = {best_acc} | Best {clf_param} value = {best_param_value}')
    return best_acc, best_param_value


def save_model(model, name, X, y):
    model.fit(X, y)
    path = 'clf_models/models/'
    pickle.dump(f'{path}{model}', open(name, 'wb'))
    print(f'{model} succesfully saved as {name}')


if __name__ == "__main__":

    df = open_covtype_sample()
    X, y = X_y_split(df)

    clf_SVC = SVC
    clf_RF = RandomForestClassifier
    acc_SVC, C_param = tweak_param(clf_SVC, 'C', X, y, opening_value=0.01, method='*', increase=10, loops=7)
    acc_RF, depth_param = tweak_param(clf_RF, 'max_depth', X, y, opening_value=2, method='+', increase=2, loops=10)

    final_SVC = SVC(C=C_param)
    save_model(final_SVC, 'SVC_clf.sav', X, y)
    final_RF = RandomForestClassifier(max_depth=depth_param)
    save_model(final_RF, 'RF_clf.sav', X, y)