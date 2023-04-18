from flask import Flask
from flask import jsonify
from flask import request
from tensorflow import keras
import pickle
import numpy as np
from clf_models.heuristic_classification import SimpleHeuristicModel

app = Flask(__name__)


@app.route("/shallow/<model>", methods=['POST'])
def return_shallow_model(model):

    requested_data = request.get_json()
    features = requested_data['features']

    if not isinstance(features[0], list):
        features = [features]

    if model == 'rfc':

        RF_model = pickle.load(open('clf_models/models/RF_clf.sav', 'rb'))
        prediction = RF_model.predict(features)
    
    
    elif model == 'svc':

        RF_model = pickle.load(open('clf_models/models/SVC_clf.sav', 'rb'))
        prediction = RF_model.predict(features)
    
    else:
        return jsonify(error='Provide rfc or svc model names')

    return jsonify(prediction=prediction.tolist())


@app.route("/deep", methods=['POST'])
def return_deep_model():

    requested_data = request.get_json()
    features = requested_data['features']

    if not isinstance(features[0], list):
        features = [features]

    NN_model = keras.models.load_model('clf_models/models/NN_clf')
    NN_pred = NN_model.predict(features)
    NN_pred = np.round(NN_pred).argmax(axis=1)
    NN_pred += 1

    return jsonify(prediction=NN_pred.tolist())


@app.route("/heuristic", methods=['POST'])
def return_heuristic_model():

    requested_data = request.get_json()
    features = requested_data['features']

    if not isinstance(features[0], list):
        features = [features]
    
    heuristic_model = SimpleHeuristicModel()
    heuristic_pred = heuristic_model.predict(features)

    return jsonify(prediction=heuristic_pred)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
