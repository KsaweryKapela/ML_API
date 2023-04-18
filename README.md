# ML_API

This project aims to create classification models for the Convertype dataset
(https://archive.ics.uci.edu/ml/datasets/Covertype).

Code to create and evaluate models can be found in /clf_models directory, 
which also contains requirements for model construction and evaluation. 
Trained models can be found in /clf_models/models directory.
Main directory contains API, requirements for API and dockerfile.

## Models description

The Heuristic model categorizes input features based on the lowest sum of differences between the
input feature and overall means of features from the dataset sample (n=10,000). It collects means 
from three variables and uses them as a features: Elevation, Slope and Horizontal_Distance_To_Roadways.

The Shallow models include Random Forest Classification and Support Vector Classification, both
trained on a sample from the dataset (n=10,000) to save computing time. SVC could use additional
kernel tweaking, which was not done due to time limitations. The two models are saved as .sav files in
the clf_models/models.

The Deep model is a Neural Network also trained on a sample (n=10,000) from the dataset. Labels are encoded
with one-hot encoding. The model uses a parameter grid to find the best neurons amount, activation
functions, and optimizer. To save computing time, the parameters are limited, and batch size and
epochs are trained in a separate function. The model with the best performing parameters also uses
an additional sample (n=2,000) from the dataset to plot the eval loss function and accuracy. Saved
model can be found in the deep_model directory.

All four models are evaluated in the models_evaluation file. Accuracy is the metric used, and a
Confusion Matrix is plotted for each model. The relatively low performance of neural networks could
be explained by the limited sample size and lack of time and computational resources to expand the
parameter grid and try different architectures. Heuristic model isn't accurate at all.

## API

The api.py file is a Flask app created with three endpoints. All endpoints use JSON file with a one or
more set of features as payload and return values predicted by specific model. One of the endpoints serves 
two models, which the user can choose from by providing a different variable in the URL.

Dockerfile was created for a project, from which API image can be build and run.

### Example code to make API request on local server:
import requests
from clf_models.helpers import open_covtype_sample, X_y_split

X, y = X_y_split(open_covtype_sample('Eval_2')) # Load dataset and get features
json = {'features': X.tolist()} # Turn arr to list so it's JSON serializable
r = requests.post('http://127.0.0.1:3000/shallow/rfc', json=json) # Make a request to url
print(r.json()) # Get model predictions

### Possible urls:
/shallow/rfc
/shallow/svc
/deep
/heuristic