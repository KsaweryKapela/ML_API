# ML_API

This project aims to create classification models for the Convertype dataset
(https://archive.ics.uci.edu/ml/datasets/Covertype). The Exploring_dataset file contains information
on the dataset.

The Heuristic model categorizes input features based on the lowest sum of differences between the
input feature and overall means of features from the dataset. It uses three variables as features:
Elevation, Slope and Horizontal_Distance_To_Roadways.

The Shallow models include Random Forest Classification and Support Vector Classification, both
trained on a sample from the dataset (n=10,000) to save computing time. SVC could use additional
kernel tweaking, which was not done due to time limitations. The two models are saved as .sav files in
the folder.

The Deep model is a Neural Network also trained on a sample (n=10,000) from the dataset. Labels are encoded
with one-hot encoding. The model uses a parameter grid to find the best neurons amount, activation
functions, and optimizer. To save computing time, the parameters are limited, and batch size and
epochs are trained in a separate function. The model with the best performing parameters also uses
an additional sample (n=2,000) from the dataset to plot the eval loss function and accuracy. Saved
model can be found in the deep_model directory.

All four models are evaluated in the models_evaluation file. Accuracy is the metric used, and a
Confusion Matrix is plotted for each model. The relatively low performance of neural networks could
be explained by the limited sample size and lack of time and computational resources to expand the
parameter grid and try different architectures.

The API file is a Flask API created with three endpoints. All endpoints take a JSON file with a one or
more set of features as input and return values predicted by specific model. One of the endpoints serves 
two models, which the user can choose from by providing a different variable in the URL. The requests
made to all API endpoints can be seen in the using_API file.

Docker container image serving API can be found in a link below:
https://hub.docker.com/repository/docker/ksawerykapela/api_ml/general