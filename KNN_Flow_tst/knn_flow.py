# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.


import logging
import sys
import warnings
from urllib.parse import urlparse
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import json
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from HR_pp import x_train, x_test, y_train, y_test
from sklearn.neighbors import KNeighborsClassifier
from hyperparameters import knn_hyperparameter_selection



logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    train_x = x_train
    train_y = y_train
    test_x = x_test
    test_y = y_test

    # alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    # l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5



    with mlflow.start_run():
        knn = KNeighborsClassifier(n_neighbors=5)
        op = knn_hyperparameter_selection(train_x,train_y)

        n_neighbors,distance_type,weights = (op.n_neighbors,
        ['Manhattan' if op.p == 1 else 'Euclidean'],op.weights)
        op.fit(x_train,y_train)
        pred = op.predict(x_test)
        print(pred[:5])
        print(y_train[:5])
        print(type(pred))
        print(type(y_train.array))
        print(classification_report(y_test.array,pred))
        print(f"Number of Neighbors= {n_neighbors}")
        print(f"Distance Type= {distance_type[0]}")        
        print(f"Weight Type= {weights}")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                knn, "model", registered_model_name="ElasticnetWineModel"
            )
        else:
            mlflow.sklearn.log_model(knn, "model")
