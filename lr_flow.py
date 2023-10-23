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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import json

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# with open("parameters.yaml",'r') as f:
#     params = yaml.safe_load(f)
# print(params)

with open('param.json') as f:
    d = json.load(f)


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

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # def optimize_yield(param):
    #     r1 = []
    #     r2=[]
    #     i1=[]
    #     lst = {r1,r2,i1}
    #     for i in range(.1,1,.1):
    #         i1 = i*10
    #         alpha = i
    #         l1_ratio = i
    #         lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    #         lr.fit(train_x, train_y)

    #         predicted_qualities = lr.predict(test_x)

    #         (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    #         lst.append(rmse,r2,i1)
    
    temp_dict = {"rmse":[],"mae":[],"r2":[],"ualpha":[],"ul1":[]}
    for i in d['alpha']:
        for j in d['l1']:
            lr = ElasticNet(alpha=i, l1_ratio=j, random_state=42)
            lr.fit(train_x, train_y)

            predicted_qualities = lr.predict(test_x)
            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
            temp_dict['rmse'].append(round(rmse,4))
            temp_dict['mae'].append(round(mae,4))
            temp_dict['r2'].append(round(r2,4))
            temp_dict['ualpha'].append(i)
            temp_dict['ul1'].append(j)

    p = d['optimization_param'][0]
    ind = temp_dict[p].index(max(temp_dict[p]))


    rmse_usage_alpha = temp_dict['ualpha'][ind]
    rmse_usage_l1 = temp_dict['ul1'][ind]
    # r2_usage_alpha = temp_dict['ualpha'][ind]
    # r2_usage_l1 = temp_dict['ul1'][ind]


    with mlflow.start_run():
#        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr = ElasticNet(alpha=rmse_usage_alpha, l1_ratio=rmse_usage_l1, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model optimize for {p}(alpha={rmse_usage_alpha:f}, l1_ratio={rmse_usage_l1:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)
