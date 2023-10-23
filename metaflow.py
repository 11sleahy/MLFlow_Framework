from metaflow import step,flowSpec,conda_base
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import sklearn

class HelloFlow(flowSpec):
    @step
    def start(self):
        print("Initiated MF Workflow")
        self.next(self.run)

    @step
    def run_knn(self):
        sklearn.knn()

    @step
    def run_k_means(self):
        sklearn.kmeans()