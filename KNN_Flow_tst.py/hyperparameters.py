from sklearn.model_selection import GridSearchCV
from HR_pp import x_test,x_train,y_test,y_train
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

knn = KNeighborsClassifier()

# params_knn = {'n_neighbors': np.arange(3, 15), 'weights': ['uniform', 'distance'], 'p': [1, 2]}

# grid_knn = GridSearchCV(estimator = knn, param_grid = params_knn, scoring = 'recall', cv = 10)

# model_knn = grid_knn.fit(x_train,y_train)

# knn_estimator = model_knn.best_estimator_

# print(knn_estimator)

def knn_hyperparameter_selection(x_train,y_train,k=[3,15],weights = ['uniform','distance'],dist_calc=[1,2],
                                 metric='recall',cv_n=5):
    params_knn = {'n_neighbors':np.arange(k[0],k[1]),'weights':weights,'p':dist_calc}
    grid_knn = GridSearchCV(estimator = knn,param_grid = params_knn,scoring=metric,cv=cv_n)    
    model_knn=grid_knn.fit(x_train,y_train)
    knn_estimator = model_knn.best_estimator_
    return knn_estimator