import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split



def knn_app(dt_x,dt_y,test_size,k_min,k_max):
    x_train,x_test,y_train,y_test = train_test_split(dt_x,dt_y,test_size = test_size)
    knn = KNeighborsClassifier()
    train_error = []
    test_error = []
    knn_many_split = {}
    error_df_knn = pd.DataFrame()
    features = dt_x.columns
    for k in range(k_min, k_max):
        train_error = []
        test_error = []        
        lista = []        
        knn = KNeighborsClassifier(n_neighbors = k)
        
        for i in range(30):
            x_train_new, x_val, y_train_new, y_val = train_test_split(x_train, y_train, test_size = test_size)
            knn.fit(x_train_new, y_train_new)
            train_error.append(1 - knn.score(x_train_new, y_train_new))          
            test_error.append(1 - knn.score(x_val, y_val))
        
        lista.append(sum(train_error)/len(train_error))        
        lista.append(sum(test_error)/len(test_error))        
        knn_many_split[k] = lista
    knn_many_split

