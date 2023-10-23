import pandas as pd
import matplotlib as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes



dt = pd.read_csv('predictive_maintenance.csv')
print(dt)

refined_dt = dt[['Priority','Affected Facility','Identification Source','Significance']]
print(refined_dt)

# kmode = KModes(n_clusters=3,init="random",verbose=1)
# clusters = kmode.fit()


sse = {} 

# Iterate for a range of Ks and fit the scaled data to the algorithm. 
# Use inertia attribute from the clustering object and store the inertia value for that K 
# for k in range(1, 10):
#     kmeans = KMeans(n_clusters = k, random_state = 1).fit(dt)
    
#     sse[k] = kmeans.inertia_

# # Elbow plot
# plt.figure()

# plt.plot(list(sse.keys()), list(sse.values()), 'bx-')

# plt.xlabel("Number of cluster")

# plt.ylabel("SSE")

# plt.show()
