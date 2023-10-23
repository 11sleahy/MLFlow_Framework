import pandas as pd
from sklearn.preprocessing import StandardScaler


dt = pd.read_csv('predictive_maintenance.csv')
print(dt)

dt.info()