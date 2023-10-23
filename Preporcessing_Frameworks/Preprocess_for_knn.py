import sys
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

sys.path.append(r'C:\Users\simle\OneDrive\Documents\Apexon\MLFlow')
from Models import KNN


#sys.path.append(r'C:\Users\simle\OneDrive\Documents\Apexon\MLFlow')
dt = pd.read_csv(r'C:\Users\simle\OneDrive\Documents\Apexon\MLFlow\Preporcessing_Frameworks\KNN.csv')
dt=dt.drop(['Unnamed: 0'],axis=1)

def scaler(num_df):
    scaler = StandardScaler()
    scaled_cols = num_df.select_dtypes(include='number')
    op = data_scaled = pd.DataFrame(scaler.fit_transform(scaled_cols),columns = scaled_cols.columns)
    return op

def get_dummies(cat_df):
    dummy_cols = cat_df.select_dtypes(include='object')
    op = pd.get_dummies(data=dummy_cols,columns=dummy_cols,drop_first=True)
    return op

dt_cat=dt[['Priority','Affected Facility','Identification Source','Significance']]
df_cat_dum = pd.get_dummies(data=dt,columns=['Priority','Affected Facility','Identification Source'],drop_first=True,dtype=int)
df_x = df_cat_dum.drop(columns=['Significance'])
print(df_cat_dum)

