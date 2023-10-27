import os
from pathlib import Path
import sys
import pandas as pd

def infolder_file(fl):
    afname = os.path.abspath(__file__)
    current_folder = os.path.dirname(afname)
    uf = os.path.join(current_folder,fl)
    return uf


fpth = infolder_file('HR_Employee_Attrition_Dataset.csv')
print(fpth)
# import yaml

# with open(r'C:\Users\simle\OneDrive\Documents\Apexon\MLFlow\config\csv_config.yaml','r') as f:
#     tmp = yaml.safe_load(f)

# print(tmp)
