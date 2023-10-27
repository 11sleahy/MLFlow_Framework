import sys
import os

def infolder_file( filename ):
    afname = os.path.abspath(__file__)
    current_folder = os.path.dirname(afname)
    uf = os.path.join(current_folder, filename )
    return uf

a = sys.path.append(infolder_file('predictive_maintenance.csv'))
print(a)

# from config import csv_config
# import pandas as pd

# pth = csv_config.infolder_file('predictive_maintenance.csv')

# print(pth)
