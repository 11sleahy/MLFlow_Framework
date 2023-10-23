from sqlalchemy import create_engine
from boto3.session import Session
import pandas as pd
import json, os, boto3
import gzip, shutil
import awswrangler as awsw
import io
import json
import gspread
import numpy as np



# Service account for google sheets API
sa = gspread.service_account('***.json')
# Access Delivery Operations
sh = sa.open("***")
# Access volume reference worksheet
wks = sh.worksheet("***")
# Point to new worksheet to export updates 
op_wks = sh.worksheet("***")