
# 임시 데이터 사용하기.
import numpy as np
import pandas as pd
import random as rd
import datetime as dt
import sys
from adwon_index_algorithm.input_data import read_data
import warnings
warnings.filterwarnings(action='ignore')
    
PATH = sys.path[0] + "/adwon_index_algorithm/input_data"

def __make_datetime(x):
    x     = str(x)
    year  = int(x[:4])
    month = int(x[5:7])
    day   = int(x[8:10])
    hour  = int(x[11:13])
    mim  = int(x[14:16])
    #sec  = int(x[15:])
    return dt.datetime(year, month, day, hour, mim)
    
def read_df():
    read_df = pd.read_csv(PATH + "/created_data/new_data.csv")
    read_df["timestemp"] = read_df["timestemp"].apply(lambda x : __make_datetime(x))
    return read_df

def read_user_df():
    user_df = pd.read_csv(PATH + "/created_data/user_data.csv")
    return user_df
