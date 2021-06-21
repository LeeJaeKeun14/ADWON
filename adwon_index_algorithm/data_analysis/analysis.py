
from adwon_index_algorithm.input_data import data_handling
import pandas as pd
import numpy as np
import math

def print_analysis():
    
    df = data_handling.print_cost()
    p_df = df.groupby("광고").sum()[["count", "cost"]].reset_index()
    p_df = p_df.iloc[[0,1,3,2]]
    p_df = p_df.reset_index(drop = True)
    p_df["coef"] = data_handling.regression()
    
    df = pd.DataFrame([p_df["count"] / p_df["cost"] * p_df["coef"] / sum(p_df["count"] / p_df["cost"] * p_df["coef"]) * 100])
    
    df[0] = math.ceil(df[0])
    df[1] = df[1].astype("int")
    df[2] = df[2].astype("int")
    df[3] = math.ceil(df[3])
    df = df.append(p_df["count"])

    df.columns=["TV", "Mobile", "PC", "DOOH"]
    df = df.rename(index={0: "index"})
    
    return df

def __sum_data(t_df):
    c_df = t_df

    TV = c_df[c_df["광고"] == "TV"]
    TV = TV.sort_values("count", ascending=False).reset_index(drop = True)
    TV_sum = TV.cost.sum()

    인터넷 = c_df[c_df["광고"] == "인터넷"]
    인터넷 = 인터넷.sort_values("count", ascending=False).reset_index(drop = True)
    인터넷_sum = 인터넷.cost.sum()

    모바일 = c_df[c_df["광고"] == "모바일"]
    모바일 = 모바일.sort_values("count", ascending=False).reset_index(drop = True)
    모바일_sum = 모바일.cost.sum()

    옥외광고 = c_df[c_df["광고"] == "옥외광고"]
    옥외광고 = 옥외광고.sort_values("count", ascending=False).reset_index(drop = True)
    옥외광고_sum = 옥외광고.cost.sum()

    return TV_sum, 인터넷_sum, 모바일_sum, 옥외광고_sum

def print_handling():

    t_df = data_handling.print_cost()

    key = ["TV", "Mobile", "PC", "DOOH"]
    TV_sum, 인터넷_sum, 모바일_sum, 옥외광고_sum = __sum_data(t_df)
    
    TV = t_df[t_df["광고"] == "TV"]
    모바일 = t_df[t_df["광고"] == "모바일"]
    인터넷 = t_df[t_df["광고"] == "인터넷"]
    옥외광고 = t_df[t_df["광고"] == "옥외광고"]

    c_df = pd.DataFrame([TV[TV.columns[6:]].sum()])
    c_df = c_df.append(pd.DataFrame([모바일[모바일.columns[6:]].sum()]))
    c_df = c_df.append(pd.DataFrame([인터넷[인터넷.columns[6:]].sum()]))
    c_df = c_df.append(pd.DataFrame([옥외광고[옥외광고.columns[6:]].sum()]))
    
    total_df = c_df[c_df.columns[0:7]]
    total_df = total_df.T
    total_df.columns = key
    col = c_df.columns[0:7]

    male_df = c_df[c_df.columns[7:14]]
    male_df = male_df.T
    male_df.columns = key
    male_df = male_df.reset_index(drop = True)
    male_df = male_df.rename(index={0: "10대", 1: "20대", 2: "30대", 3: "40대", 4: "50대", 5: "60대 이상", 6: "전체"})

    female_df = c_df[c_df.columns[14:21]]
    female_df = female_df.T
    female_df.columns = key
    female_df = female_df.reset_index(drop = True)
    female_df = female_df.rename(index={0: "10대", 1: "20대", 2: "30대", 3: "40대", 4: "50대", 5: "60대 이상", 6: "전체"})

    t_df = pd.concat([total_df, male_df, female_df], axis = 1)
    
    return t_df

def print_analysis2():

    t_df = data_handling.print_cost()
    r_df = data_handling.total_regression()

    key = ["TV", "Mobile", "PC", "DOOH"]
    TV_sum, 인터넷_sum, 모바일_sum, 옥외광고_sum = __sum_data(t_df)
    
    TV = t_df[t_df["광고"] == "TV"]
    모바일 = t_df[t_df["광고"] == "모바일"]
    인터넷 = t_df[t_df["광고"] == "인터넷"]
    옥외광고 = t_df[t_df["광고"] == "옥외광고"]

    c_df = pd.DataFrame([TV[TV.columns[6:]].sum()])
    c_df = c_df.append(pd.DataFrame([모바일[모바일.columns[6:]].sum()]))
    c_df = c_df.append(pd.DataFrame([인터넷[인터넷.columns[6:]].sum()]))
    c_df = c_df.append(pd.DataFrame([옥외광고[옥외광고.columns[6:]].sum()]))
    
    
    total_df = c_df[c_df.columns[0:7]]
    total_df = total_df.T
    total_df.columns = key
    total_df = total_df * r_df.iloc[:,:4]
    col = c_df.columns[0:7]

    male_df = c_df[c_df.columns[7:14]]
    male_df = male_df.T
    male_df.columns = key
    male_df = male_df.reset_index(drop = True)
    male_df = male_df.rename(index={0: "10대", 1: "20대", 2: "30대", 3: "40대", 4: "50대", 5: "60대 이상", 6: "전체"})
    male_df = male_df * r_df.iloc[:,4:8]

    female_df = c_df[c_df.columns[14:21]]
    female_df = female_df.T
    female_df.columns = key
    female_df = female_df.reset_index(drop = True)
    female_df = female_df.rename(index={0: "10대", 1: "20대", 2: "30대", 3: "40대", 4: "50대", 5: "60대 이상", 6: "전체"})
    female_df = female_df * r_df.iloc[:,8:]

    total_df = pd.DataFrame(np.array(total_df) / np.array([TV_sum , 모바일_sum, 인터넷_sum, 옥외광고_sum]))
    data_sum = np.array(total_df.sum(axis = 1))[np.newaxis].T
    total_df = pd.DataFrame(np.array(total_df) / data_sum * 100)
    inx = dict(zip(range(7), col))
    total_df = total_df.rename(index=inx)
    total_df[0] = total_df[0].apply(lambda x : math.ceil(x))
    total_df[1] = total_df[1].astype("int")
    total_df[2] = total_df[2].astype("int")
    total_df[3] = total_df[3].apply(lambda x : math.ceil(x))
    total_df.columns = key

    male_df = pd.DataFrame(np.array(male_df) / np.array([TV_sum , 모바일_sum, 인터넷_sum, 옥외광고_sum]))
    data_sum = np.array(male_df.sum(axis = 1))[np.newaxis].T
    male_df = pd.DataFrame(np.array(male_df) / data_sum * 100)
    inx = dict(zip(range(7), col))
    male_df = male_df.rename(index=inx)
    male_df[0] = male_df[0].apply(lambda x : math.ceil(x))
    male_df[1] = male_df[1].astype("int")
    male_df[2] = male_df[2].astype("int")
    male_df[3] = male_df[3].apply(lambda x : math.ceil(x))
    male_df.columns = key

    female_df = pd.DataFrame(np.array(female_df) / np.array([TV_sum , 모바일_sum, 인터넷_sum, 옥외광고_sum]))
    data_sum = np.array(female_df.sum(axis = 1))[np.newaxis].T
    female_df = pd.DataFrame(np.array(female_df) / data_sum * 100)
    inx = dict(zip(range(7), col))
    female_df = female_df.rename(index=inx)
    female_df[0] = female_df[0].apply(lambda x : math.ceil(x))
    female_df[1] = female_df[1].astype("int")
    female_df[2] = female_df[2].astype("int")
    female_df[3] = female_df[3].apply(lambda x : math.ceil(x))
    female_df.columns = key
    
    t_df = pd.concat([total_df, male_df, female_df], axis = 1)
    
    return t_df

def print_TV():

    c_df = data_handling.print_cost()
    c_df = c_df[c_df["광고"] == "TV"]
    ##
    c_df["index"] = c_df["count"] / c_df["cost"]
    c_df["index"] = round(c_df["index"] / c_df["index"].sum() * 100, 1)
    p_df = c_df[["채널", "count", "index"]]
    p_df = p_df.sort_values("index", ascending=False).reset_index(drop = True)

    return p_df

def print_모바일():

    c_df = data_handling.print_cost()
    c_df = c_df[c_df["광고"] == "모바일"]

    c_df["index"] = c_df["count"] / c_df["cost"]
    c_df["index"] = round(c_df["index"] / c_df["index"].sum() * 100, 1)
    p_df = c_df[["채널", "count", "index"]]
    p_df = p_df.sort_values("index", ascending=False).reset_index(drop = True)

    return p_df

def print_인터넷():

    c_df = data_handling.print_cost()
    c_df = c_df[c_df["광고"] == "인터넷"]

    c_df["index"] = c_df["count"] / c_df["cost"]
    c_df["index"] = round(c_df["index"] / c_df["index"].sum() * 100, 1)
    p_df = c_df[["채널", "count", "index"]]
    p_df = p_df.sort_values("index", ascending=False).reset_index(drop = True)

    return p_df

def print_옥외광고():

    c_df = data_handling.print_cost()
    c_df = c_df[c_df["광고"] == "옥외광고"]

    c_df["index"] = c_df["count"] / c_df["cost"]
    c_df["index"] = round(c_df["index"] / c_df["index"].sum() * 100, 1)
    p_df = c_df[["채널", "count", "index"]]
    p_df = p_df.sort_values("index", ascending=False).reset_index(drop = True)

    return p_df
