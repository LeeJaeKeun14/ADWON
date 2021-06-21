
import numpy as np
import pandas as pd
import random as rd
import datetime as dt
import statsmodels.api as sm
import sys
from adwon_index_algorithm.input_data.make_temp_data import read_df
import warnings
warnings.filterwarnings(action='ignore')

PATH = sys.path[0] + "/adwon_index_algorithm/input_data"
    
def __요일(x):
#     if x == 0: return "월"
#     elif x == 1: return "화"
#     elif x == 2: return "수"
#     elif x == 3: return "목"
#     elif x == 4: return "금"
#     elif x == 5: return "토"
#     elif x == 6: return "일"
    if x == 0: return "평일"
    elif x == 1: return "평일"
    elif x == 2: return "평일"
    elif x == 3: return "평일"
    elif x == 4: return "평일"
    elif x == 5: return "토요일"
    elif x == 6: return "일요일"

def __make_datetime(x):
    x     = str(x)
    year  = int(x[:4])
    month = int(x[5:7])
    day   = int(x[8:10])
    hour  = int(x[11:13])
    mim  = int(x[14:16])
    #sec  = int(x[15:])
    return dt.datetime(year, month, day, hour, mim)
    
def print_cost():
    df = read_df()

    df["요일"] = df.timestemp.apply(lambda x : __요일(x.weekday()))
    df["시"] = df.timestemp.apply(lambda x : "0" + str(x.hour) if x.hour < 10 else str(x.hour))
    df["분"] = df.timestemp.apply(lambda x : "0" + str(x.minute) if x.minute < 10 else str(x.minute))
    # df["분"] = "00"
    df["시간"] = df["시"] + ":" + df["분"]
    # df = __인접시간(df)

    idx = df[df["광고"] == "TV"].reset_index()["index"]

    df.채널 = df.채널.astype("str")
    df.cost = df.cost.astype("str")
    df["count"] = df.광고 + "_"+ df.채널
    # df["count"][idx] = df.loc[idx]["count"] + "_" + df.loc[idx]["요일"] + "_" + df.loc[idx]["등급"]
    df["count"][idx] = df.loc[idx]["count"] + "_" + df.loc[idx]["요일"] + "_" + df.loc[idx]["cost"]
    idx = df[df["광고"] != "TV"].reset_index()["index"]
    df["count"][idx] = df.loc[idx]["count"] + "_" + df.loc[idx]["cost"]

    t_df = df["count"].value_counts().reset_index()
    t_df = t_df.sort_values("index").reset_index(drop = True)

    t_df["광고"] = t_df["index"].apply(lambda x : x.split("_")[0])
    t_df["채널"] = t_df["index"].apply(lambda x : x.split("_")[1])
    t_df["요일"] = t_df["index"].apply(lambda x : x.split("_")[2] if x.split("_")[0] == "TV" else None)
    t_df["cost"] = t_df["index"].apply(lambda x : x.split("_")[3] if x.split("_")[0] == "TV" else x.split("_")[2])

    ##
    tmp_df = df
    tmp_df = tmp_df[tmp_df["연령"] >= 10]
    tmp_df = tmp_df[tmp_df["연령"] < 20]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "10대"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df
    tmp_df = tmp_df[tmp_df["연령"] >= 20]
    tmp_df = tmp_df[tmp_df["연령"] < 30]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "20대"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df
    tmp_df = tmp_df[tmp_df["연령"] >= 30]
    tmp_df = tmp_df[tmp_df["연령"] < 40]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "30대"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df
    tmp_df = tmp_df[tmp_df["연령"] >= 40]
    tmp_df = tmp_df[tmp_df["연령"] < 50]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "40대"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df
    tmp_df = tmp_df[tmp_df["연령"] >= 50]
    tmp_df = tmp_df[tmp_df["연령"] < 60]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "50대"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df
    tmp_df = tmp_df[tmp_df["연령"] >= 60]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "60대 이상"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df
    tmp_df = tmp_df["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "전체"]
    t_df = t_df.merge(tmp_df, "left", "index")

    ##
    tmp_df = df[df["성별"] == "남"]
    tmp_df = tmp_df[tmp_df["연령"] >= 10]
    tmp_df = tmp_df[tmp_df["연령"] < 20]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "m_10"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "남"]
    tmp_df = tmp_df[tmp_df["연령"] >= 20]
    tmp_df = tmp_df[tmp_df["연령"] < 30]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "m_20"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "남"]
    tmp_df = tmp_df[tmp_df["연령"] >= 30]
    tmp_df = tmp_df[tmp_df["연령"] < 40]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "m_30"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "남"]
    tmp_df = tmp_df[tmp_df["연령"] >= 40]
    tmp_df = tmp_df[tmp_df["연령"] < 50]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "m_40"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "남"]
    tmp_df = tmp_df[tmp_df["연령"] >= 50]
    tmp_df = tmp_df[tmp_df["연령"] < 60]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "m_50"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "남"]
    tmp_df = tmp_df[tmp_df["연령"] >= 60]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "m_60"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "남"]
    tmp_df = tmp_df["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "m_total"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "여"]
    tmp_df = tmp_df[tmp_df["연령"] >= 10]
    tmp_df = tmp_df[tmp_df["연령"] < 20]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "f_10"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "여"]
    tmp_df = tmp_df[tmp_df["연령"] >= 20]
    tmp_df = tmp_df[tmp_df["연령"] < 30]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "f_20"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "여"]
    tmp_df = tmp_df[tmp_df["연령"] >= 30]
    tmp_df = tmp_df[tmp_df["연령"] < 40]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "f_30"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "여"]
    tmp_df = tmp_df[tmp_df["연령"] >= 40]
    tmp_df = tmp_df[tmp_df["연령"] < 50]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "f_40"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "여"]
    tmp_df = tmp_df[tmp_df["연령"] >= 50]
    tmp_df = tmp_df[tmp_df["연령"] < 60]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "f_50"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "여"]
    tmp_df = tmp_df[tmp_df["연령"] >= 60]["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "f_60"]
    t_df = t_df.merge(tmp_df, "left", "index")

    tmp_df = df[df["성별"] == "여"]
    tmp_df = tmp_df["count"].value_counts().reset_index()
    tmp_df.columns = ["index", "f_total"]
    t_df = t_df.merge(tmp_df, "left", "index")
    
    t_df["cost"] = t_df["cost"].astype("int")
    # t_df["index"] = t_df["index"].apply(lambda x : "_".join(x.split("_")[:-1]) if x.split("_")[0] == "TV" else x)
    
    return t_df

def age_group(df, 연령):
    
    if 연령 != "total":
        성별, 연령 = 연령.split("_")
        if 성별 == "남":
            df = df[df["성별"] == "남"]
        elif 성별 == "여":
            df = df[df["성별"] == "여"]
        
        if 연령 == "10대":
            df = df[df["연령"] >= 10]
            df = df[df["연령"] < 20]
        elif 연령 == "20대":
            df = df[df["연령"] >= 20]
            df = df[df["연령"] < 30]
        elif 연령 == "30대":
            df = df[df["연령"] >= 30]
            df = df[df["연령"] < 40]
        elif 연령 == "40대":
            df = df[df["연령"] >= 40]
            df = df[df["연령"] < 50]
        elif 연령 == "50대":
            df = df[df["연령"] >= 50]
            df = df[df["연령"] < 60]
        elif 연령 == "60대":
            df = df[df["연령"] >= 60]
    return df

# 임시 데이터 사용하기.

def handling(연령):
    ## 데이터 베이스에 접속하기.
    # df = make_temp_data.make_data()
    df = read_df()
    df = age_group(df, 연령)


    ## 임시 데이터 읽기
    names = list(df.광고명.unique())
    names.sort()
    df["week"] = df.timestemp.apply(lambda x : x.week)
    df.week = df.week.astype("str")
    df["광고_week"] = df.광고 + "_" + df.week

    t_df = df["광고_week"].value_counts().reset_index()
    t_df = t_df.sort_values("index").reset_index(drop = True)

    t_df["광고"] = t_df["index"].apply(lambda x : x.split("_")[0])
    t_df["week"] = t_df["index"].apply(lambda x : x.split("_")[1])

    reg_df = t_df[t_df["광고"] == "TV"][["week", "광고_week"]]
    r_df = t_df[t_df["광고"] == "모바일"][["week", "광고_week"]]
    r_df2 = t_df[t_df["광고"] == "인터넷"][["week", "광고_week"]]
    r_df3 = t_df[t_df["광고"] == "옥외광고"][["week", "광고_week"]]

    reg_df = reg_df.merge(r_df, "left", "week")
    reg_df = reg_df.merge(r_df2, "left", "week")
    reg_df = reg_df.merge(r_df3, "left", "week")

    reg_df.columns = ["week", "TV", "모바일", "인터넷", "옥외광고"]
    reg_df["week"] = reg_df["week"].astype("int")
    reg_df = reg_df[reg_df["week"] < 17]
    reg_df = reg_df.sort_values("week").reset_index(drop = True)

    y = pd.read_csv("./adwon_index_algorithm/input_data/created_data/y_data.csv")
    y = y.iloc[:, 9:]
    y["9"] = y["9"].apply(__make_datetime)
    y["week"] = y["9"].apply(lambda x : x.week)
    y = y["week"].value_counts().reset_index()
    y = y.sort_values("index").reset_index(drop = True)
    y.columns = ["week", "count"]

    reg_df = reg_df.merge(y, "left", "week").iloc[2:].reset_index(drop = True)
    reg_df = reg_df.rename(columns = {"옥외광고" : "옥외", "count": "y"})
    
    return reg_df

def print_regression(연령 = "total"):
    final_df = handling(연령)
    model = sm.OLS.from_formula("y ~ np.log(TV) + np.log(모바일) + np.log(인터넷) + np.log(옥외)", data=final_df)
    result = model.fit()
    print(result.summary())

def regression(연령 = "total"):
    final_df = handling(연령)
    model = sm.OLS.from_formula("y ~ np.log(TV) + np.log(모바일) + np.log(인터넷) + np.log(옥외)", data=final_df)
    result = model.fit()
    p = result.params
    # 실재 데이터일 경우 절대값 필요 없음
    return np.absolute(list(p[1:]))

def p_regression(연령 = "total"):
    final_df = handling(연령)
    model = sm.OLS.from_formula("y ~ np.log(TV) + np.log(모바일) + np.log(인터넷) + np.log(옥외)", data=final_df)
    result = model.fit()
    p = result.params
    # 실재 데이터일 경우 절대값 필요 없음
    return np.absolute(list(p))

## 모든 회귀계수 출력
def total_regression():
    ls = ["전체_10대", "전체_20대", "전체_30대", "전체_40대", "전체_50대", "전체_60대", "전체_전체"]
    r_df = pd.DataFrame()
    for name in ls:
        final_df = handling(name)
        model = sm.OLS.from_formula("y ~ np.log(TV) + np.log(모바일) + np.log(인터넷) + np.log(옥외)", data=final_df)
        result = model.fit()
        p = result.params
        r_df = r_df.append(pd.DataFrame(np.absolute([list(p[1:])])))
    r_df = r_df.reset_index(drop = True)
    r_df.columns = ["TV", "Mobile", "PC", "DOOH"]
    
    ls = ["남_10대", "남_20대", "남_30대", "남_40대", "남_50대", "남_60대", "남_전체"]
    r_df2 = pd.DataFrame()
    for name in ls:
        final_df = handling(name)
        model = sm.OLS.from_formula("y ~ np.log(TV) + np.log(모바일) + np.log(인터넷) + np.log(옥외)", data=final_df)
        result = model.fit()
        p = result.params
        r_df2 = r_df2.append(pd.DataFrame(np.absolute([list(p[1:])])))
        r_df2 = r_df2.reset_index(drop = True)
    r_df2.columns = ["TV", "Mobile", "PC", "DOOH"]
    r_df = pd.concat([r_df, r_df2], axis = 1)
    
    ls = ["여_10대", "여_20대", "여_30대", "여_40대", "여_50대", "여_60대", "여_전체"]
    r_df2 = pd.DataFrame()
    for name in ls:
        final_df = handling(name)
        model = sm.OLS.from_formula("y ~ np.log(TV) + np.log(모바일) + np.log(인터넷) + np.log(옥외)", data=final_df)
        result = model.fit()
        p = result.params
        r_df2 = r_df2.append(pd.DataFrame(np.absolute([list(p[1:])])))
        r_df2 = r_df2.reset_index(drop = True)
    r_df2.columns = ["TV", "Mobile", "PC", "DOOH"]
    r_df = pd.concat([r_df, r_df2], axis = 1)
    r_df = r_df.rename(index={0: "10대", 1: "20대", 2: "30대", 3: "40대", 4: "50대", 5: "60대 이상", 6: "전체"})
    return r_df

def Mnb():
    # 데이터 전처리 : 데이터 셋 분리
    X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:,:-1], df.iloc[:,-1], test_size=0.4, random_state=1)
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pickle.dump(model, open(PATH + "Mnb.pkl", "wb"))
