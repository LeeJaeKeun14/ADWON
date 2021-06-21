
from adwon_index_algorithm.input_data import data_handling
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def __광고_COST(광고, category, limit):
    광고_cost = 0
    광고_limit = limit
    광고_list = []
    for i in range(len(광고)):
        광고_cost += 광고["cost"][i]
        if 광고_cost > 광고_limit:
            광고_cost -= 광고["cost"][i]
            break
        광고_list.append(광고[category][i])
    return sum(광고_list)

def __광고리스트(광고, limit):
    광고_cost = 0
    광고_limit = limit
    광고_list = pd.DataFrame()
    for i in range(len(광고)):
        광고_cost += 광고["cost"][i]
        if 광고_cost > 광고_limit:
            광고_cost -= 광고["cost"][i]
            break
        광고_list = 광고_list.append(광고.iloc[i])
    return 광고_list
    
    

def obtimiz(광고명, total = 100000, category = "전체_전체", tv_p = None, 모바일_p = None, 옥외광고_p = None, 인터넷_p = None):
    if tv_p != None:
        return obtimiz_1(광고명, total, category, tv_p)
    elif 모바일_p != None:
        return obtimiz_2(광고명, total, category, 모바일_p)
    elif 옥외광고_p != None:
        return obtimiz_3(광고명, total, category, 옥외광고_p)
    elif 인터넷_p != None:
        return obtimiz_4(광고명, total, category, 인터넷_p)

    cal_df = data_handling.print_cost(광고명)
    p = data_handling.p_regression(category)
    category = category.split("_")[1]
    TV = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "TV"]
    TV["per"] = TV[category] / TV["cost"]
    모바일 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "모바일"]
    모바일["per"] = 모바일[category] / 모바일["cost"]
    옥외광고 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "옥외광고"]
    옥외광고["per"] = 옥외광고[category] / 옥외광고["cost"]
    인터넷 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "인터넷"]
    인터넷["per"] = 인터넷[category] / 인터넷["cost"]

    ## 특정 연령 정렬 기능 필요
    TV = TV.sort_values("per", ascending=False).reset_index(drop = True)
    TV1 = TV
    TV = TV[[category, "cost"]]
    모바일 = 모바일.sort_values("per", ascending=False).reset_index(drop = True)
    모바일1 = 모바일
    모바일 = 모바일[[category, "cost"]]
    옥외광고 = 옥외광고.sort_values("per", ascending=False).reset_index(drop = True)
    옥외광고1 = 옥외광고
    옥외광고 = 옥외광고[[category, "cost"]]
    인터넷 = 인터넷.sort_values("per", ascending=False).reset_index(drop = True)
    인터넷1 = 인터넷
    인터넷 = 인터넷[[category, "cost"]]

    ## 알고리즘 개선 필요.
    ## 정렬한 후 계산이 아닌
    ## 해당 금액일 경우의 최대값 계산 필요
    ## 가격 구성이 100% 아닌 임의의 값 설정(모든 광고사가 모든 광고를 사용)
    tv_ls = [__광고_COST(TV, category, total / 100 * i) for i in range(0,101)]
    모바일_ls = [__광고_COST(모바일, category, total / 100 * i) for i in range(0,101)]
    인터넷_ls = [__광고_COST(인터넷, category, total / 100 * i) for i in range(0,101)]
    옥외광고_ls = [__광고_COST(옥외광고, category, total / 100 * i) for i in range(0,101)]
    
    M = 0
    limit = 101
    ls = []
    cost_ls = []
    for i in range(0, 101):
        limit -= i
        for j in range(0, limit):
            limit -= j
            for q in range(0, limit):
                limit -= q
                obt = p[0] + p[1] * np.log(tv_ls[i]) + p[2] * np.log(모바일_ls[j]) + p[3] * np.log(인터넷_ls[limit - 1]) + p[4] * np.log(옥외광고_ls[q])

                if obt > M:
                    M = obt
                    ls = [i, j, (limit - 1), q]
                    cost_ls = [total / 100 *i, total / 100 *j, total / 100 *(limit - 1), total / 100 *q]
                limit += q
            limit += j
        limit += i
        
    result = pd.DataFrame([ls, cost_ls])
    result.columns = ["TV", "Mobile", "PC", "DOOH"]
    result = result.rename(index={0: "index", 1: "cost"})
    
    tv_ls = __광고리스트(TV1, cost_ls[0])
    모바일_ls = __광고리스트(모바일1, cost_ls[1])
    인터넷_ls = __광고리스트(인터넷1, cost_ls[2])
    옥외광고_ls = __광고리스트(옥외광고1, cost_ls[3])
    
    load_obj = pickle.load(open(PATH+"Mnb.pkl", "rb"))
    percent = load_obj.predict_proba([[ls[0] * 10, ls[1], ls[2], ls[3] * 10]])[0][1]
    print(int(round(percent,2) *100), "%")
    
    return M, result, tv_ls, 모바일_ls, 인터넷_ls, 옥외광고_ls, 


def obtimiz_1(광고명, total, category, tv_p):

    cal_df = data_handling.print_cost(광고명)
    p = data_handling.p_regression(category)
    category = category.split("_")[1]
    TV = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "TV"]
    TV["per"] = TV[category] / TV["cost"]
    모바일 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "모바일"]
    모바일["per"] = 모바일[category] / 모바일["cost"]
    옥외광고 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "옥외광고"]
    옥외광고["per"] = 옥외광고[category] / 옥외광고["cost"]
    인터넷 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "인터넷"]
    인터넷["per"] = 인터넷[category] / 인터넷["cost"]

    ## 특정 연령 정렬 기능 필요
    TV = TV.sort_values("per", ascending=False).reset_index(drop = True)
    TV1 = TV
    TV = TV[[category, "cost"]]
    모바일 = 모바일.sort_values("per", ascending=False).reset_index(drop = True)
    모바일1 = 모바일
    모바일 = 모바일[[category, "cost"]]
    옥외광고 = 옥외광고.sort_values("per", ascending=False).reset_index(drop = True)
    옥외광고1 = 옥외광고
    옥외광고 = 옥외광고[[category, "cost"]]
    인터넷 = 인터넷.sort_values("per", ascending=False).reset_index(drop = True)
    인터넷1 = 인터넷
    인터넷 = 인터넷[[category, "cost"]]

    ## 알고리즘 개선 필요.
    ## 정렬한 후 계산이 아닌
    ## 해당 금액일 경우의 최대값 계산 필요
    ## 가격 구성이 100% 아닌 임의의 값 설정(모든 광고사가 모든 광고를 사용)
    tv_ls = [__광고_COST(TV, category, total / 100 * i) for i in [tv_p]]
    모바일_ls = [__광고_COST(모바일, category, total / 100 * i) for i in range(0,101)]
    인터넷_ls = [__광고_COST(인터넷, category, total / 100 * i) for i in range(0,101)]
    옥외광고_ls = [__광고_COST(옥외광고, category, total / 100 * i) for i in range(0,101)]
    
    M = 0
    limit = 101
    ls = []
    cost_ls = []
    for i in [tv_p]:
        limit -= i
        for j in range(0, limit):
            limit -= j
            for q in range(0, limit):
                limit -= q
                obt = p[0] + p[1] * np.log(tv_ls[0]) + p[2] * np.log(모바일_ls[j]) + p[4] * np.log(옥외광고_ls[q]) + p[3] * np.log(인터넷_ls[limit - 1])
                if obt > M:
                    M = obt
                    ls = [i, j, (limit - 1), q]
                    cost_ls = [total / 100 *i, total / 100 *j, total / 100 *(limit - 1), total / 100 *q]
                limit += q
            limit += j
        limit += i
    
    result = pd.DataFrame([ls, cost_ls])
    result.columns = ["TV", "Mobile", "PC", "DOOH"]
    result = result.rename(index={0: "index", 1: "cost"})
    
    tv_ls = __광고리스트(TV1, cost_ls[0])
    모바일_ls = __광고리스트(모바일1, cost_ls[1])
    인터넷_ls = __광고리스트(인터넷1, cost_ls[2])
    옥외광고_ls = __광고리스트(옥외광고1, cost_ls[3])
    
    load_obj = pickle.load(open(PATH+"Mnb.pkl", "rb"))
    percent = load_obj.predict_proba([[ls[0] * 10, ls[1], ls[2], ls[3] * 10]])[0][1]
    print(int(round(percent,2) *100), "%")
    
    return M, result, tv_ls, 모바일_ls, 인터넷_ls, 옥외광고_ls, 

def obtimiz_2(광고명, total, category, 모바일_p):

    cal_df = data_handling.print_cost(광고명)
    p = data_handling.p_regression(category)
    category = category.split("_")[1]
    TV = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "TV"]
    TV["per"] = TV[category] / TV["cost"]
    모바일 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "모바일"]
    모바일["per"] = 모바일[category] / 모바일["cost"]
    옥외광고 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "옥외광고"]
    옥외광고["per"] = 옥외광고[category] / 옥외광고["cost"]
    인터넷 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "인터넷"]
    인터넷["per"] = 인터넷[category] / 인터넷["cost"]

    ## 특정 연령 정렬 기능 필요
    TV = TV.sort_values("per", ascending=False).reset_index(drop = True)
    TV1 = TV
    TV = TV[[category, "cost"]]
    모바일 = 모바일.sort_values("per", ascending=False).reset_index(drop = True)
    모바일1 = 모바일
    모바일 = 모바일[[category, "cost"]]
    옥외광고 = 옥외광고.sort_values("per", ascending=False).reset_index(drop = True)
    옥외광고1 = 옥외광고
    옥외광고 = 옥외광고[[category, "cost"]]
    인터넷 = 인터넷.sort_values("per", ascending=False).reset_index(drop = True)
    인터넷1 = 인터넷
    인터넷 = 인터넷[[category, "cost"]]

    ## 알고리즘 개선 필요.
    ## 정렬한 후 계산이 아닌
    ## 해당 금액일 경우의 최대값 계산 필요
    ## 가격 구성이 100% 아닌 임의의 값 설정(모든 광고사가 모든 광고를 사용)
    tv_ls = [__광고_COST(TV, category, total / 100 * i) for i in range(0,101)]
    모바일_ls = [__광고_COST(모바일, category, total / 100 * i) for i in [모바일_p]]
    인터넷_ls = [__광고_COST(인터넷, category, total / 100 * i) for i in range(0,101)]
    옥외광고_ls = [__광고_COST(옥외광고, category, total / 100 * i) for i in range(0,101)]
    
    M = 0
    limit = 101
    ls = []
    cost_ls = []
    for j in [모바일_p]:
        limit -= j
        for i in range(0, limit):
            limit -= i
            for q in range(0, limit):
                limit -= q
                obt = p[0] + p[1] * np.log(tv_ls[i]) + p[2] * np.log(모바일_ls[0]) + p[4] * np.log(옥외광고_ls[q]) + p[3] * np.log(인터넷_ls[limit - 1])
                if obt > M:
                    M = obt
                    ls = [i, j, (limit - 1), q]
                    cost_ls = [total / 100 *i, total / 100 *j, total / 100 *(limit - 1), total / 100 *q]
                limit += q
            limit += i
        limit += j
    result = pd.DataFrame([ls, cost_ls])
    result.columns = ["TV", "Mobile", "PC", "DOOH"]
    result = result.rename(index={0: "index", 1: "cost"})
    
    tv_ls = __광고리스트(TV1, cost_ls[0])
    모바일_ls = __광고리스트(모바일1, cost_ls[1])
    인터넷_ls = __광고리스트(인터넷1, cost_ls[2])
    옥외광고_ls = __광고리스트(옥외광고1, cost_ls[3])
    
    load_obj = pickle.load(open(PATH+"Mnb.pkl", "rb"))
    percent = load_obj.predict_proba([[ls[0] * 10, ls[1], ls[2], ls[3] * 10]])[0][1]
    print(int(round(percent,2) *100), "%")
    
    return M, result, tv_ls, 모바일_ls, 인터넷_ls, 옥외광고_ls, 
def obtimiz_3(광고명, total, category, 옥외광고_p):

    cal_df = data_handling.print_cost(광고명)
    p = data_handling.p_regression(category)
    category = category.split("_")[1]
    TV = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "TV"]
    TV["per"] = TV[category] / TV["cost"]
    모바일 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "모바일"]
    모바일["per"] = 모바일[category] / 모바일["cost"]
    옥외광고 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "옥외광고"]
    옥외광고["per"] = 옥외광고[category] / 옥외광고["cost"]
    인터넷 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "인터넷"]
    인터넷["per"] = 인터넷[category] / 인터넷["cost"]

    ## 특정 연령 정렬 기능 필요
    TV = TV.sort_values("per", ascending=False).reset_index(drop = True)
    TV1 = TV
    TV = TV[[category, "cost"]]
    모바일 = 모바일.sort_values("per", ascending=False).reset_index(drop = True)
    모바일1 = 모바일
    모바일 = 모바일[[category, "cost"]]
    옥외광고 = 옥외광고.sort_values("per", ascending=False).reset_index(drop = True)
    옥외광고1 = 옥외광고
    옥외광고 = 옥외광고[[category, "cost"]]
    인터넷 = 인터넷.sort_values("per", ascending=False).reset_index(drop = True)
    인터넷1 = 인터넷
    인터넷 = 인터넷[[category, "cost"]]

    ## 알고리즘 개선 필요.
    ## 정렬한 후 계산이 아닌
    ## 해당 금액일 경우의 최대값 계산 필요
    ## 가격 구성이 100% 아닌 임의의 값 설정(모든 광고사가 모든 광고를 사용)
    tv_ls = [__광고_COST(TV, category, total / 100 * i) for i in range(0,101)]
    모바일_ls = [__광고_COST(모바일, category, total / 100 * i) for i in range(0,101)]
    인터넷_ls = [__광고_COST(인터넷, category, total / 100 * i) for i in range(0,101)]
    옥외광고_ls = [__광고_COST(옥외광고, category, total / 100 * i) for i in [옥외광고_p]]
    
    M = 0
    limit = 101
    ls = []
    cost_ls = []
    for q in [옥외광고_p]:
        limit -= q
        for i in range(0, limit):
            limit -= i
            for j in range(0, limit):
                limit -= j
                obt = p[0] + p[1] * np.log(tv_ls[i]) + p[2] * np.log(모바일_ls[j]) + p[4] * np.log(옥외광고_ls[0]) + p[3] * np.log(인터넷_ls[limit - 1])
                if obt > M:
                    M = obt
                    ls = [i, j, (limit - 1), q]
                    cost_ls = [total / 100 *i, total / 100 *j, total / 100 *(limit - 1), total / 100 *q]
                limit += j
            limit += i
        limit += q
        
    result = pd.DataFrame([ls, cost_ls])
    result.columns = ["TV", "Mobile", "PC", "DOOH"]
    result = result.rename(index={0: "index", 1: "cost"})
    
    tv_ls = __광고리스트(TV1, cost_ls[0])
    모바일_ls = __광고리스트(모바일1, cost_ls[1])
    인터넷_ls = __광고리스트(인터넷1, cost_ls[2])
    옥외광고_ls = __광고리스트(옥외광고1, cost_ls[3])
    
    load_obj = pickle.load(open(PATH+"Mnb.pkl", "rb"))
    percent = load_obj.predict_proba([[ls[0] * 10, ls[1], ls[2], ls[3] * 10]])[0][1]
    print(int(round(percent,2) *100), "%")
    
    return M, result, tv_ls, 모바일_ls, 인터넷_ls, 옥외광고_ls, 

def obtimiz_4(광고명, total, category, 인터넷_p):

    cal_df = data_handling.print_cost(광고명)
    p = data_handling.p_regression(category)
    category = category.split("_")[1]
    TV = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "TV"]
    TV["per"] = TV[category] / TV["cost"]
    모바일 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "모바일"]
    모바일["per"] = 모바일[category] / 모바일["cost"]
    옥외광고 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "옥외광고"]
    옥외광고["per"] = 옥외광고[category] / 옥외광고["cost"]
    인터넷 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "인터넷"]
    인터넷["per"] = 인터넷[category] / 인터넷["cost"]

    ## 특정 연령 정렬 기능 필요
    TV = TV.sort_values("per", ascending=False).reset_index(drop = True)
    TV1 = TV
    TV = TV[[category, "cost"]]
    모바일 = 모바일.sort_values("per", ascending=False).reset_index(drop = True)
    모바일1 = 모바일
    모바일 = 모바일[[category, "cost"]]
    옥외광고 = 옥외광고.sort_values("per", ascending=False).reset_index(drop = True)
    옥외광고1 = 옥외광고
    옥외광고 = 옥외광고[[category, "cost"]]
    인터넷 = 인터넷.sort_values("per", ascending=False).reset_index(drop = True)
    인터넷1 = 인터넷
    인터넷 = 인터넷[[category, "cost"]]

    ## 알고리즘 개선 필요.
    ## 정렬한 후 계산이 아닌
    ## 해당 금액일 경우의 최대값 계산 필요
    ## 가격 구성이 100% 아닌 임의의 값 설정(모든 광고사가 모든 광고를 사용)
    tv_ls = [__광고_COST(TV, category, total / 100 * i) for i in range(0,101)]
    모바일_ls = [__광고_COST(모바일, category, total / 100 * i) for i in range(0,101)]
    인터넷_ls = [__광고_COST(인터넷, category, total / 100 * i) for i in [인터넷_p]]
    옥외광고_ls = [__광고_COST(옥외광고, category, total / 100 * i) for i in range(0,101)]
    
    M = 0
    limit = 101
    ls = []
    cost_ls = []
    for i in [인터넷_p]:
        limit -= i
        for j in range(0, limit):
            limit -= j
            for q in range(0, limit):
                limit -= q
                obt = p[0] + p[1] * np.log(tv_ls[limit - 1]) + p[2] * np.log(모바일_ls[j]) + p[4] * np.log(옥외광고_ls[q]) + p[3] * np.log(인터넷_ls[0])
                if obt > M:
                    M = obt
                    ls = [(limit - 1), j, i, q]
                    cost_ls = [total / 100 *(limit - 1), total / 100 *j, total / 100 *i, total / 100 *q]
                limit += q
            limit += j
        limit += i
    
    result = pd.DataFrame([ls, cost_ls])
    result.columns = ["TV", "Mobile", "PC", "DOOH"]
    result = result.rename(index={0: "index", 1: "cost"})
    
    tv_ls = __광고리스트(TV1, cost_ls[0])
    모바일_ls = __광고리스트(모바일1, cost_ls[1])
    인터넷_ls = __광고리스트(인터넷1, cost_ls[2])
    옥외광고_ls = __광고리스트(옥외광고1, cost_ls[3])
    
    load_obj = pickle.load(open(PATH+"Mnb.pkl", "rb"))
    percent = load_obj.predict_proba([[ls[0] * 10, ls[1], ls[2], ls[3] * 10]])[0][1]
    print(int(round(percent,2) *100), "%")
    
    return M, result, tv_ls, 모바일_ls, 인터넷_ls, 옥외광고_ls, 

def Mnb_obtimiz(광고명, total = 100000, category = "전체_전체", tv_p = None, 모바일_p = None, 옥외광고_p = None, 인터넷_p = None):
    if tv_p != None:
        return obtimiz_1(광고명, total, category, tv_p)
    elif 모바일_p != None:
        return obtimiz_2(광고명, total, category, 모바일_p)
    elif 옥외광고_p != None:
        return obtimiz_3(광고명, total, category, 옥외광고_p)
    elif 인터넷_p != None:
        return obtimiz_4(광고명, total, category, 인터넷_p)
    
    cal_df = data_handling.print_cost(광고명)
    p = data_handling.p_regression(category)
    category = category.split("_")[1]
    TV = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "TV"]
    TV["per"] = TV[category] / TV["cost"]
    모바일 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "모바일"]
    모바일["per"] = 모바일[category] / 모바일["cost"]
    옥외광고 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "옥외광고"]
    옥외광고["per"] = 옥외광고[category] / 옥외광고["cost"]
    인터넷 = cal_df[cal_df["index"].apply(lambda x : x.split("_")[0]) == "인터넷"]
    인터넷["per"] = 인터넷[category] / 인터넷["cost"]

    ## 특정 연령 정렬 기능 필요
    TV = TV.sort_values("per", ascending=False).reset_index(drop = True)
    TV1 = TV
    TV = TV[[category, "cost"]]
    모바일 = 모바일.sort_values("per", ascending=False).reset_index(drop = True)
    모바일1 = 모바일
    모바일 = 모바일[[category, "cost"]]
    옥외광고 = 옥외광고.sort_values("per", ascending=False).reset_index(drop = True)
    옥외광고1 = 옥외광고
    옥외광고 = 옥외광고[[category, "cost"]]
    인터넷 = 인터넷.sort_values("per", ascending=False).reset_index(drop = True)
    인터넷1 = 인터넷
    인터넷 = 인터넷[[category, "cost"]]
    
    ## 알고리즘 개선 필요.
    ## 정렬한 후 계산이 아닌
    ## 해당 금액일 경우의 최대값 계산 필요
    ## 가격 구성이 100% 아닌 임의의 값 설정(모든 광고사가 모든 광고를 사용)
    tv_ls = [__광고_COST(TV, category, total / 100 * i) for i in range(0,101)]
    모바일_ls = [__광고_COST(모바일, category, total / 100 * i) for i in range(0,101)]
    인터넷_ls = [__광고_COST(인터넷, category, total / 100 * i) for i in range(0,101)]
    옥외광고_ls = [__광고_COST(옥외광고, category, total / 100 * i) for i in range(0,101)]
    
    M = 0
    limit = 101
    ls = []
    cost_ls = []
    for i in range(0, 101):
        limit -= i
        for j in range(0, limit):
            limit -= j
            for q in range(0, limit):
                limit -= q
                obt = p[0] + p[1] * np.log(tv_ls[i]) + p[2] * np.log(모바일_ls[j]) + p[3] * np.log(인터넷_ls[limit - 1]) + p[4] * np.log(옥외광고_ls[q])

                if obt > M:
                    M = obt
                    ls = [i, j, (limit - 1), q]
                    cost_ls = [total / 100 *i, total / 100 *j, total / 100 *(limit - 1), total / 100 *q]
                limit += q
            limit += j
        limit += i
        
    result = pd.DataFrame([ls, cost_ls])
    result.columns = ["TV", "Mobile", "PC", "DOOH"]
    result = result.rename(index={0: "index", 1: "cost"})
    
    tv_ls = __광고리스트(TV1, cost_ls[0])
    모바일_ls = __광고리스트(모바일1, cost_ls[1])
    인터넷_ls = __광고리스트(인터넷1, cost_ls[2])
    옥외광고_ls = __광고리스트(옥외광고1, cost_ls[3])
    
    load_obj = pickle.load(open(PATH+"Mnb.pkl", "rb"))
    percent = load_obj.predict_proba([[ls[0] * 10, ls[1], ls[2], ls[3] * 10]])[0][1]
    print(int(round(percent,2) *100), "%")
    
    return M, result, tv_ls, 모바일_ls, 인터넷_ls, 옥외광고_ls, 
