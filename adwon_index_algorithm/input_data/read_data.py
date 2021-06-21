import psycopg2
import pandas as pd
import sys
from urllib.parse import urlparse
from adwon_index_algorithm.input_data import ___id_pass


PATH = sys.path[0] + "/adwon_index_algorithm/input_data"

ls = ___id_pass.id_pass

u_id = ls[0]
u_ps = ls[1]
url = ls[2]
db = ls[3]

def read():
    result = urlparse("postgresql://{}:{}!@{}/{}".format(u_id, u_ps, url, db))
    username = result.username
    password = result.password
    database = result.path[1:]
    hostname = result.hostname
    port = result.port
    connection = psycopg2.connect(
        database = database,
        user = username,
        password = password,
        host = hostname,
        port = port
    )
    cur = connection.cursor()
    
    return cur

def ad(cur, a):
    print(a)
    QUERY = """
    SELECT *
    FROM {}
    """.format(a)
    cur.execute(QUERY)
    result = cur.fetchall()
    df = pd.DataFrame(result)
    return df

def new_read():
    cur = read()
    
    panels = ad("panels")
    panels["성별"] = panels[3].apply(lambda x : "남" if x == "m" else "여")
    panels["연령"] = (panels[10] - 2022) * -1
    panels = panels[[0, "성별", "연령"]]

    raw_tv = ad(cur, "raw_tv")
    raw_mo = ad(cur, "raw_mo")
    raw_pc = ad(cur, "raw_pc")
    raw_out = ad(cur, "spot_foot_traffics")

    t_tv = raw_tv.merge(panels, "left", left_on=1, right_on=0)
    t_tv["광고"] = "TV"
    t_tv["timestemp"] = t_tv[5]
    t_tv["채널"] = t_tv[2]
    t_tv["광고명"] = "None"
    t_tv["cost"] = t_tv[4]
    t_tv = t_tv[["광고명", "광고", "채널", "성별", "연령", "timestemp", "cost"]]

    t_mo = raw_mo.merge(panels, "left", left_on=1, right_on=0)
    t_mo["광고"] = "모바일"
    t_mo["timestemp"] = t_mo[4]
    t_mo["채널"] = t_mo[2]
    t_mo["광고명"] = "None"
    t_mo["cost"] = 500000000
    t_mo = t_mo[["광고명", "광고", "채널", "성별", "연령", "timestemp", "cost"]]

    t_pc = raw_pc.merge(panels, "left", left_on=1, right_on=0)
    t_pc["광고"] = "인터넷"
    t_pc["timestemp"] = t_pc[4]
    t_pc["채널"] = t_pc[2]
    t_pc["광고명"] = "None"
    t_pc["cost"] = 500000000
    t_pc = t_pc[["광고명", "광고", "채널", "성별", "연령", "timestemp", "cost"]]

    t_out = raw_out.merge(panels, "left", left_on=1, right_on=0)
    t_out["광고"] = "옥외광고"
    t_out["timestemp"] = t_out[4]
    t_out["채널"] = t_out[2]
    t_out["광고명"] = "None"
    t_out["cost"] = 500000000
    t_out = t_out[["광고명", "광고", "채널", "성별", "연령", "timestemp", "cost"]]

    t_tv = t_tv.append(t_mo)
    t_tv = t_tv.append(t_pc)
    t_tv = t_tv.append(t_out)
    t_tv = t_tv.reset_index(drop = True)

    t_tv.to_csv(PATH + "new_data.csv", index=False)
    
def user_read():
    
    cur = read()
    
    QUERY = """
    SELECT 
       p.id, tv.count AS tv, 
       pc.count AS pc, 
       dooh.count AS dooh, 
       mobile.count AS mobile,
       case when install.count > 0 then true else false end AS downloaded
    FROM panels p
    inner JOIN (SELECT panel_id, COUNT(*) AS count FROM raw_tv where tv_advertisement_id = 4 GROUP BY panel_id) tv ON p.id = tv.panel_id
    inner JOIN (SELECT panel_id, COUNT(*) AS count FROM raw_pc where media_company_id = 1 GROUP BY panel_id) pc ON p.id = pc.panel_id
    inner JOIN (SELECT panel_id, COUNT(*) AS count FROM spot_foot_traffics where spot_id = 27 GROUP BY panel_id) dooh ON p.id = dooh.panel_id
    inner JOIN (SELECT panel_id, COUNT(*) AS count FROM raw_mo where media_company_id = 45 GROUP BY panel_id) mobile ON p.id = mobile.panel_id
    inner JOIN (SELECT device_id, COUNT(*) AS count FROM raw_singular_install GROUP BY device_id) install ON p.adid = install.device_id
    ORDER BY p.id
    """
    cur.execute(QUERY)
    result = cur.fetchall()
    df = pd.DataFrame(result)
    
    df = df.iloc[:,1:] 
    d_df = df.iloc[:,:-1] / 2
    d_df[5] = False
    df = df.append(d_df)
    df.to_csv(PATH + "user_data.csv", index=False)

    # 데이터 전처리 : 데이터 셋 분리
    X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:,:-1], df.iloc[:,-1], test_size=0.4, random_state=1)
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pickle.dump(model, open("Mnb.pkl", "wb"))
