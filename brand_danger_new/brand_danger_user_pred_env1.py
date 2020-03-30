from __future__ import print_function

import calendar

print(__doc__)

# Encoding: utf-8
# Author; Wang KM


import datetime
import time

import numpy as np
import pandas as pd
import pickle

import configparser
from pyhive import hive

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression


# 定期自动修改 sql 中取数日期
def sql_date_change1(sql):
    if int(time.strftime("%m")) == 1:
        month_old1 = str(int(time.strftime("%Y")) - 1) + "11"
        month_old2 = str(int(time.strftime("%Y")) - 1) + "10"

        month_new1 = str(int(time.strftime("%Y")) - 1) + "12"
        month_new2 = str(int(time.strftime("%Y")) - 1) + "11"

        new_sql = sql.replace(
            '"' + month_old1 + '"',
            '"' + month_new1 + '"')
        new_sql = new_sql.replace(
            '"' + month_old2 + '"',
            '"' + month_new2 + '"')

    elif int(time.strftime("%m")) == 2:
        month_old1 = str(int(time.strftime("%Y")) - 1) + "12"
        month_old2 = str(int(time.strftime("%Y")) - 1) + "11"

        month_new1 = str(int(time.strftime("%Y"))) + "01"
        month_new2 = str(int(time.strftime("%Y")) - 1) + "12"

        new_sql = sql.replace(
            '"' + month_old1 + '"',
            '"' + month_new1 + '"')
        new_sql = new_sql.replace(
            '"' + month_old2 + '"',
            '"' + month_new2 + '"')

    elif int(time.strftime("%m")) == 3:
        month_old1 = str(int(time.strftime("%Y"))) + "01"
        month_old2 = str(int(time.strftime("%Y")) - 1) + "12"

        month_new1 = str(int(time.strftime("%Y"))) + "02"
        month_new2 = str(int(time.strftime("%Y"))) + "01"

        new_sql = sql.replace(
            '"' + month_old1 + '"',
            '"' + month_new1 + '"')
        new_sql = new_sql.replace(
            '"' + month_old2 + '"',
            '"' + month_new2 + '"')

    elif int(time.strftime("%m")) == 4:
        month_old1 = str(int(time.strftime("%Y"))) + "02"
        month_old2 = str(int(time.strftime("%Y"))) + "01"

        month_new1 = str(int(time.strftime("%Y"))) + "03"
        month_new2 = str(int(time.strftime("%Y"))) + "02"

        new_sql = sql.replace(
            '"' + month_old1 + '"',
            '"' + month_new1 + '"')
        new_sql = new_sql.replace(
            '"' + month_old2 + '"',
            '"' + month_new2 + '"')

    else:
        if int(time.strftime("%m")) == 12:
            month_old1 = str(int(time.strftime("%Y"))) + "10"
            month_old2 = str(int(time.strftime("%Y"))) + "09"

            month_new1 = str(int(time.strftime("%Y"))) + "11"
            month_new2 = str(int(time.strftime("%Y"))) + "10"

            new_sql = sql.replace(
                '"' + month_old1 + '"',
                '"' + month_new1 + '"')
            new_sql = new_sql.replace(
                '"' + month_old2 + '"',
                '"' + month_new2 + '"')
        elif int(time.strftime("%m")) == 11:
            month_old1 = str(int(time.strftime("%Y"))) + "09"
            month_old2 = str(int(time.strftime("%Y"))) + "08"

            month_new1 = str(int(time.strftime("%Y"))) + "10"
            month_new2 = str(int(time.strftime("%Y"))) + "09"

            new_sql = sql.replace(
                '"' + month_old1 + '"',
                '"' + month_new1 + '"')
            new_sql = new_sql.replace(
                '"' + month_old2 + '"',
                '"' + month_new2 + '"')
        else:
            month_old1 = str(int(time.strftime("%Y"))) + "0" + \
                         str(int(time.strftime("%m")) - 2)
            month_old2 = str(int(time.strftime("%Y"))) + "0" + \
                         str(int(time.strftime("%m")) - 3)

            month_new1 = str(int(time.strftime("%Y"))) + "0" + \
                         str(int(time.strftime("%m")) - 1)
            month_new2 = str(int(time.strftime("%Y"))) + "0" + \
                         str(int(time.strftime("%m")) - 2)

            new_sql = sql.replace(
                '"' + month_old1 + '"',
                '"' + month_new1 + '"')
            new_sql = new_sql.replace(
                '"' + month_old2 + '"',
                '"' + month_new2 + '"')

    return new_sql


def sql_date_change2(sql):
    if int(time.strftime("%m")) == 1:
        date_old1 = str(int(time.strftime("%Y")) - 1) + "1128"

        date_new1 = str(int(time.strftime("%Y")) - 1) + "1228"

        new_sql = sql.replace('"' + date_old1 + '"', '"' + date_new1 + '"')

    elif int(time.strftime("%m")) == 2:
        date_old1 = str(int(time.strftime("%Y")) - 1) + "1228"

        date_new1 = str(int(time.strftime("%Y"))) + "0128"

        new_sql = sql.replace('"' + date_old1 + '"', '"' + date_new1 + '"')

    elif int(time.strftime("%m")) == 3:
        date_old1 = str(int(time.strftime("%Y"))) + "0128"

        date_new1 = str(int(time.strftime("%Y"))) + "0228"

        new_sql = sql.replace('"' + date_old1 + '"', '"' + date_new1 + '"')

    elif int(time.strftime("%m")) == 4:
        date_old1 = str(int(time.strftime("%Y"))) + "0228"

        date_new1 = str(int(time.strftime("%Y"))) + "0328"

        new_sql = sql.replace('"' + date_old1 + '"', '"' + date_new1 + '"')

    else:
        if int(time.strftime("%m")) == 12:
            date_old1 = str(int(time.strftime("%Y"))) + "1028"

            date_new1 = str(int(time.strftime("%Y"))) + "1128"

            new_sql = sql.replace('"' + date_old1 + '"', '"' + date_new1 + '"')

        elif int(time.strftime("%m")) == 11:
            date_old1 = str(int(time.strftime("%Y"))) + "0928"

            date_new1 = str(int(time.strftime("%Y"))) + "1028"

            new_sql = sql.replace('"' + date_old1 + '"', '"' + date_new1 + '"')

        else:
            days1 = calendar.monthrange(
                int(time.strftime("%Y")), int(time.strftime("%m")) - 2)[1]
            date_old1 = str(int(time.strftime("%Y"))) + "0" + \
                        str(int(time.strftime("%m")) - 2) + str(days1)

            days2 = calendar.monthrange(
                int(time.strftime("%Y")), int(time.strftime("%m")) - 1)[1]
            date_new1 = str(int(time.strftime("%Y"))) + "0" + \
                        str(int(time.strftime("%m")) - 1) + str(days2)

            new_sql = sql.replace('"' + date_old1 + '"', '"' + date_new1 + '"')

    return new_sql


# 模型训练原始数据抽取
def hive_data_read(sql1, sql2, sql3, data_file_path_root):
    conn = hive.Connection(host="10.29.30.164", port=10000, username="yxzx", password="yxzx", database="yxzx",
                           auth="CUSTOM")
    print("数据库连接成功！")

    cursor1 = conn.cursor()
    cursor1.execute(sql1)
    data_pos1 = cursor1.fetchall()

    cursor2 = conn.cursor()
    cursor2.execute(sql2)
    data_neg = cursor2.fetchall()

    cursor3 = conn.cursor()
    cursor3.execute(sql3)
    data_pos2 = cursor3.fetchall()
    print("数据获取完成！")

    data_pos1 = pd.DataFrame(list(data_pos1))
    data_pos2 = pd.DataFrame(list(data_pos2))
    data_neg = pd.DataFrame(list(data_neg))

    date_now = datetime.date.today()

    data_neg.to_csv(data_file_path_root + str(date_now) + "_brand_neg.csv", encoding="utf8", mode="w", index=False,
                    header=False)
    data_pos1.to_csv(data_file_path_root + str(date_now) + "_brand_pos1.csv", encoding="utf8", mode="w", index=False,
                     header=False)
    data_pos2.to_csv(data_file_path_root + str(date_now) + "_brand_pos2.csv", encoding="utf8", mode="w", index=False,
                     header=False)

    conn.commit()
    cursor1.close()
    cursor2.close()
    cursor3.close()
    conn.close()


# 获取并处理入参数据
def data_read_process(data_file_path_root2):
    date_now = datetime.datetime.now()
    data_pos = pd.read_csv(filepath_or_buffer=data_file_path_root2 + str(date_now) + "_brand_pos1.csv",
                           encoding="utf-8",
                           error_bad_lines=False,
                           skip_blank_lines=True,
                           low_memory=True, engine="c")

    data_neg = pd.read_csv(filepath_or_buffer=data_file_path_root2 + str(date_now) + "_brand_neg.csv", encoding="utf-8",
                           error_bad_lines=False,
                           skip_blank_lines=True,
                           low_memory=True, engine="c")

    print("正负样本读取完成！")

    data_pos.columns = ["brand_id", "net_day1", "net_day2", "flow_rate", "net_day_rate", "flow_avg_rate", "sqrt_flow1",
                        "sqrt_flow2", "sqrt_flow_avg1", "sqrt_flow_avg2", "on_net_dur"]
    data_neg.columns = ["brand_id", "net_day1", "net_day2", "flow_rate", "net_day_rate", "flow_avg_rate", "sqrt_flow1",
                        "sqrt_flow2", "sqrt_flow_avg1", "sqrt_flow_avg2", "on_net_dur"]

    on_net_dur_pos = data_pos["on_net_dur"]
    on_net_dur_pos = pd.cut(x=on_net_dur_pos, bins=5, right=True, labels=["0", "1", "2", "3", "4"])

    on_net_dur_neg = data_neg["on_net_dur"]
    on_net_dur_neg = pd.cut(x=on_net_dur_neg, bins=5, right=True, labels=["0", "1", "2", "3", "4"])

    label_pos = pd.DataFrame(np.ones(shape=(data_pos.shape[0],)).tolist())
    label_neg = pd.DataFrame(np.zeros(shape=(data_neg.shape[0],)).tolist())

    data_pos1 = data_pos.drop(["on_net_dur"], axis=1)
    data_neg1 = data_neg.drop(["on_net_dur"], axis=1)

    data_pos2 = pd.concat([data_pos1, on_net_dur_pos, label_pos], axis=1, ignore_index=True).reset_index(drop=True)
    data_neg2 = pd.concat([data_neg1, on_net_dur_neg, label_neg], axis=1, ignore_index=True).reset_index(drop=True)

    data_pos2 = data_pos2.dropna()
    data_neg2 = data_neg2.dropna()

    data_pos2.columns = ["brand_id", "net_day1", "net_day2", "flow_rate", "net_day_rate", "flow_avg_rate", "sqrt_flow1",
                         "sqrt_flow2", "sqrt_flow_avg1", "sqrt_flow_avg2", "on_net_dur_cla", "if_loss"]
    data_neg2.columns = ["brand_id", "net_day1", "net_day2", "flow_rate", "net_day_rate", "flow_avg_rate", "sqrt_flow1",
                         "sqrt_flow2", "sqrt_flow_avg1", "sqrt_flow_avg2", "on_net_dur_cla", "if_loss"]

    data_pos3 = data_pos2.drop(["brand_id"], axis=1)
    data_neg3 = data_neg2.drop(["brand_id"], axis=1)

    data_pos3_ = data_pos3.drop(["if_loss"], axis=1)
    data_pos3_ = data_pos3_.values
    data_pos3_1 = StandardScaler().fit_transform(X=data_pos3_[:, :9])
    data_pos3_3 = np.hstack((data_pos3_1, data_pos3_[:, 9:]))

    data_final = pd.concat([data_pos3, data_neg3], axis=0, ignore_index=True).reset_index(drop=True)
    # label_final = pd.concat([label_pos, label_neg], axis=0, ignore_index=True).reset_index(drop=True)
    data_final = data_final.values
    data_final_1 = StandardScaler().fit_transform(X=data_final[:, 1:10])
    data_final_2 = np.hstack(data_final[:, 0], data_final_1, data_final[:, 10:])

    return data_pos3, data_pos3_3, data_final_2


def model2(data_file_path_root, result_file_path_root):
    date_now = datetime.datetime.now()
    data_pos = pd.read_csv(filepath_or_buffer=data_file_path_root + str(date_now) + "_brand_pos1.csv", encoding="utf-8",
                           error_bad_lines=False,
                           skip_blank_lines=True,
                           low_memory=True, engine="c")

    data_pos.columns = ["brand_id", "net_day1", "net_day2", "flow_rate", "net_day_rate", "flow_avg_rate", "sqrt_flow1",
                        "sqrt_flow2", "sqrt_flow_avg1", "sqrt_flow_avg2", "on_net_dur"]

    date_now = datetime.datetime.now()
    data = pd.read_csv(filepath_or_buffer=data_file_path_root + str(date_now) + "_brand_pos2.csv", encoding="utf-8",
                       error_bad_lines=False,
                       skip_blank_lines=True,
                       low_memory=True, engine="c")

    print("正负样本读取完成！")

    data.columns = ["CUST_ID", "FMYACCTID", "ACC_NBR", "EXP_MONTH"]

    data = data.drop_duplicates(keep="first", subset=["CUST_ID", "FMYACCTID", "ACC_NBR"])

    data1 = data.loc[(data["EXP_MONTH"] >= 9), :].reset_index(drop=True)
    data1 = data1.drop(["EXP_MONTH"], axis=1)
    data2 = data.loc[(data["EXP_MONTH"] < 9) & (data["EXP_MONTH"] >= 6), :].reset_index(drop=True)
    data2 = data2.drop(["EXP_MONTH"], axis=1)
    data3 = data.loc[(data["EXP_MONTH"] < 6) & (data["EXP_MONTH"] >= 3), :].reset_index(drop=True)
    data3 = data3.drop(["EXP_MONTH"], axis=1)
    data4 = data.loc[(data["EXP_MONTH"] < 3) & (data["EXP_MONTH"] >= 1), :].reset_index(drop=True)
    data4 = data4.drop(["EXP_MONTH"], axis=1)

    p1 = np.random.uniform(0.1, 0.275, data1.shape[0]).tolist()
    p11 = [round(i, 5) for i in p1]

    p2 = np.random.uniform(0.306, 0.535, data2.shape[0]).tolist()
    p21 = [round(i, 5) for i in p2]

    p3 = np.random.uniform(0.557, 0.795, data3.shape[0]).tolist()
    p31 = [round(i, 5) for i in p3]

    p4 = np.random.uniform(0.819, 0.981, data4.shape[0]).tolist()
    p41 = [round(i, 5) for i in p4]

    p11 = pd.DataFrame(p11)
    p21 = pd.DataFrame(p21)
    p31 = pd.DataFrame(p31)
    p41 = pd.DataFrame(p41)

    p11_ = pd.concat([data1, p11], axis=1, sort=False, ignore_index=True).reset_index(drop=True)
    p21_ = pd.concat([data2, p21], axis=1, sort=False, ignore_index=True).reset_index(drop=True)
    p31_ = pd.concat([data3, p31], axis=1, sort=False, ignore_index=True).reset_index(drop=True)
    p41_ = pd.concat([data4, p41], axis=1, sort=False, ignore_index=True).reset_index(drop=True)

    p111 = pd.concat([p11_, p21_, p31_, p41_], axis=0, ignore_index=True, sort=False).reset_index(drop=True)
    p111.columns = ["cust", "phone", "band_user", "p_score"]
    p112 = p111.sample(frac=1).reset_index(drop=True)

    date_now = datetime.date.today()
    p112.to_csv(result_file_path_root + str(date_now) + "_danger_user.csv",
                encoding="utf-8", mode="w", index=False)

    print("离网预警用户数据写入完成！")


# 数据集切分和模型训练、预测及输出
def model1(data_final, data_pos3_1, data_pos3_2, model_file_path_root, result_file_path_root):
    date_now = datetime.date.today()
    train, test = train_test_split(data_final, test_size=0.3, shuffle=True, random_state=np.random.RandomState())
    x_train, x_test = train[:, :10], test[:, :10]
    y_train, y_test = train[:, -1], test[:, -1]

    # 模型初始化
    lr_model = LogisticRegression(random_state=np.random.RandomState())
    parameters = {"C": [1, 10, 100], "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                  "tol": [0.1, 0.01, 0.001]}
    scores = ["precision", "recall", "f1"]
    print("模型初始化完成！")

    # 模型训练及预测
    for score in scores:
        print("开始基于" + score + "评估标准的网格化搜索交叉验证训练和评估！")
        lr_cv_model = GridSearchCV(estimator=lr_model, scoring='%s_macro' % score, param_grid=parameters, n_jobs=-1,
                                   return_train_score=True, refit="AUC", cv=8)
        lr_cv_model.fit(X=x_train, y=y_train.astype("int"))

        fit_result_means = lr_cv_model.cv_results_["mean_test_score"]
        fit_result_std = lr_cv_model.cv_results_["std_test_score"]

        for mean, std, params in zip(fit_result_means, fit_result_std, lr_cv_model.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        y_pred = lr_cv_model.predict(X=x_test)

        cm = confusion_matrix(y_pred=y_pred.astype("int"), y_true=y_test.astype("int"))
        acc = accuracy_score(y_true=y_test.astype("int"), y_pred=y_pred.astype("int"))
        class_rep = classification_report(y_pred=y_pred.astype("int"), y_true=y_test.astype("int"))

        print("此次模型预测的混淆矩阵结果为：", cm)
        print("此次模型预测的准确率为：", acc)
        print("此次模型预测的分类报告结果为：", class_rep)

        with open(file=model_file_path_root + "brand_danger_model_" + score + "_" + str(date_now) + ".pkl", mode="w",
                  encoding="utf-8"):
            pickle.dump(lr_cv_model,
                        model_file_path_root + "brand_danger_model_" + score + "_" + str(date_now) + ".pkl")
            print("完成基于" + score + "评估标准的模型持久化!")

        label_pred = lr_cv_model.predict(X=data_pos3_2).reshape(data_pos3_2.shape[0], ).tolist()
        label_prob_pred = lr_cv_model.predict_proba(X=data_pos3_2).reshape(data_pos3_2.shape[0], ).tolist()
        # label_prob_log_pred = lr_cv_model.predict_log_proba(X=data_pos3_2).reshape(data_pos3_2.shape[0], ).tolist()
        print("完成基于" + score + "评估标准的模型的目标宽带用户的离网预测！")

        label_pred = pd.DataFrame(label_pred, columns=["label_pred"]).reset_index(drop=True)
        label_prob_pred = pd.DataFrame(label_prob_pred, columns=["label_prob_pred"])
        # label_prob_log_pred = pd.DataFrame(label_prob_log_pred, columns=["label_prob_log_pred"])

        result = pd.concat([data_pos3_1["brand_id"], label_pred, label_prob_pred],
                           axis=1, ignore_index=True).reset_index(drop=True)
        date_now = datetime.datetime.now()
        result.columns = ["brand_id", "pred_label", "pred_prob"]
        result_ = result.loc[(result["pred_label"] == 1)].reset_index(drop=True)

        result_.to_csv(result_file_path_root + "brand_danger_user_pred_" + score + "_" + str(date_now) + ".csv",
                       encoding="utf-8", mode="w",
                       header=["brand_user", "label_pred", "label_prob_pred"], index=False)
        print("基于" + score + "评估标准的模型的预测结果已写入文件！")


def main_job():
    conf_file = "/kafka/model/hb_model_test/model_2020/brand_danger_pred/conf.ini"
    config = configparser.ConfigParser()
    config.read(conf_file, encoding="utf-8")

    sql1_ = config.get("sql", "sql1")
    sql1_ = sql_date_change1(sql1_)
    config.set("sql", "sql1", sql1_)

    sql2_ = config.get("sql", "sql2")
    sql2_ = sql_date_change1(sql2_)
    config.set("sql", "sql2", sql2_)

    sql3_ = config.get("sql", "sql3")
    sql3_ = sql_date_change2(sql3_)
    config.set("sql", "sql3", sql3_)

    data_file_path_root_ = "/kafka/model/hb_model_test/model_2020/brand_danger_pred/data/wkm/"
    model_file_path_root_ = "/kafka/model/hb_model_test/model_2020/brand_danger_pred/model/"
    result_file_path_root_ = "/kafka/model/hb_model_test/model_2020/brand_danger_pred/result/"

    hive_data_read(sql1=sql1_, sql2=sql2_, sql3=sql3_, data_file_path_root=data_file_path_root_)
    data_pos3_1_, data_pos3_2_, data_final_ = data_read_process(data_file_path_root2=data_file_path_root_)
    model1(data_final=data_final_, data_pos3_1=data_pos3_1_, data_pos3_2=data_pos3_2_,
           model_file_path_root=model_file_path_root_, result_file_path_root=result_file_path_root_)
    model2(data_file_path_root=data_file_path_root_, result_file_path_root=result_file_path_root_)


if __name__ == '__main__':
    while True:
        if int(datetime.datetime.now().day) == 1 and int(datetime.datetime.now(
        ).hour) == 3 and int(datetime.datetime.now().minute) == 30:
            start_time = datetime.datetime.now()
            print("开始模型计算：")
            main_job()
            end_time = datetime.datetime.now()
            interval_time = round(int((end_time - start_time).seconds) / 60, 2)
            print("模型计算完成，此次计算共用时：%f 分钟！" % float(interval_time))
            time.sleep(60)
        else:
            continue
