# Encoding: utf-8
# Author; KM Wang


import configparser
import datetime
import pickle

import numpy as np
import pandas as pd
from pandasql import sqldf
from pyhive import hive
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

"""原始数据读取"""


def hive_data_read(sql1, sql2, sql3, data_file_path_root1):
    conn = hive.Connection(host="10.29.30.164", port=10000, username="ocdc", password="ocdc", database="zjcoc",
                           auth="CUSTOM")
    print("数据库连接成功！")

    cursor1 = conn.cursor()
    cursor1.execute(sql1)
    data_pos = cursor1.fetchall()

    cursor2 = conn.cursor()
    cursor2.execute(sql2)
    data_neg = cursor2.fetchall()

    cursor3 = conn.cursor()
    cursor3.execute(sql3)
    data_5g_user = cursor3.fetchall()
    print("数据获取完成！")

    data_pos = pd.DataFrame(list(data_pos))
    data_neg = pd.DataFrame(list(data_neg))
    data_5g = pd.DataFrame(list(data_5g_user))

    print("数据信息汇总", data_pos.info())

    date_now = datetime.datetime.now()

    data_neg.to_csv(data_file_path_root1 + str(date_now) + "_5g_phone_neg.csv", encoding="utf8", mode="w", index=False)
    data_pos.to_csv(data_file_path_root1 + str(date_now) + "_5g_phone_pos.csv", encoding="utf8", mode="w", index=False)
    data_5g.to_csv(data_file_path_root1 + str(date_now) + "_5g_user.csv", encoding="utf8", mode="w", index=False)
    print("原始数据以写入本地文件")

    conn.commit()
    cursor1.close()
    cursor2.close()
    cursor3.close()
    conn.close()


""" 潜在用户识别 """


def user_recog(data_file_path_root2, result_file_path_root, model_file_path_root):
    # 正负样本读取
    date_now = datetime.date.today()
    data_pos = pd.read_csv(filepath_or_buffer=data_file_path_root2 + "_5g_phone_pos.csv", encoding="utf-8",
                           error_bad_lines=False,
                           skip_blank_lines=True,
                           low_memory=True, engine="c")

    data_neg = pd.read_csv(filepath_or_buffer=data_file_path_root2 + "_5g_phone_neg.csv", encoding="utf-8",
                           error_bad_lines=False,
                           skip_blank_lines=True,
                           low_memory=True, engine="c")

    data_5g_user = pd.read_csv(filepath_or_buffer=data_file_path_root2 + "_5g_user.csv", encoding="utf-8",
                               error_bad_lines=False,
                               skip_blank_lines=True,
                               low_memory=True, engine="c")

    print("正负样本读取完成！")

    data_pos.columns = ["user_id", "phone_no", "age", "user_state", "sex", "onnet_day_num", "cust_type", "vip_type",
                        "if_global", "avg_bill_fee", "gprs_down_flow", "sell_price", "use_date", "term_brand_id",
                        "term_brand", "term_model", "suppt_double_card", "suppt_lte", "screen_size"]
    data_neg.columns = ["user_id", "phone_no", "age", "user_state", "sex", "onnet_day_num", "cust_type", "vip_type",
                        "if_global", "avg_bill_fee", "gprs_down_flow", "sell_price", "use_date", "term_brand_id",
                        "term_brand", "term_model", "suppt_double_card", "suppt_lte", "screen_size"]
    data_5g_user.columns = ["user_id"]

    data_pos1 = data_pos.dropna().reset_index(drop=True)
    data_neg1 = data_neg.dropna().reset_index(drop=True)
    # data_neg.fillna(method="backfill")

    # 数据预处理--类别型数据数值替换
    data_neg1["sex"] = data_neg1["sex"].astype("int")
    data_neg1["user_state"] = data_neg1["user_state"].astype("int")
    data_neg1["cust_type"] = data_neg1["cust_type"].astype("int")
    data_neg1["suppt_double_card"] = data_neg1["suppt_double_card"].astype("int")
    data_neg1["suppt_lte"] = data_neg1["suppt_double_card"].astype("suppt_lte")

    data_pos1["sex"] = data_pos1["sex"].astype("int")
    data_pos1["user_state"] = data_pos1["user_state"].astype("int")
    data_pos1["cust_type"] = data_pos1["cust_type"].astype("int")
    data_pos1["suppt_double_card"] = data_pos1["suppt_double_card"].astype("int")
    data_pos1["suppt_lte"] = data_pos1["suppt_double_card"].astype("suppt_lte")

    data_pos1["vip_type"] = data_pos1["vip_type"].replace("VC1208", 1).replace("VC1603", 2).replace("VC0000",
                                                                                                    3).replace("VC1607",
                                                                                                               4).replace(
        "VC160B", 5).replace("000", 6).replace("101", 7).replace("102", 8).replace("103", 9).replace("104", 10)
    data_neg1["vip_type"] = data_neg1["vip_type"].replace("VC1208", 1).replace("VC1603", 2).replace("VC0000",
                                                                                                    3).replace("VC1607",
                                                                                                               4).replace(
        "VC160B", 5).replace("000", 6).replace("101", 7).replace("102", 8).replace("103", 9).replace("104", 10)

    # 数据预处理--数值型数据标准化处理
    data_pos1["age"] = (data_pos1["age"] - data_pos1["age"].mean()) / data_pos1["age"].std()
    data_pos1["screen_size"] = (data_pos1["screen_size"] - data_pos1["screen_size"].mean()) / data_pos1[
        "screen_size"].std()
    data_pos1["onnet_day_num"] = (data_pos1["onnet_day_num"] - data_pos1["onnet_day_num"].mean()) / data_pos1[
        "onnet_day_num"].std()
    data_pos1["avg_bill_fee"] = (data_pos1["avg_bill_fee"] - data_pos1["avg_bill_fee"].mean()) / data_pos1[
        "avg_bill_fee"].std()
    data_pos1["gprs_down_flow"] = (data_pos1["gprs_down_flow"] - data_pos1["gprs_down_flow"].mean()) / data_pos1[
        "gprs_down_flow"].std()

    data_neg1["age"] = (data_neg1["age"] - data_neg1["age"].mean()) / data_neg1["age"].std()
    data_neg1["sell_price"] = (data_neg1["sell_price"] - data_neg1["sell_price"].mean()) / data_neg1["sell_price"].std()
    data_neg1["screen_size"] = (data_neg1["screen_size"] - data_neg1["screen_size"].mean()) / data_neg1[
        "screen_size"].std()
    data_neg1["onnet_day_num"] = (data_neg1["onnet_day_num"] - data_neg1["onnet_day_num"].mean()) / data_neg1[
        "onnet_day_num"].std()
    data_neg1["avg_bill_fee"] = (data_neg1["avg_bill_fee"] - data_neg1["avg_bill_fee"].mean()) / data_neg1[
        "avg_bill_fee"].std()
    data_neg1["gprs_down_flow"] = (data_neg1["gprs_down_flow"] - data_neg1["gprs_down_flow"].mean()) / data_neg1[
        "gprs_down_flow"].std()

    data_full = pd.concat([data_pos1, data_neg1], axis=0, ignore_index=True, sort=False, keys=["user_id"]).reset_index(
        drop=True)
    data_full["if_5g_prod_pay"] = None

    data_5g_user_id = np.array(data_5g_user["user_id"]).reshape(-1, ).tolist()
    data_full_user_id = np.array(data_full["user_id"].reset_index(drop=True)).reshape(-1, ).tolist()
    for i in range(len(data_full_user_id)):
        if i in data_5g_user_id:
            data_full["if_5g_prod_pay"][i:(i + 1)] = 1
        else:
            data_full["if_5g_prod_pay"][i:(i + 1)] = 0

    data_full_np = data_full.values

    # 基于 k-means 的用户分群
    model1 = KMeans(init="k-means++", n_jobs=-1, max_iter=100, n_clusters=6, n_init=8, tol=0.001)
    model1.fit(data_full_np)
    labels = model1.labels_.reshape(-1, ).tolist()
    labels_ = pd.DataFrame(labels, columns=["cluster_label"]).reset_index(drop=True)
    # data_full_cluster = pd.concat([data_full, labels_], axis=1, ignore_index=True, sort=False).reset_index(drop=True)

    label_pos = pd.DataFrame(np.ones(shape=(data_pos.shape[0],)).tolist())
    label_neg = pd.DataFrame(np.zeros(shape=(data_neg.shape[0],)).tolist())
    data_pos2 = pd.concat([data_pos1, label_pos], axis=1, ignore_index=True).reset_index(drop=True)
    data_neg2 = pd.concat([data_neg1, label_neg], axis=1, ignore_index=True).reset_index(drop=True)
    data_full2_ = pd.concat([data_pos2, data_neg2], axis=0, ignore_index=True, sort=False).reset_index(drop=True)
    data_full_cluster = pd.concat([data_full2_, labels_], axis=1, ignore_index=True, sort=False).reset_index(drop=True)

    # 特征选择和基于随机森林算法的分类预测
    for i in labels:
        data_full3 = data_full_cluster.loc[(data_full_cluster["cluster_label"] == i)].drop(
            ["user_id", "phone_no", "term_brand_id", "term_brand"], axis=1)
        data_train, data_test = train_test_split(data_full3.drop(["cluster_label"]).values, test_size=0.3,
                                                 random_state=np.random.RandomState(),
                                                 shuffle=True)
        x_train, y_train, x_test, y_test = data_train[:, :15], data_train[:, -1], data_test[:, :15], data_test[:, -1]

        # 基于卡方检验的特征选择
        # x_train_ = SelectKBest(chi2, k=10).fit(x_train, y_train.astype("int"))
        # 基于F检验的特征选择
        x_train_ = SelectKBest(f_classif, k=12).fit(x_train, y_train.astype("int"))

        # 分类预测--基于随机森林算法
        model3 = RandomForestClassifier(bootstrap=True, random_state=np.random.RandomState(), n_jobs=-1,
                                        n_estimators=100, max_depth=10)
        model3.fit(x_train_, y_train.astype("int"))
        y_pred = model3.predict(x_test)

        cm = confusion_matrix(y_pred=y_pred.astype("int"), y_true=y_test.astype("int"))
        acc = accuracy_score(y_true=y_test.astype("int"), y_pred=y_pred.astype("int"))
        class_rep = classification_report(y_pred=y_pred.astype("int"), y_true=y_test.astype("int"))

        print("此次模型预测的混淆矩阵结果为：", cm)
        print("此次模型预测的准确率为：", acc)
        print("此次模型预测的分类报告结果为：", class_rep)

        with open(file=model_file_path_root + "brand_danger_model_" + str(date_now) + ".pkl", mode="w",
                  encoding="utf-8"):
            pickle.dump(model3, model_file_path_root + "brand_danger_model_" + str(date_now) + ".pkl")
            print("完成模型持久化!")

        data_neg1_ = data_neg1.drop(["user_id", "phone_no", "term_brand_id", "term_brand"], axis=1)
        data_pos1_ = data_pos1.drop(["user_id", "phone_no", "term_brand_id", "term_brand"], axis=1)
        data10 = pd.concat([data_neg1_.reset_index(drop=True), data_pos1_.reset_index(drop=True)], axis=0,
                           ignore_index=True, sort=True)

        neg_label_pred = model3.predict(data10).reshape(-1, ).tolist()
        neg_label_pred_prob = np.mean(model3.predict_proba(data10), axis=1).reshape(-1, ).tolist()

        result = pd.concat([data_neg1["user_id"], data_neg1["phone_no"], neg_label_pred, neg_label_pred_prob],
                           axis=1, ignore_index=True).reset_index(drop=True)
        date_now = datetime.datetime.now()
        result.columns = ["user_id", "phone_no", "pred_label", "pred_label_prob"]
        result.to_csv(result_file_path_root + "5g_phone_user_pred_" + str(date_now) + ".csv",
                      encoding="utf-8", mode="w",
                      header=["user_id", "phone_no", "label_pred", "label_prob_pred"], index=False)
        print("模型的预测结果已写入文件！")

        return data_full, result


""" sql 执行函数"""


def pysqldf(q):
    a = sqldf(q, globals())
    return a


""" 原因细分"""


def phone_user_reason(data_full1_, sql1, sql2, sql3, sql4, sql5, sql6):
    data_full1_.columns = ["user_id", "phone_no", "age", "user_state", "sex", "onnet_day_num", "cust_type", "vip_type",
                           "if_global", "avg_bill_fee", "gprs_down_flow", "sell_price", "use_date", "term_brand_id",
                           "term_brand", "term_model", "suppt_double_card", "suppt_lte", "screen_size"]
    # pysqldf = lambda q: sqldf(q, globals())
    # 是否终端使用时间长
    data_ = pysqldf(sql1)
    # 是否高流量需求用户
    data1 = pysqldf(sql2).drop(["user_id"])
    # 是否高价值用户
    data2 = pysqldf(sql3).drop(["user_id"])
    # 当前终端是否高价值终端
    data3 = pysqldf(sql4).drop(["user_id"])
    # 是否换机频繁
    data4 = pysqldf(sql5).drop(["user_id"])
    # 是否合约终端且快到期
    data5 = pysqldf(sql6).drop(["user_id"])

    data = pd.concat([data_, data1, data2, data3, data4, data5], axis=1, sort=False, ignore_index=True,
                     keys=["user_id"]).reset_index(
        drop=True)
    data.columns = ["user_id", "if_use_long", "if_flow_high", "if_value_high", "if_phone_price_high", "if_change_short",
                    "if_deadline"]
    return data


"""终端偏好识别和价格挡位分析"""


def terminal_info(data_full2_, sql1, sql2, sql3):
    data_full2_.columns = ["user_id", "phone_no", "age", "user_state", "sex", "onnet_day_num", "cust_type", "vip_type",
                           "if_global", "avg_bill_fee", "gprs_down_flow", "sell_price", "use_date", "term_brand_id",
                           "term_brand", "term_model", "suppt_double_card", "suppt_lte", "screen_size"]
    # pysqldf_ = lambda q: sqldf(q, globals())
    # 终端品牌1
    data1 = pysqldf(sql1)
    # 终端品牌2
    data2 = pysqldf(sql2).drop(["user_id"])
    # 价格挡位
    data3 = pysqldf(sql3).drop(["user_id"])

    data = pd.concat([data1, data2, data3], axis=1, sort=False, ignore_index=True, keys=["user_id"]).reset_index(
        drop=True)
    data.columns = ["user_id", "phone_brand1", "phone_brand2", "brand_price_interval"]
    return data


"""模型结果数据合并"""


def result_concat(result_, data1_, data4_, result_file_path_root):
    date_now = datetime.datetime.now()
    final_result = pd.concat([result_, data1_, data4_], axis=1, sort=False, ignore_index=True,
                             keys=["user_id"]).reset_index(drop=True)
    final_result.to_csv(result_file_path_root + "5g_phone_user_pred_" + str(date_now) + ".csv",
                        encoding="utf-8", mode="w",
                        header=["user_id", "phone_no", "label_pred", "label_prob_pred", "phone_brand1", "phone_brand2",
                                "brand_price_interval", "if_use_long", "if_flow_high", "if_value_high",
                                "if_phone_price_high", "if_change_short",
                                "if_deadline"], index=False)
    print("模型的预测结果已写入文件！")


if __name__ == '__main__':
    conf_file = "/kafka/model/hb_model_test/model_2020/5g_phone_user_pred/conf_test2.ini"
    config = configparser.ConfigParser()
    config.read(conf_file, encoding="utf-8")

    sql11_ = config.get("sql1", "sql11")
    sql12_ = config.get("sql1", "sql12")
    sql13_ = config.get("sql1", "sql13")

    sql21_ = config.get("sql2", "sql21")
    sql22_ = config.get("sql2", "sql22")
    sql23_ = config.get("sql2", "sql23")
    sql24_ = config.get("sql2", "sql24")
    sql25_ = config.get("sql2", "sql25")
    sql26_ = config.get("sql2", "sql26")

    sql31_ = config.get("sql3", "sql31")
    sql32_ = config.get("sql3", "sql32")
    sql33_ = config.get("sql3", "sql33")

    data_file_path_root_ = "/kafka/model/hb_model_test/model_2020/5g_phone_user_pred/data/"
    model_file_path_root_ = "/kafka/model/hb_model_test/model_2020/5g_phone_user_pred/model/"
    result_file_path_root_ = "/kafka/model/hb_model_test/model_2020/5g_phone_user_pred/result/"

    hive_data_read(sql1=sql11_, sql2=sql12_, sql3=sql13_, data_file_path_root1=data_file_path_root_)
    print("训练数据已生成！")
    start_time1 = datetime.datetime.now()
    data_full2, result2 = user_recog(data_file_path_root2=data_file_path_root_,
                                     model_file_path_root=model_file_path_root_,
                                     result_file_path_root=result_file_path_root_)
    end_time1 = datetime.datetime.now()
    print("用户识别模型训练和预测任务已完成！")
    time_interval1 = format(end_time1 - start_time1)
    print("此次训练和预测共耗时：", time_interval1)

    start_time2 = datetime.datetime.now()
    data2_ = phone_user_reason(data_full1_=data_full2, sql1=sql21_, sql2=sql22_, sql3=sql23_, sql4=sql24_, sql5=sql25_,
                               sql6=sql26_)
    end_time2 = datetime.datetime.now()
    print("原因细分模型训练和预测任务已完成！")
    time_interval2 = format(end_time2 - start_time2)
    print("此次训练和预测共耗时：", time_interval2)

    start_time3 = datetime.datetime.now()
    data3_ = terminal_info(data_full2_=data_full2, sql1=sql31_, sql2=sql32_, sql3=sql33_)
    end_time3 = datetime.datetime.now()
    print("终端偏好识别模型和价格挡位偏好分析模型训练和预测任务已完成！")
    time_interval3 = format(end_time3 - start_time3)
    print("此次训练和预测共耗时：", time_interval3)

    start_time4 = datetime.datetime.now()
    result_concat(result_=result2, data1_=data2_, data4_=data3_, result_file_path_root=result_file_path_root_)
    end_time3 = datetime.datetime.now()
    print("终端偏好识别模型和价格挡位偏好分析模型训练和预测任务已完成！")
    time_interval3 = format(end_time3 - start_time3)
    print("此次训练和预测共耗时：", time_interval3)
