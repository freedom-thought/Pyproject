# -*- encoding: utf-8 -*-
# Authors: Wang KM


import calendar
import configparser
import csv
import datetime
import os
import time
import warnings

import numpy as np
import pandas
import pandas as pd
from pyhive import hive
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

# ###################################################
"""定期自动化修改 sql 中取数日期"""


def sql_date_change(sql):
    if int(time.strftime("%m")) == 1:
        date_old = str(int(time.strftime("%Y")) - 1) + "1130"
        month_old = str(int(time.strftime("%Y")) - 1) + "11"

        date_new = str(int(time.strftime("%Y")) - 1) + "1231"
        month_new = str(int(time.strftime("%Y")) - 1) + "12"

        new_sql = sql.replace('"' + date_old + '"', '"' + date_new + '"')
        new_sql = new_sql.replace('"' + month_old + '"', '"' + month_new + '"')

    elif int(time.strftime("%m")) == 2:
        date_old = str(int(time.strftime("%Y")) - 1) + "1231"
        month_old = str(int(time.strftime("%Y")) - 1) + "12"

        date_new = str(int(time.strftime("%Y"))) + "0131"
        month_new = str(int(time.strftime("%Y"))) + "01"

        new_sql = sql.replace('"' + date_old + '"', '"' + date_new + '"')
        new_sql = new_sql.replace('"' + month_old + '"', '"' + month_new + '"')
    else:
        if int(time.strftime("%m")) == 12:
            date_old = str(int(time.strftime("%Y"))) + "1031"
            month_old = str(int(time.strftime("%Y"))) + "10"

            date_new = str(int(time.strftime("%Y"))) + "1130"
            month_new = str(int(time.strftime("%Y"))) + "11"

            new_sql = sql.replace('"' + date_old + '"', '"' + date_new + '"')
            new_sql = new_sql.replace('"' + month_old + '"', '"' + month_new + '"')
        elif int(time.strftime("%m")) == 11:
            date_old = str(int(time.strftime("%Y"))) + "0930"
            month_old = str(int(time.strftime("%Y"))) + "09"

            date_new = str(int(time.strftime("%Y"))) + "1031"
            month_new = str(int(time.strftime("%Y"))) + "10"

            new_sql = sql.replace('"' + date_old + '"', '"' + date_new + '"')
            new_sql = new_sql.replace('"' + month_old + '"', '"' + month_new + '"')
        else:
            days1 = calendar.monthrange(int(time.strftime("%Y")), int(time.strftime("%m")) - 2)[1]
            date_old = str(int(time.strftime("%Y"))) + "0" + str(int(time.strftime("%m")) - 2) + str(days1)
            month_old = str(int(time.strftime("%Y"))) + "0" + str(int(time.strftime("%m")) - 2)

            days2 = calendar.monthrange(int(time.strftime("%Y")), int(time.strftime("%m")) - 1)[1]
            date_new = str(int(time.strftime("%Y"))) + "0" + str(int(time.strftime("%m")) - 1) + str(days2)
            month_new = str(int(time.strftime("%Y"))) + "0" + str(int(time.strftime("%m")) - 1)

            new_sql = sql.replace('"' + date_old + '"', '"' + date_new + '"')
            new_sql = new_sql.replace('"' + month_old + '"', '"' + month_new + '"')

    return new_sql


"""原始数据读取"""


def csv_process_to_model(sql):
    conn = hive.Connection(host="10.29.30.164", port=10000, username="yxzx", password="yxzx", database="yxzx",
                           auth="CUSTOM")
    cursor1 = conn.cursor()
    cursor1.execute(sql)

    data1 = cursor1.fetchall()

    data_original1 = pd.DataFrame(list(data1))

    data_original1.columns = ["user_id", "xq_code", "pre_night_latitude", "pre_night_longitude"]

    data_original1 = data_original1.drop_duplicates(subset=["pre_night_latitude", "pre_night_longitude"], keep="first")

    data_original2 = data_original1[["xq_code", "user_id"]]
    data_original2 = data_original2.groupby(
        ["xq_code"], as_index=False)["user_id"].count()

    data_original1_list = []
    data_original11 = data_original1.iloc[:data_original1.shape[0] * 25, :]
    data_original12 = data_original1.iloc[data_original1.shape[0] * 25:data_original1.shape[0] * 50, :]
    data_original13 = data_original1.iloc[data_original1.shape[0] * 50:data_original1.shape[0] * 75, :]
    data_original14 = data_original1.iloc[data_original1.shape[0] * 75:, :]
    data_original1_list.append(data_original11)
    data_original1_list.append(data_original12)
    data_original1_list.append(data_original13)
    data_original1_list.append(data_original14)

    data_original2_list = []
    data_original21 = data_original2.iloc[:data_original2.shape[0] * 25, :]
    data_original22 = data_original2.iloc[data_original2.shape[0] * 25:data_original2.shape[0] * 50, :]
    data_original23 = data_original2.iloc[data_original2.shape[0] * 50:data_original2.shape[0] * 75, :]
    data_original24 = data_original2.iloc[data_original2.shape[0] * 75:, :]
    data_original2_list.append(data_original21)
    data_original2_list.append(data_original22)
    data_original2_list.append(data_original23)
    data_original2_list.append(data_original24)

    conn.commit()
    cursor1.close()
    conn.close()

    return data_original1_list, data_original2_list


"""模型计算"""


def location_com_allpute(data_original1, data_original2, threshold1, threshold2, file_path, file_path_result,
                         file_path_original):
    location_data = pd.read_csv(file_path, error_bad_lines=False, low_memory=False, encoding="utf8",
                                skip_blank_lines=True, engine="c", header=None)

    location_data.columns = ["xq_code", "east_longitude", "west_longitude", "north_latitude", "south_latitude"]

    xq_code1 = data_original2["xq_code"].tolist()

    loop_outer = data_original2.shape[0]
    print("完成模型所需数据的读取，开始模型计算！")

    for i in range(loop_outer):
        try:
            xq_code_ = xq_code1[i]

            x2 = data_original1.loc[data_original1["xq_code"]
                                    == xq_code_]["pre_night_latitude"]
            x2 = pd.DataFrame(x2, dtype=np.float)
            x2_ = np.array(x2)

            y2 = data_original1.loc[data_original1["xq_code"]
                                    == xq_code_]["pre_night_longitude"]
            y2 = pd.DataFrame(y2, dtype=np.float)
            y2_ = np.array(y2)

            """基于孤立森林检测并去除异常点"""
            """x2_"""
            warnings.filterwarnings("ignore")
            isa_model = IsolationForest(n_jobs=-1, n_estimators=x2_.shape[0], contamination=0.5, bootstrap=False)
            isa_model.fit(x2_)
            anomaly_result = isa_model.predict(x2_)
            anomaly_result = anomaly_result.reshape(anomaly_result.shape[0], 1)
            label_anomaly_index1 = np.where(anomaly_result == -1)[0].tolist()

            x2_ = x2_.reshape(x2_.shape[0], ).tolist()
            x2_new = []
            for m in range(len(x2_)):
                if m not in label_anomaly_index1:
                    x2_new.append(x2_[m])
            x2_new = np.array(x2_new).reshape(len(x2_new), 1)

            """y2_"""
            warnings.filterwarnings("ignore")
            isa_model = IsolationForest(n_jobs=-1, n_estimators=y2_.shape[0], contamination=0.5, bootstrap=False)
            isa_model.fit(y2_)
            anomaly_result = isa_model.predict(y2_)
            anomaly_result = anomaly_result.reshape(anomaly_result.shape[0], 1)
            label_anomaly_index2 = np.where(anomaly_result == -1)[0].tolist()

            y2_ = y2_.reshape(y2_.shape[0], ).tolist()
            y2_new = []
            for m in range(len(y2_)):
                if m not in label_anomaly_index2:
                    y2_new.append(y2_[m])
            y2_new = np.array(y2_new).reshape(len(y2_new), 1)

            """基于基本统计指标的检测与划分"""
            x2_med, y2_med = np.median(x2_new), np.median(y2_new)
            x2_std, y2_std = np.std(x2_new), np.std(y2_new)

            x2_num_total = x2_new.shape[0]
            y2_num_total = y2_new.shape[0]

            x2_num_ = 0
            y2_num_ = 0

            for item in x2_new:
                np.seterr(divide="ignore", invalid="ignore")
                if float(abs(item - x2_med) / x2_std) > float(threshold1):
                    x2_num_ += 1

            for item in y2_new:
                np.seterr(divide="ignore", invalid="ignore")
                if float(abs(item - y2_med) / y2_std) > float(threshold1):
                    y2_num_ += 1

            if x2_num_total != 0 and y2_num_total != 0 and float(x2_num_ / x2_num_total) < float(threshold2) and float(
                    y2_num_ / y2_num_total) < float(threshold2):
                # 基于 DBSCAN 算法的计算模块
                # 1.构建 k=1 的 dbscan 聚类模型，并拟合数据
                dbscan_model1 = DBSCAN(eps=0.18, min_samples=35, algorithm="auto", metric="euclidean", n_jobs=1).fit(
                    x2_new)
                dbscan_model2 = DBSCAN(eps=0.18, min_samples=35, algorithm="auto", metric="euclidean", n_jobs=1).fit(
                    y2_new)

                # 2.计算输出属于聚类的核心样本点
                dbscan_components1 = dbscan_model1.components_
                dbscan_components2 = dbscan_model2.components_
                print("dbscan_components1", dbscan_components1)
                print("dbscan_components2", dbscan_components2)

                # 3.计算输出聚类中心
                dbscan_core_sample_indices1 = dbscan_model1.core_sample_indices_
                dbscan_core_sample_indices2 = dbscan_model2.core_sample_indices_
                print("dbscan_core_sample_indices1", dbscan_core_sample_indices1)
                print("dbscan_core_sample_indices2", dbscan_core_sample_indices2)

                # 3.计算输出各小区类的核心点和成员点
                labels1 = dbscan_model1.labels_.tolist()
                labels2 = dbscan_model2.labels_.tolist()

                label1_1 = x2_new[labels1.index(labels1 == 1)]
                label2_1 = y2_new[labels2.index(labels2 == 1)]

                east_longitude_new = np.max(label1_1)
                west_longitude_new = np.min(label1_1)
                north_latitude_new = np.max(label2_1)
                south_latitude_new = np.max(label2_1)

                print(east_longitude_new, "\n", west_longitude_new, "\n", north_latitude_new, "\n", south_latitude_new)

                x2_perc_up, y2_perc_up = np.percentile(
                    x2_new, 80, interpolation="linear"), np.percentile(
                    y2_new, 80, interpolation="linear")
                x2_perc_low, y2_perc_low = np.percentile(
                    x2_new, 20, interpolation="linear"), np.percentile(
                    y2_new, 20, interpolation="linear")

                x2_max, y2_max = np.max(x2_new), np.max(y2_new)
                x2_min, y2_min = np.min(x2_new), np.min(y2_new)

                # 去除最大最小值
                x2_max_index = np.where(x2_new == x2_max)
                x2_min_index = np.where(x2_new == x2_min)
                x2_new = np.delete(x2_new, x2_max_index)
                x2_new = np.delete(x2_new, x2_min_index)
                x2_new = x2_new.reshape(x2_new.shape[0], 1)

                y2_max_index = np.where(y2_new == y2_max)
                y2_min_index = np.where(y2_new == y2_min)
                y2_new = np.delete(y2_new, y2_max_index)
                y2_new = np.delete(y2_new, y2_min_index)
                y2_new = y2_new.reshape(y2_new.shape[0], 1)

                if len(x2_new) != 0 and len(y2_new) != 0:
                    x2_perc_up_, y2_perc_up_ = np.percentile(
                        x2_new, 80, interpolation="midpoint"), np.percentile(
                        y2_new, 80, interpolation="midpoint")
                    x2_perc_low_, y2_perc_low_ = np.percentile(
                        x2_new, 20, interpolation="midpoint"), np.percentile(
                        y2_new, 20, interpolation="midpoint")

                    x2_max_new, y2_max_new = x2_perc_up_, y2_perc_up_
                    x2_min_new, y2_min_new = x2_perc_low_, y2_perc_low_

                else:
                    x2_max_new, y2_max_new = x2_perc_up, y2_perc_up
                    x2_min_new, y2_min_new = x2_perc_low, y2_perc_low

                # 基于 pre_night 数据计算
                north_latitude = x2_max_new
                south_latitude = x2_min_new
                east_longitude = y2_max_new
                west_longitude = y2_min_new
                insert_time = format(datetime.datetime.now())

                if (east_longitude - west_longitude) > 0.015 and (north_latitude - south_latitude) > 0.015:
                    location_data[location_data["xq_code"] == xq_code_].to_csv(
                        file_path_original, index=False, header=False, mode="a", encoding="utf8")
                    print("第 %d 个小区( %s )的位置坐标计算取消，并输出网管位置数据！" % ((i + 1), xq_code_))
                elif (east_longitude - west_longitude) > 0.018 or (north_latitude - south_latitude) > 0.018:
                    location_data[location_data["xq_code"] == xq_code_].to_csv(
                        file_path_original, index=False, header=False, mode="a", encoding="utf8")
                    print("第 %d 个小区( %s )的位置坐标计算取消，并输出网管位置数据！" % ((i + 1), xq_code_))
                elif (east_longitude - west_longitude) < 0.0003 or (north_latitude - south_latitude) < 0.0003:
                    location_data[location_data["xq_code"] == xq_code_].to_csv(
                        file_path_original, index=False, header=False, mode="a", encoding="utf8")
                    print("第 %d 个小区( %s )的位置坐标计算取消，并输出网管位置数据！" % ((i + 1), xq_code_))
                elif (east_longitude - west_longitude) < 0.0005 and (north_latitude - south_latitude) < 0.0005:
                    location_data[location_data["xq_code"] == xq_code_].to_csv(
                        file_path_original, index=False, header=False, mode="a", encoding="utf8")
                    print("第 %d 个小区( %s )的位置坐标计算取消，并输出网管位置数据！" % ((i + 1), xq_code_))
                else:
                    data = [xq_code_,
                            str(east_longitude),
                            str(west_longitude),
                            str(north_latitude),
                            str(south_latitude),
                            str(insert_time)]
                    data = list(np.array(data).reshape(1, 6))
                    data_result = pd.DataFrame(data,
                                               columns=["xq_code", "east_longitude",
                                                        "west_longitude", "north_latitude",
                                                        "south_latitude", "insert_time"])

                    """位置计算结果写入文件"""
                    data_result.to_csv(file_path_result, index=False,
                                       header=False,
                                       mode="a", encoding="utf8")
                    print("已完成第 %d 个小区( %s )的位置坐标计算" % ((i + 1), xq_code_))

            else:
                location_data[location_data["xq_code"] == xq_code_].to_csv(
                    file_path_original, index=False, header=False, mode="a",
                    encoding="utf8")
                print("第 %d 个小区( %s )的位置坐标计算取消，并输出网管位置数据！" % ((i + 1), xq_code_))

        except (ValueError, IndexError, OSError):
            pass
        continue

    print("已完成全部小区的位置计算及数据输出！")


"""比较网管测和模型计算数据，进行小区坐标纠正"""


def location_com_allp(file_path1, file_path2, location_right1, location_wrong1, file_original):
    data1 = []
    with open(file_path1, encoding="utf-8") as csvfile1:
        csv_reader = csv.reader(csvfile1)
        for row in csv_reader:
            data1.append(row)

    data2 = []
    with open(file_path2, encoding="utf-8") as csvfile2:
        csv_reader = csv.reader(csvfile2)
        for row in csv_reader:
            data2.append(row)

    data1 = pd.DataFrame(data1)
    data2 = pd.DataFrame(data2)

    data1.columns = ["xq_code", "east_longitude1", "west_longitude1", "north_latitude1", "south_latitude1"]
    data2.columns = ["xq_code", "east_longitude2", "west_longitude2", "north_latitude2", "south_latitude2",
                     "insert_time"]

    data1 = data1[(data1["east_longitude1"] != "") & (data1["west_longitude1"] != "")]

    data3 = pd.merge(left=data2, right=data1, how="inner", on=["xq_code"], sort=False).reset_index().drop(["index"],
                                                                                                          axis=1)

    xq_code = data3["xq_code"].tolist()

    data1_east_longitude = data3["east_longitude1"]
    data1_west_longitude = data3["west_longitude1"]
    data1_north_latitude = data3["north_latitude1"]
    data1_south_latitude = data3["south_latitude1"]

    data2_east_longitude = data3["east_longitude2"]
    data2_west_longitude = data3["west_longitude2"]
    data2_north_latitude = data3["north_latitude2"]
    data2_south_latitude = data3["south_latitude2"]

    for i in range(len(xq_code)):
        try:
            a1 = np.double(data1_east_longitude[i]) - np.double(data1_west_longitude[i])
            b1 = np.double(data1_north_latitude[i]) - np.double(data1_south_latitude[i])
            a2 = np.double(data2_east_longitude[i]) - np.double(data2_west_longitude[i])
            b2 = np.double(data2_north_latitude[i]) - np.double(data2_south_latitude[i])

            if np.double(data2_east_longitude[i]) < np.double(data1_west_longitude[i]) or np.double(
                    data2_west_longitude[i]) > np.double(data1_east_longitude[i]) or np.double(
                data2_south_latitude[i]) > np.double(data1_north_latitude[i]) or np.double(
                data2_north_latitude[i]) < np.double(data1_south_latitude[i]):
                if a2 == 0 or b2 == 0 and (a1 - a2) > 0.02 or (b1 - b2) > 0.02:
                    data1.loc[data1["xq_code"] == xq_code[i]].to_csv(file_original, header=False, index=False, mode="a",
                                                                     encoding="utf8")
                elif a2 == 0 or b2 == 0 and (a1 - a2) < 0.01 and (b1 - b2) < 0.01:
                    data3[i:(i + 1)].to_csv(location_right1, header=False, index=False, mode="a", encoding="utf8")
                elif (a2 - a1) > 0.015 or (b2 - b1) > 0.015 and b1 != 0 and a1 != 0:
                    data3[i:(i + 1)].to_csv(location_right1, header=False, index=False, mode="a", encoding="utf8")
                elif a1 == 0 and b1 == 0 and a2 != 0 or b2 != 0:
                    data3[i:(i + 1)].to_csv(location_wrong1, header=False, index=False, mode="a", encoding="utf8")
                elif a1 < 0.0005 or b1 < 0.0005 and a2 == 0 or b2 == 0:
                    data3[i:(i + 1)].to_csv(location_right1, header=False, index=False, mode="a", encoding="utf8")
                else:
                    data3[i:(i + 1)].to_csv(location_wrong1, header=False, index=False, mode="a", encoding="utf8")
            else:
                if a2 != 0 and b2 != 0 and (a1 - a2) > 0.02 and (b1 - b2) > 0.02:
                    data3[i:(i + 1)].to_csv(location_wrong1, header=False, index=False, mode="a", encoding="utf8")
                elif a2 == 0 or b2 == 0 and (a1 - a2) <= 0.01 and (b1 - b2) <= 0.01:
                    data3[i:(i + 1)].to_csv(location_right1, header=False, index=False, mode="a", encoding="utf8")
                elif a2 == 0 or b2 == 0 and (a1 - a2) > 0.02 and (b1 - b2) > 0.02:
                    data1.loc[data1["xq_code"] == xq_code[i]].to_csv(file_original, header=False, index=False, mode="a",
                                                                     encoding="utf8")
                elif a2 == 0 and b2 == 0 and (a1 - a2) > 0.01 and (b1 - b2) > 0.01:
                    data3[i:(i + 1)].to_csv(location_wrong1, header=False, index=False, mode="a", encoding="utf8")
                elif a1 <= 0.0005 or b1 <= 0.0005 and a2 > 0.003 and b2 > 0.003:
                    data1.loc[data1["xq_code"] == xq_code[i]].to_csv(file_original, header=False, index=False, mode="a",
                                                                     encoding="utf8")
                elif a1 <= 0.0005 or b1 <= 0.0005 and a1 > 0 > 0 and a2 == 0 or b2 == 0:
                    data3[i:(i + 1)].to_csv(location_right1, header=False, index=False, mode="a", encoding="utf8")
                elif a1 == 0 and b1 == 0 and a2 != 0 or b2 != 0:
                    data3[i:(i + 1)].to_csv(location_wrong1, header=False, index=False, mode="a", encoding="utf8")
                elif (a2 - a1) > 0.015 or (b2 - b1) > 0.015 and b1 != 0 and a1 != 0:
                    data3[i:(i + 1)].to_csv(location_right1, header=False, index=False, mode="a", encoding="utf8")
                else:
                    data3[i:(i + 1)].to_csv(location_wrong1, header=False, index=False, mode="a", encoding="utf8")

        except (ValueError, IndexError, OSError):
            pass
        continue


"""数据整合"""


def file_concat(file_path_now, file_path_wrong, file_path_right):
    open(file_path_now, encoding="utf8", mode="a")

    data1 = pd.read_csv(file_path_wrong, encoding="utf8",
                        usecols=[0, 1, 2, 3, 4, 5], low_memory=False, index_col=False, engine="c",
                        error_bad_lines=False)
    data1.columns = ["xq_code", "east_longitude", "west_longitude", "north_latitude", "south_latitude", "insert_time"]

    file_size = os.path.getsize(file_path_right)

    if file_size == 0:
        data1.to_csv(file_path_now, encoding="utf8", mode="a", index=False, header=False)
        print("模型纠正和网管无误数据合并完成，并写入文件!")
    else:
        data2 = pd.read_csv(file_path_right, encoding="utf8",
                            usecols=[0, 6, 7, 8, 9, 5], low_memory=False, index_col=False, engine="c",
                            error_bad_lines=False)

        data3 = data2.iloc[:, [0, 2, 3, 4, 5]]
        data4 = data2.iloc[:, 1]

        data2_ = pd.concat([data3, data4], axis=1, ignore_index=True, sort=False, join_axes=None)

        data2_.columns = ["xq_code", "east_longitude", "west_longitude", "north_latitude", "south_latitude",
                          "insert_time"]

        data = pd.concat([data1, data2_], axis=0, ignore_index=True, sort=False, join_axes=None, join="outer")

        data.to_csv(file_path_now, encoding="utf8", mode="a", index=False, header=False)

        print("模型纠正和网管无误数据合并完成，并写入文件!")


"""判断用户坐标是否为数值"""


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


"""找出坐标非数值的数据的索引"""


def str_index_lookup(df_):
    list1 = []
    for i in range(df_.shape[0]):
        if is_number(df_['pre_night_mr_longitude'][i]) and is_number(
                df_['pre_night_mr_latitude'][i]):
            pass
        else:
            list1.append(i)
    return list1


"""框取用户数计算与新小区数据获取"""


def user_count_filter_output(file_path_original1, file_path_original2,
                             file_path_user, file_path_xq_add):
    df = pd.read_csv(file_path_original2)
    usr_df = pd.read_csv(file_path_user)
    usr_df.columns = ["user_id", "pre_night_mr_latitude", "pre_night_mr_longitude"]

    df.columns = ["0", "1", "2", "3", "4", "5"]
    df = df.drop("5", 1)

    df["usr_count"] = None
    df["insert_time"] = None

    # 剔除坐标非数值的数据
    usr_df_index = [i for i in range(usr_df.shape[0])]
    str_index = str_index_lookup(usr_df)
    num_list = list(set(usr_df_index) - set(str_index))
    usr_df = usr_df.loc[num_list]

    # 计数每个小区用户数
    for i in range(0, df.shape[0]):
        test_df = usr_df.loc[(usr_df.pre_night_mr_longitude < df['1'][i]) &
                             (usr_df.pre_night_mr_longitude > df['2'][i]) &
                             (usr_df.pre_night_mr_latitude < df['3'][i]) &
                             (usr_df.pre_night_mr_latitude > df['4'][i])]
        df["usr_count"][i] = test_df.shape[0]
        df["insert_time"][i] = format(datetime.datetime.now())

    df = df.loc[(df["usr_count"] >= 2) & (df["usr_count"] <= 3500)]

    df_final = df.drop("usr_count", 1)

    df_xq_list = df.loc[:, "0"]

    df_xq_list = np.array(df_xq_list).reshape(df_xq_list.shape[0], ).tolist()

    if os.path.exists(file_path_original1):
        xq_original = pd.read_csv(file_path_original1, header=None, low_memory=False, encoding="utf8",
                                  skip_blank_lines=True, engine="c", error_bad_lines=False)

        xq_original_list = xq_original.iloc[:, 0]

        xq_original_list = np.array(xq_original_list).reshape(
            xq_original_list.shape[0], ).tolist()

        open("/kafka/model/location_com_all/location_month/location_add" + "_" + format(
            datetime.datetime.now())[:7] + ".csv", "a", encoding="utf-8")

        a = 0
        for i in range(len(df_xq_list)):
            if df_xq_list[i] not in xq_original_list:
                df_final[i:(i + 1)].to_csv(
                    "/kafka/model/location_com_all/location_month/location_add" + "_" + format(
                        datetime.datetime.now())[:7] + ".csv", index=False,
                    header=False, mode="a", encoding="utf8")
                df_final[i:(i + 1)].to_csv(file_path_original1, index=False, header=False, mode="a", encoding="utf8")
                a += 1

        print("本月模型迭代后新增计算小区个数：%d" % a)

        with open(file_path_xq_add, encoding="utf-8", mode="a") as f:
            f.write(str(datetime.datetime.now().year) + "-" +
                    str(datetime.datetime.now().month) + " : " + str(a) + "\n")
        f.close()

    else:
        open(file_path_original1, "a", encoding="utf-8")
        df_final.to_csv(file_path_original1, index=False, header=False, mode="a", encoding="utf8")


def main_job2_():
    conf_file = "/kafka/model/location_com_all/config/Location_Conf_all.ini"
    config = configparser.ConfigParser()
    config.read(conf_file, encoding="utf-8")
    file_path1_ = config.get("file_path1", "file_path1")
    file_path2_ = config.get("file_path2", "file_path2")
    file_original_path = "/kafka/model/location_com_all/model_middle_data/data_original" + "_" + str(
        datetime.date.today()) + ".csv"

    start_time = datetime.datetime.now()
    print("start_time:", start_time)

    open("/kafka/model/location_com_all/model_middle_data/location_wrong" + "_" + str(datetime.date.today()) + ".csv",
         encoding="utf-8", mode="a")
    open("/kafka/model/location_com_all/model_middle_data/location_right" + "_" + str(datetime.date.today()) + ".csv",
         encoding="utf-8", mode="a")

    location_wrong_path = "/kafka/model/location_com_all/model_middle_data/location_wrong" + "_" + str(
        datetime.date.today()) + ".csv"
    location_right_path = "/kafka/model/location_com_all/model_middle_data/location_right" + "_" + str(
        datetime.date.today()) + ".csv"

    print("开始比较原始数据和模型计算数据，并作两类输出：")
    location_com_allp(file_path1=file_path1_, file_path2=file_path2_, location_right1=location_right_path,
                      location_wrong1=location_wrong_path, file_original=file_original_path)
    print("已完成数据比较，并输出两类数据到三个文件！")

    end_time = datetime.datetime.now()
    print("end_time:", end_time)
    interval_time = round(int((end_time - start_time).seconds) / 60, 2)
    print("共用时：%f 分钟！" % interval_time)

    os.remove(file_path1_)
    os.remove(file_path2_)


def main_job3_():
    conf_file = "/kafka/model/location_com_all/config/Location_Conf_all.ini"
    config = configparser.ConfigParser()
    config.read(conf_file, encoding="utf-8")

    location_now_path = config.get("file_path_original2_", "file_path_original2_")
    location_wrong_path = "/kafka/model/location_com_all/model_middle_data/location_wrong" + "_" + str(
        datetime.date.today()) + ".csv"
    location_right_path = "/kafka/model/location_com_all/model_middle_data/location_right" + "_" + str(
        datetime.date.today()) + ".csv"

    file_concat(file_path_now=location_now_path, file_path_wrong=location_wrong_path,
                file_path_right=location_right_path)


def main_job4_():
    conf_file = "/kafka/model/location_com_all/config/Location_Conf_all.ini"
    config = configparser.ConfigParser()
    config.read(conf_file, encoding="utf-8")

    file_path_original1_ = config.get(
        "file_path_original1_",
        "file_path_original1_")
    file_path_original2_ = config.get(
        "file_path_original2_",
        "file_path_original2_")
    file_path_user_ = config.get("file_path_user_", "file_path_user_")
    file_path_xq_add_ = config.get("file_path_xq_add_", "file_path_xq_add_")

    user_count_filter_output(file_path_original1_, file_path_original2_, file_path_user_, file_path_xq_add_)

    os.remove(file_path_user_)
    os.remove(file_path_original2_)


"""主程序模块"""


def main_job():
    conf_file = "/kafka/model/location_com_all/config/Location_Conf_all.ini"
    config = configparser.ConfigParser()
    config.read(conf_file, encoding="utf-8")
    sql_ = config.get("sql1_", "sql")
    sql_ = sql_date_change(sql_)
    config.set("sql1_", "sql", sql_)

    file_path1_ = config.get("file_path1", "file_path1")
    thres1_ = config.get("thres", "thres")
    thres2_ = config.get("thres2", "thres2")
    file_path2_ = config.get("file_path2", "file_path2")

    start_time = datetime.datetime.now()
    print(format(start_time))
    print("开始读取原始数据并生成模型所需数据！")
    data_original1_lists, data_original2_lists = csv_process_to_model(sql_)
    print("已生成模型所需数据！")
    end_time1 = datetime.datetime.now()
    print(format(end_time1))

    open("/kafka/model/location_com_all/model_middle_data/location_result.csv", "a", encoding="utf-8")

    open("/kafka/model/location_com_all/model_middle_data/data_original" + "_" + str(datetime.date.today()) + ".csv",
         "a", encoding="utf-8")

    file_path_original_ = "/kafka/model/location_com_all/model_middle_data/data_original" + "_" + str(
        datetime.date.today()) + ".csv"

    print("开始模型计算：")
    for i in range(len(data_original1_lists)):
        try:
            data_original1_, data_original2_ = data_original1_lists[i], data_original2_lists[i]
            location_com_allpute(data_original1=data_original1_, data_original2=data_original2_, threshold1=thres1_,
                                 threshold2=thres2_, file_path=file_path1_, file_path_result=file_path2_,
                                 file_path_original=file_path_original_)
            main_job2_()
            main_job3_()
            main_job4_()
            time.sleep(5000)
        except (ValueError, pandas.errors.EmptyDataError):
            open("/kafka/model/location_com_all/location_month/location_add_" + str(datetime.date.today()) + "_" + str(
                i) + "times.csv",
                 mode="a", encoding="utf-8")
            time.sleep(5000)
        continue

    end_time2 = datetime.datetime.now()
    print(format(end_time2))
    interval_time = round(int((end_time2 - start_time).seconds) / 60, 2)
    print("完成模型计算，共用时：%f 分钟！" % interval_time)


if __name__ == '__main__':
    while True:
        if int(datetime.datetime.now().day) == 8 and int(datetime.datetime.now(
        ).hour) == 3 and int(datetime.datetime.now().minute) == 30:
            start_time_ = datetime.datetime.now()
            print("开始模型计算：")
            main_job()
            end_time = datetime.datetime.now()
            interval_time_ = round(int((end_time - start_time_).seconds) / 60, 2)
            print("模型计算完成，此次计算共用时：%f 分钟！" % float(interval_time_))
            time.sleep(60)
        else:
            continue
