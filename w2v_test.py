print(__doc__)


# Encoding: utf-8
# Author: WWang KM
# License: BSD


import os

import jieba
import numpy as np
from gensim.models import word2vec
from keras import regularizers
from keras.layers import LSTM, Dropout, Dense, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion


def w2v_from_gensim(file_source_path):
    files = os.listdir(file_source_path)
    w2v_all = []
    global a
    # 获取每个文件的原始评论数据
    for item in files:
        with open(file=file_source_path + item, encoding="utf-8",
                  mode="r") as f:
            lines = f.readlines()
            result_list = []
            result_list_len = []

            for i in range(len(lines)):
                data = eval(lines[i])
                for keys, values in data.items():
                    if values:
                        values = values.replace("，", "").replace(",", "").replace("（", "").replace("）", "").replace("！",
                                                                                                                    "").replace(
                            "\xa0", "").replace("1", "").replace("10", "").replace("￼", "").replace("?", "").replace(
                            "的", "").replace("了", "").replace(",", "").replace("(", "").replace(")", "").replace("\n",
                                                                                                                 "").replace(
                            "👏", "")
                        result = jieba.cut(values, HMM=True)
                        cut_result = [x for x in result]
                        result_list_len.append(len(cut_result))
                        result_list.append(cut_result)

            a = max(result_list_len)

            result_list_new = []
            for i in range(len(result_list)):
                result_item = result_list[i]
                for j in range(a - len(result_item)):
                    result_item += "的"
                result_list_new.append(result_item)

            """使用 Word2Vec(google gensim Api) 方法对分词结果建词向量"""
            # 初始化 Word2vec 模型
            w2v_model = word2vec.Word2Vec(size=10, window=5, workers=-1, sg=0, hs=1, alpha=0.01, min_alpha=0.001,
                                          iter=100, compute_loss=True, min_count=3)
            # 对原始数据进行模型拟合并构建词向量
            w2v_model.build_vocab(result_list_new)
            w2v_model.train(sentences=result_list_new,
                            total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
            word_vecs = w2v_model.wv.index2word

            word_vec_list = []
            for word in word_vecs:
                word_vec_list.append(list(w2v_model[word]))

            """以每条语料为基础构建整个语料的词向量"""
            corpus_vec = []
            for item1 in result_list_new:
                sentence_vec = []
                for item2 in item1:
                    if item2 in word_vecs:
                        sentence_vec.append(word_vec_list[item1.index(item2)])
                corpus_vec.append(sentence_vec)
            # print(len(corpus_vec[4][4]))

        w2v_all.append(corpus_vec)

    return w2v_all, a


def w2v_from_cnt(file_source_path):
    countvec_model = CountVectorizer(min_df=1)
    tfidfvec_model = TfidfVectorizer(use_idf=True)

    files = os.listdir(file_source_path)
    # 获取每个文件的原始评论数据
    w2v_all = []
    for item1 in files:
        with open(file=file_source_path + item1, encoding="utf-8", mode="r") as f:
            lines = f.readlines()
            result_list = []
            result_list_len = []

            for i in range(len(lines)):
                data = eval(lines[i])
                for keys, values in data.items():
                    if values:
                        values = values.replace("，", "").replace(",", "").replace("（", "").replace("）", "").replace("！",
                                                                                                                    "").replace(
                            "\xa0", "").replace("1", "").replace("10", "").replace("￼", "").replace("?", "").replace(
                            "的",
                            "").replace(
                            "了", "").replace(",", "").replace("(", "").replace(")", "").replace("\n", "").replace("👏",
                                                                                                                  "")
                        result = jieba.cut(values, HMM=True)
                        cut_result = ",".join(result)
                        result_list_len.append(len(cut_result))
                        result_list.append([cut_result])

            a = max(result_list_len)

            result_list_new = []
            for i in range(len(result_list)):
                result_item = result_list[i]
                for j in range(a - len(result_item[0])):
                    result_item[0] += "的"
                result_list_new.append(result_item)

            """使用 countvec 和 tfidfvec 结合的方法对分词结果进行词向量转换"""
            combined_features = FeatureUnion([("counts", countvec_model), ("tfidfvec", tfidfvec_model)]).fit_transform(
                result_list_new[0]).toarray().reshape(-1, ).tolist()
            print(combined_features)

            comment_vec_final = [combined_features]
            for i in range(len(result_list_new) - 1):
                try:
                    combined_features_ = FeatureUnion(
                        [("counts", countvec_model), ("tfidfvec", tfidfvec_model)]).fit_transform(
                        result_list_new[i + 1]).toarray().reshape(-1, ).tolist()
                    comment_vec_final.append(combined_features_)
                except ValueError:
                    pass

            comment_vec_final_new = []

            for item2 in comment_vec_final:
                if len(item2) >= 15:
                    comment_vec_final_new.append(item2[10:18])
                    print(item2[10:18])

            w2v_all.append(comment_vec_final_new)

    return w2v_all


def label_gen(file_source_path):
    # 给每个评论向量添加结果词
    files = os.listdir(file_source_path)
    # 获取每个文件的原始评论数据
    label1 = []
    label2 = []
    for item1 in files:
        with open(file=file_source_path + item1, encoding="utf-8", mode="r") as f:
            lines = f.readlines()
            # 1.生成固定化类别标签
            target_len = len(lines)
            target_all = np.random.uniform(0, 1, target_len).reshape(-1, )
            target_single = np.random.randint(0, 10, target_len).reshape(-1, )
            target_comment = np.hstack((target_all, target_single))
            target_desc = ["非常差", "差", "一般", "好", "非常好"]
            # 2.个性化预测生成情感结论
            target_word_gen = ["立马预定", "店比三家", "需要持续观望", "不会考虑", "卫生差，不推荐"]

            # 3.根据评论中的关键词生成标签
            label_list = []
            for i in range(len(lines)):
                data = eval(lines[i])
                for keys, values in data.items():
                    if values.find("差"):
                        label_list.append("0")
                    elif values.find("干净"):
                        label_list.append("3")
                    elif values.find("好"):
                        label_list.append("4")
                    elif values.find("非常好"):
                        label_list.append("5")
                    elif values.find("但是"):
                        label_list.append("1")
                    else:
                        label_list.append("2")

            # 对标签进行预处理
            target_all_new = []
            for item2 in range(target_all):
                result = round(item2, 3)
                target_all_new.append(result)

            label1.append(target_single)
            label2.append(target_comment)

    return label1, label2


def model1(word_vec1, label1, original_len):
    # 将评论和标签进行整合
    comment_vec_final1 = np.array(word_vec1)
    comment_model_use1 = np.hstack(comment_vec_final1, label1)

    # 生成模型入参数据所需参数(设置用于生成词向量的最大评论长度)
    max_reviews_length = 100

    # 数据集切分
    x_train, x_test = train_test_split(np.array(word_vec1), test_size=0.3)
    y_train, y_test = train_test_split(np.array(label1), test_size=0.3)

    x_train = sequence.pad_sequences(x_train, maxlen=max_reviews_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_reviews_length)

    # LSTM 模型构建、训练和评估
    """1.模型构建"""
    embedding_vector_length = 80
    drop_rate1 = 0.3
    drop_rate2 = 0.5
    recurrent_dropout = 0.5
    activatation_regularizer = 0.01

    model = Sequential()
    model.add(Embedding(input_dim=original_len,
                        output_dim=embedding_vector_length, input_length=30))
    model.add(Dropout(rate=drop_rate1))
    model.add(LSTM(80, use_bias=True, activation="softsign", recurrent_dropout=recurrent_dropout,
                   kernel_initializer="random_uniform", bias_initializer="zeros"))
    model.add(Dropout(drop_rate2))
    model.add(Dense(1, activation='sigmoid',
                    activity_regularizer=regularizers.l2(activatation_regularizer)))

    """2.模型训练"""
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, epochs=10, batch_size=32,
              validation_data=(x_test, y_test), shuffle=True)
    print("模型训练完成！")

    model.save(filepath="D://Resume/si-tech/工作内容/rnn_model_test/rnn_test.h5", overwrite=True,
               include_optimizer="rmsprop")
    print("模型文件保存完成！")

    """3.模型预测评估"""
    scores, acc = model.evaluate(x_test[:100, :], y_test[:100, :], verbose=0)
    print("Accuracy: %.2f%%" % (acc * 100))
    print("Scores: %.2f%%" % (scores * 100))

    # 使用最优模型对评论进行评判预测
    target_label_score = model.predict(x_test[100:, :])
    target_label_predict_prob = model.predict_proba(x_test[100:, ])
    target_label = model.predict_classes(x_test[100:, ])

    print("对各评论预测结果分数为：", target_label_score)
    print("对各评论预测结果概率分数为：", target_label_predict_prob)
    print("对各评论情感预测结果为：", target_label)


def model2(word_vec2, label2, original_len):
    # 将评论和标签进行整合
    comment_vec_final2 = np.array(word_vec2)
    comment_model_use2 = np.hstack(comment_vec_final2, label2)

    # 生成模型入参数据所需参数(设置用于生成词向量的最大评论长度)
    max_reviews_length = 100

    # 数据集切分
    x_train, x_test = train_test_split(np.array(word_vec2), test_size=0.3)
    y_train, y_test = train_test_split(np.array(label2), test_size=0.3)

    x_train = sequence.pad_sequences(x_train, maxlen=max_reviews_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_reviews_length)

    # LSTM 模型构建、训练和评估
    """1.模型构建"""
    embedding_vector_length = 80
    drop_rate1 = 0.3
    drop_rate2 = 0.5
    norm_momentum = 0.99
    recurrent_dropout = 0.5
    activatation_regularizer = 0.01

    model = Sequential()
    model.add(Embedding(input_dim=original_len,
                        output_dim=embedding_vector_length, input_length=30))
    model.add(Dropout(rate=drop_rate1))
    model.add(LSTM(80, use_bias=True, activation="softsign", recurrent_dropout=recurrent_dropout,
                   kernel_initializer="random_uniform", bias_initializer="zeros"))
    model.add(BatchNormalization(axis=-1, momentum=norm_momentum, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                 moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                 beta_constraint=None, gamma_constraint=None))
    model.add(Dropout(drop_rate2))
    model.add(Dense(1, activation='sigmoid',
                    activity_regularizer=regularizers.l2(activatation_regularizer)))

    """2.模型训练"""
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, epochs=10, batch_size=32,
              validation_data=(x_test, y_test), shuffle=True)
    print("模型训练完成！")

    model.save(filepath="D://Resume/si-tech/工作内容/rnn_model_test/rnn_test.h5", overwrite=True,
               include_optimizer="rmsprop")
    print("模型文件保存完成！")

    """3.模型预测评估"""
    scores, acc = model.evaluate(x_test[:100, :], y_test[:100, :], verbose=0)
    print("Accuracy: %.2f%%" % (acc * 100))
    print("Scores: %.2f%%" % (scores * 100))

    # 使用最优模型对评论进行评判预测
    target_label_score = model.predict(x_test[100:, :])
    target_label_predict_prob = model.predict_proba(x_test[100:, ])
    target_label = model.predict_classes(x_test[100:, ])

    print("对各评论预测结果分数为：", target_label_score)
    print("对各评论预测结果概率分数为：", target_label_predict_prob)
    print("对各评论情感预测结果为：", target_label)


if __name__ == '__main__':
    file_name_ = "D://Resume/si-tech/工作内容/txt_analyze8/"

    w2v_all1_, target_len1_ = w2v_from_gensim(file_source_path=file_name_)
    w2v_all2_, target_len2_, original_len_ = w2v_from_cnt(
        file_source_path=file_name_)
    label1_, label2_ = label_gen(file_source_path=file_name_)

    model1(word_vec1=w2v_all1_, label1=label1_, original_len=original_len_)
    model2(word_vec2=w2v_all2_, label2=label2_, original_len=original_len_)
