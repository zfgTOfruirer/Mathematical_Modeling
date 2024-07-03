# zfg 1.xlsx 患者列表及临床信息
# zfg 2.xlsx 患者影像信息血肿及水肿的体积及位置
# zfg 3.xlsx 患者影像信息血肿及水肿的形状及灰度分布
# zfg p1.xlsx 流水号对应时间检索表

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
import random
import scipy
import torch
import torch.nn as nn
import warnings
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from seaborn import heatmap
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

table_1 = pd.read_excel(r'./data/1.xlsx').values
table_2 = pd.read_excel(r'./data/2.xlsx').values

table_3_ED = pd.read_excel(r'./data/3.xlsx', sheet_name='ED').values[:, 1:]
table_3_ED_dict = {}  # 以流水号为key，value包含{医学影像图的31个特征}

table_3_Hemo = pd.read_excel(r'./data/3.xlsx', sheet_name='Hemo').values[:, 1:]
table_3_Hemo_dict = {}  # 以流水号为key，value包含{医学影像图的31个特征}

for i in range(table_3_ED.shape[0]):
    table_3_ED_dict[int(table_3_ED[i, 0])] = table_3_ED[i, 1:]

for i in range(table_3_Hemo.shape[0]):
    table_3_Hemo_dict[int(table_3_Hemo[i, 0])] = table_3_Hemo[i, 1:]

table_p1 = pd.read_excel(r'./data/p1.xlsx').values
repeat_times_list = table_p1[:, 1]
order_num_list = []  # 里面的元素为：[流水号，时间，人ID，首次]
for i in range(table_p1.shape[0]):
    # 添加首次
    sample = [table_p1[i, 3], table_p1[i, 2], table_p1[i, 0], 1]
    order_num_list.append(sample)
    for r in range(repeat_times_list[i] - 1):
        sample = [table_p1[i, 5 + r * 2], table_p1[i, 4 + r * 2], table_p1[i, 0], 0]
        order_num_list.append(sample)

table_2_dict = {}
for i in range(table_p1.shape[0]):
    table_2_dict[table_p1[i, 0]] = [table_2[i, 1 + r * 23:1 + (r + 1) * 23] for r in range(repeat_times_list[i])]

order_num_dict = {}  # 以流水号为key，value包含{对应时间，患者ID，是否首次}

for i in range(len(order_num_list)):
    order_num_dict[int(order_num_list[i][0])] = order_num_list[i][1:]

table_p1_dict = {}  # 以患者id作为key，value为该患者的流水号集合
for i in range(table_p1.shape[0]):
    table_p1_dict[table_p1[i, 0]] = [int(table_p1[i, 3 + r * 2]) for r in range(repeat_times_list[i])]

df_ED = pd.DataFrame(table_3_ED)
df_Hemo = pd.DataFrame(table_3_Hemo)
matrix_ED = df_ED.corr().values
heatmap(matrix_ED, cmap='RdBu', vmax=1, vmin=-1)
plt.savefig(r'./result/matrix_ED.png', dpi=1200)
plt.clf()

matrix_Hemo = df_Hemo.corr().values
heatmap(matrix_Hemo, cmap='RdBu', vmax=1, vmin=-1)
plt.savefig(r'./result/matrix_Hemo.png', dpi=1200)
plt.clf()
threshold = 0.7

ED_remove_list = []
for i in range(table_3_ED.shape[1]):
    for j in range(i + 1, table_3_ED.shape[1]):
        if matrix_ED[i, j] > threshold or matrix_ED[i, j] < -threshold:
            ED_remove_list.append(i)
ED_remove_list = np.array(list(set(ED_remove_list)))

Hemo_remove_list = []
for i in range(table_3_Hemo.shape[1]):
    for j in range(i + 1, table_3_Hemo.shape[1]):
        if matrix_Hemo[i, j] > threshold or matrix_Hemo[i, j] < -threshold:
            Hemo_remove_list.append(i)
Hemo_remove_list = np.array(list(set(Hemo_remove_list)))

table_2_all = []
for p in range(table_2.shape[0]):
    person_id = 'sub{:03}'.format(p + 1)
    table_2_all.extend(table_2_dict[person_id])
table_2_all = np.array(table_2_all, dtype=float)[:, 1:]
df_table2 = pd.DataFrame(table_2_all)
matrix_table2 = df_table2.corr().values
heatmap(matrix_table2, cmap='RdBu', vmax=1, vmin=-1)
plt.savefig(r'./result/matrix_table2.png', dpi=1200)
plt.clf()

table2_remove_list = []
for i in range(table_2_all.shape[1]):
    for j in range(i + 1, table_2_all.shape[1]):
        if matrix_table2[i, j] > threshold or matrix_table2[i, j] < -threshold:
            table2_remove_list.append(i)
table2_remove_list = np.array(list(set(table2_remove_list)))
print()


# zfg table_1 为表1
# zfg table_2 为表2
# zfg table_3 为表3
# zfg table_p1_dict 以患者id作为key，value为该患者的流水号集合
# zfg table_2_dict 以患者id作为key，value为该患者的水肿的体积及位置（22个特征）的多次记录
# zfg table_3_dict 以流水号为key，value为该影像的血肿和水肿的形状及灰度分布（31个特征）


def Q_1a():
    person_idx = []
    two_day = 48 * 60 * 60 * 1e9
    time_gap_all = []
    for p in range(100):
        # 该患者包含的影像记录个数为r
        # 根据p获取患者的影像记录编号，即流水号
        person_id = 'sub{:03}'.format(p + 1)
        person_order_list = table_p1_dict[person_id]
        start_time = table_1[p, 14]  # 小时
        for i in range(1, repeat_times_list[p]):
            # 根据流水号查找MH_volume值和时间点
            # 计算随访记录与首次记录的差值、相对值、时间差
            MH_increase_absolutely = table_2_dict[person_id][i][1] - table_2_dict[person_id][0][1]
            MH_increase_relatively = MH_increase_absolutely / table_2_dict[person_id][0][1]
            time_gap = order_num_dict[person_order_list[i]][0].value - order_num_dict[person_order_list[0]][0].value
            time_gap = time_gap + start_time * 60 * 60 * 1e9
            if (MH_increase_absolutely >= 6000 or MH_increase_relatively >= 0.33) and time_gap <= two_day:
                person_idx.append(p)
                time_gap_all.append(time_gap)
                break
    time_gap_all = np.array(time_gap_all) / (1e9 * 60 * 60)  # 单位小时
    return np.array(person_idx), time_gap_all


def Q_1b():
    feature_1_2 = np.concatenate([table_1[:, 4:], table_2[:, 2:24]], axis=1)
    # 根据个人属性与是否扩展的卡方分析结果，
    # 删除性别、高血压病史、糖尿病史3个特征，即这3个特征与是否发生扩张关系不大
    feature_1_2 = np.delete(feature_1_2, [1, 3, 5], axis=1)
    # 删除相关性高的样本，table2
    feature_1_2 = np.delete(feature_1_2, table2_remove_list + 16, axis=1)
    first_order_id = table_1[:, 3]
    feature_3 = []
    for i in range(len(feature_1_2)):
        order_id = first_order_id[i]
        feature_3.append(table_3_Hemo_dict[order_id])

    feature_3 = np.array(feature_3)
    feature_3 = np.delete(feature_3, Hemo_remove_list, axis=1)  # 10维
    feature = np.concatenate([feature_1_2, feature_3], axis=1)
    blood_press = feature[:, 8]
    blood_press_h = []
    blood_press_l = []
    for b in blood_press:
        temp_h, temp_l = str(b).split('/')
        blood_press_h.append(temp_h)
        blood_press_l.append(temp_l)
    blood_press = np.concatenate([np.array(blood_press_h).reshape(-1, 1), np.array(blood_press_l).reshape(-1, 1)],
                                 axis=1)
    feature = np.delete(feature, 8, 1)
    feature = np.concatenate([blood_press, feature], axis=1)
    feature = feature.astype(np.float64)
    feature = minmax_scale(feature)
    train_y = np.zeros(100)
    temp, _ = Q1a()
    for i in temp:
        train_y[i] = 1
    train_y = train_y.reshape(-1, 1)

    feature1 = feature[:, :17]
    train_x1 = feature1[:100, :]
    model = DecisionTreeClassifier()
    model.fit(train_x1, train_y)
    most_important_feature1 = np.argsort(model.feature_importances_)[-feature1.shape[1] // 2:]
    print('most_important_feature1:', most_important_feature1)

    feature2 = feature[:, 17:-feature_3.shape[1]]
    train_x2 = feature2[:100, :]
    model = DecisionTreeClassifier()
    model.fit(train_x2, train_y)
    most_important_feature2 = np.argsort(model.feature_importances_)[-feature2.shape[1] // 2:]
    print('most_important_feature2:', most_important_feature2)

    feature3 = feature[:, -feature_3.shape[1]:]
    train_x3 = feature3[:100, :]
    model = DecisionTreeClassifier()
    model.fit(train_x3, train_y)
    most_important_feature3 = np.argsort(model.feature_importances_)[-feature3.shape[1] // 2:]
    print('most_important_feature3:', most_important_feature3)

    filter_feature = np.concatenate([feature1[:, most_important_feature1], feature2[:, most_important_feature2],
                                     feature3[:, most_important_feature3]], axis=1)
    test_x = filter_feature
    # model = SVC(probability=True)
    # sampler = SMOTE(k_neighbors=7, sampling_strategy={0: 80, 1: 80})
    sampler = RandomUnderSampler()
    new_train_x, new_train_y = sampler.fit_resample(filter_feature[:100, :], train_y)
    sampler = SMOTE(k_neighbors=7, sampling_strategy={0: 200, 1: 200})
    new_train_x, new_train_y = sampler.fit_resample(new_train_x, new_train_y)

    # 交叉验证、多模型对比
    models = [DecisionTreeClassifier(), SVC(probability=True), RandomForestClassifier(), GaussianNB()]
    models_name = ['DT', 'SVM', 'RF', 'NB']
    cross_result = {}
    for name in models_name:
        cross_result[name] = {}
    # 交叉验证
    cv = 5
    for model, name in zip(models, models_name):
        temp = cross_val_score(model, new_train_x, new_train_y, scoring='accuracy', cv=cv)
        cross_result[name]['accuracy'] = temp
        temp = cross_val_score(model, new_train_x, new_train_y, scoring='average_precision', cv=cv)
        cross_result[name]['precision'] = temp
        temp = cross_val_score(model, new_train_x, new_train_y, scoring='recall', cv=cv)
        cross_result[name]['recall'] = temp
        temp = cross_val_score(model, new_train_x, new_train_y, scoring='f1', cv=cv)
        cross_result[name]['f1'] = temp
        temp = cross_val_score(model, new_train_x, new_train_y, scoring='roc_auc', cv=cv)
        cross_result[name]['auc'] = temp
    means = [np.mean(cross_result[name]['f1']) for name in models_name]
    best_model_idx = np.argsort(means)[-1]
    best_model = models[best_model_idx]
    best_model.fit(new_train_x, new_train_y)
    pre_proba = best_model.predict_proba(test_x)
    print(np.sum(np.argmax(pre_proba, -1)[100:]))
    return cross_result, {'best model': models_name[best_model_idx]}, {'predict_probability': pre_proba}


def fit_func(x, a, b):
    return a + b * np.exp(-1 / x)


def Q_2a():
    # 对每个人：
    # 根据流水号查询具体时间、具体水肿体积
    # 分别计算时间差、体积差
    # 拟合时间差、体积差
    all_ED_increase = []
    all_time_increase = []
    all_id = []
    for p in range(100):
        person_id = 'sub{:03}'.format(p + 1)
        person_order_list = table_p1_dict[person_id]
        time_start = float(table_1[p][14])
        for i in range(1, repeat_times_list[p]):
            ED_increase = table_2_dict[person_id][i][12]
            ED_increase = ED_increase / 1000
            time_increase = order_num_dict[person_order_list[i]][0].value - order_num_dict[person_order_list[0]][
                0].value
            time_increase = time_increase / (1e9 * 60 * 60)
            time_increase = time_start + time_increase
            all_ED_increase.append(ED_increase)
            all_time_increase.append(time_increase)
            all_id.append(p)

    all_ED_increase = np.array(all_ED_increase)
    all_time_increase = np.array(all_time_increase)
    all_id = np.array(all_id)

    z = np.log(all_ED_increase + 1)

    mask1 = all_time_increase > 2000
    mask2 = z > 2
    mask = mask1 & mask2
    mask = np.array(1 - mask, dtype=bool)
    all_ED_increase = all_ED_increase[mask]
    all_time_increase = all_time_increase[mask]
    z = z[mask]
    all_id = all_id[mask]

    np.savetxt('./result/time.txt', all_time_increase)
    np.savetxt('./result/log_ed.txt', z)

    plt.scatter(all_time_increase, z)
    # plt.show()

    para = scipy.optimize.curve_fit(fit_func, all_time_increase, z, maxfev=10000)[0]
    plt.plot(np.linspace(np.min(all_time_increase), np.max(all_time_increase), 1000),
             [fit_func(x, *para) for x in np.linspace(np.min(all_time_increase), np.max(all_time_increase), 1000)],
             c='r')

    plt.savefig(r'./result/2a.png', dpi=800)
    plt.clf()

    pre_v = [fit_func(x, *para) for x in all_time_increase]

    pre_v = np.array(pre_v)
    pre_v = np.exp(pre_v) - 1
    residual = pre_v - all_ED_increase
    mean_residual = []
    for i in range(len(set(all_id))):
        temp = residual[all_id == i]
        mean_residual.append(np.mean(np.abs(temp)))
    np.savetxt(r'./result/2a_mean_abs_residual.txt', mean_residual)
    return mean_residual


def Q_2b(n=3):
    # 个体属性
    personal_feature = np.concatenate([table_1[:100, 4:13], table_1[:100, 15].reshape(-1, 1)], axis=1)
    for i in range(personal_feature.shape[0]):
        if personal_feature[i, 1] == '男':
            personal_feature[i, 1] = 1
        else:
            personal_feature[i, 1] = 0
    blood_press = personal_feature[:, -1]
    blood_press_h = []
    blood_press_l = []
    for b in blood_press:
        temp_h, temp_l = str(b).split('/')
        blood_press_h.append(temp_h)
        blood_press_l.append(temp_l)
    blood_press = np.concatenate([np.array(blood_press_h).reshape(-1, 1), np.array(blood_press_l).reshape(-1, 1)],
                                 axis=1)
    personal_feature = np.delete(personal_feature, -1, 1)
    personal_feature = np.concatenate([blood_press, personal_feature], axis=1)

    # 先聚类：
    cluster = KMeans(n_clusters=n)
    cluster.fit(personal_feature)
    cluster_labels = cluster.predict(personal_feature)
    colors = ['r', 'b', 'y', 'k', 'g']

    # TSNE
    tsne = TSNE()
    x_new = tsne.fit_transform(personal_feature)
    for i, (x, y) in enumerate(x_new):
        plt.scatter(x, y, c=colors[int(cluster_labels[i])])
    plt.title(f'Cluster Result Based on N={n}')
    # plt.show()
    plt.savefig(rf'./result/2b_tsne_based{n}.png', dpi=800)
    plt.clf()
    id_recode = []
    mean_residual = []

    for j in range(n):
        person_ids = table_1[:100, 0][cluster_labels == j]
        all_ED_increase = []
        all_time_increase = []
        all_id = []
        for person_id in person_ids:
            p = int(person_id[-3:]) - 1
            person_order_list = table_p1_dict[person_id]
            time_start = float(table_1[p][14])
            for i in range(1, repeat_times_list[p]):
                ED_increase = table_2_dict[person_id][i][12]
                ED_increase = ED_increase / 1000
                time_increase = order_num_dict[person_order_list[i]][0].value - order_num_dict[person_order_list[0]][
                    0].value
                time_increase = time_increase / (1e9 * 60 * 60)
                time_increase = time_start + time_increase
                all_ED_increase.append(ED_increase)
                all_time_increase.append(time_increase)
                all_id.append(p)

        all_ED_increase = np.array(all_ED_increase)
        all_time_increase = np.array(all_time_increase)
        all_id = np.array(all_id)
        z = np.log(all_ED_increase + 1)

        mask1 = all_time_increase > 2000
        mask2 = z > 2
        mask = mask1 & mask2
        mask = np.array(1 - mask, dtype=bool)
        all_ED_increase = all_ED_increase[mask]
        all_time_increase = all_time_increase[mask]
        z = z[mask]
        all_id = all_id[mask]

        np.savetxt(f'./result/base{n}_cluster{j + 1}_time.txt', all_time_increase)
        np.savetxt(f'./result/base{n}_cluster{j + 1}_log_ed.txt', z)

        plt.scatter(all_time_increase, z)

        para = scipy.optimize.curve_fit(fit_func, all_time_increase, z, maxfev=10000)[0]
        plt.plot(np.linspace(np.min(all_time_increase), np.max(all_time_increase), 1000),
                 [fit_func(x, *para) for x in np.linspace(np.min(all_time_increase), np.max(all_time_increase), 1000)],
                 c='r')

        plt.savefig(rf'./result/2b_based{n}_cluster{j + 1}.png', dpi=800)
        plt.clf()

        # 计算残差
        pre_v = [fit_func(x, *para) for x in all_time_increase]
        pre_v = np.array(pre_v)
        pre_v = np.exp(pre_v) - 1
        residual = pre_v - all_ED_increase

        for i in set(all_id):
            temp = residual[all_id == i]
            mean_residual.append(np.mean(np.abs(temp)))
            id_recode.append(i)
    id_recode = np.array(id_recode)
    mean_residual = np.array(mean_residual)
    residual_list = np.concatenate([id_recode.reshape(-1, 1), mean_residual.reshape(-1, 1)], axis=1)
    residual_list = residual_list[np.argsort(residual_list[:, 0]), :]
    np.savetxt(r'./result/2b_mean_abs_residual.txt',
               np.concatenate([residual_list, cluster_labels.reshape(-1, 1)], axis=1))
    return residual_list[:, 1], cluster_labels


def Q_2c(data_type='ED'):
    if data_type == 'ED':
        column_idx = 12
    elif data_type == 'Hemo':
        column_idx = 1
    mean_residual = []
    for n in [3, 4, 5]:
        residual, labels = Q2b(n)
        t = [np.mean(np.abs(residual[labels == i])) for i in range(n)]
        mean_residual.append(t)
    best_cluster_num = np.argmin([np.mean(mean_residual[i]) for i in range(3)]) + 3
    print('Best Cluster N:', best_cluster_num)
    residual, labels = Q2b(best_cluster_num)
    # 将患者分为四类
    # 对每类：
    # 取出患者对应的疗法以及水肿进展模型（速率）
    # 做特征重要性排序
    threaten_method = table_1[:100, 16:]
    increase_time_list = []
    increase_volume_list = []
    for p in range(100):
        person_id = 'sub{:03}'.format(p + 1)
        person_order_list = table_p1_dict[person_id]
        increase_volume = table_2_dict[person_id][1][column_idx] - table_2_dict[person_id][0][column_idx]
        increase_volume = increase_volume / 1000
        increase_time = order_num_dict[person_order_list[1]][0].value - order_num_dict[person_order_list[0]][0].value
        increase_time = increase_time / (1e9 * 60 * 60)

        increase_time_list.append(increase_time)
        increase_volume_list.append(increase_volume)
    increase_volume_list = np.array(increase_volume_list)
    increase_time_list = np.array(increase_time_list)

    # 用水肿变化速率来衡量不同的疗法的作用
    increase_rate_list = increase_volume_list / increase_time_list

    # 进过观察，患者编号为15和46的第一次随访与首次检查的时间跨度过大，因此删除两人的数据

    increase_rate_list = np.delete(increase_rate_list, [14, 45])
    np.savetxt(rf'./result/{data_type}_inrcease_rate.txt', increase_rate_list.tolist())
    threaten_method = np.delete(threaten_method, [14, 45], axis=0)
    all_personal = np.delete(table_1[:100, 0], [14, 45], axis=0)

    labels = np.delete(labels, [14, 45], axis=0)
    models = [SVR(kernel='linear'), DecisionTreeRegressor(), RandomForestRegressor(), LinearRegression()]
    models_name = ['SVM', 'DT', 'RF', 'LR']
    feature_name = np.array(['脑室引流', '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经'])
    smote = SMOTE(k_neighbors=4, sampling_strategy={0: 400, 1: 400})
    final_result = {}
    for i in range(best_cluster_num):
        final_result[f'cluster_{i}'] = {}
        final_result[f'cluster_{i}']['personal_id'] = all_personal[labels == i]
        # 对于每类患者
        increase_rate_cluster = increase_rate_list[labels == i]
        threaten_method_cluster = threaten_method[labels == i]

        cross_result = {}
        for model, name in zip(models, models_name):
            temp = cross_val_score(model, threaten_method_cluster, increase_rate_cluster, scoring='r2')
            cross_result[name] = temp
        final_result[f'cluster_{i}']['cross_validate_result'] = cross_result
        means = [np.mean(cross_result[name]) for name in models_name]
        best_model_idx = np.argsort(means)[-1]
        final_result[f'cluster_{i}']['best_model'] = models_name[best_model_idx]
        best_model = models[best_model_idx]
        best_model.fit(threaten_method_cluster, increase_rate_cluster)
        # 取最好的模型的特征重要性排序
        try:
            t = best_model.feature_importances_
        except:
            t = np.abs(best_model.coef_)
        feature_rank = feature_name[np.argsort(t)][::-1]
        final_result[f'cluster_{i}']['feature_rank'] = feature_rank

    return final_result


def Q_2d():
    Hemo_volume_list = []
    ED_volume_list = []
    for p in range(100):
        person_id = 'sub{:03}'.format(p + 1)
        person_order_list = table_p1_dict[person_id]
        for n in range(repeat_times_list[p]):
            # 根据流水号查询ED和Hemo值
            Hemo_volume_list.append(table_2_dict[person_id][n][1])
            ED_volume_list.append(table_2_dict[person_id][n][12])
    Hemo_volume_list = minmax_scale(Hemo_volume_list)
    ED_volume_list = minmax_scale(ED_volume_list)
    plt.plot(Hemo_volume_list)
    plt.plot(ED_volume_list)
    plt.savefig(r'./result/2d_Hemo_ED.png', dpi=1200)
    plt.clf()
    spearmanr = stats.spearmanr(Hemo_volume_list, ED_volume_list)
    return spearmanr


def Q_3a3b3c():
    all_person_label = table_1[:100, 1]
    all_person_feature1 = table_1[:, 4:]
    for i in range(all_person_feature1.shape[0]):
        if all_person_feature1[i, 1] == '男':
            all_person_feature1[i, 1] = 1
        else:
            all_person_feature1[i, 1] = 0

    blood_press = all_person_feature1[:, 11]
    blood_press_h = []
    blood_press_l = []
    for b in blood_press:
        temp_h, temp_l = str(b).split('/')
        blood_press_h.append(temp_h)
        blood_press_l.append(temp_l)
    blood_press = np.concatenate([np.array(blood_press_h).reshape(-1, 1), np.array(blood_press_l).reshape(-1, 1)],
                                 axis=1)
    all_person_feature1 = np.delete(all_person_feature1, 11, 1)
    all_person_feature1 = np.concatenate([blood_press, all_person_feature1], axis=1)
    all_person_feature2 = table_2[:, 2:24]
    all_person_feature2 = np.delete(all_person_feature2, table2_remove_list, axis=1)  # 根据热力图和阈值删除特征
    all_person_feature3 = []
    for p in range(160):
        person_first_order = table_1[p, 3]
        temp = table_3_Hemo_dict[person_first_order]
        all_person_feature3.append(temp)
    all_person_label = np.array(all_person_label, dtype=int)
    all_person_feature1 = np.array(all_person_feature1, dtype=float)
    all_person_feature2 = np.array(all_person_feature2, dtype=float)
    all_person_feature3 = np.array(all_person_feature3, dtype=float)
    all_person_feature3 = np.delete(all_person_feature3, Hemo_remove_list, axis=1)  # 根据热力图和阈值删除特征
    feature = np.concatenate([all_person_feature1, all_person_feature2, all_person_feature3], axis=1)
    feature = minmax_scale(feature)

    model = LogisticRegression(C=20)
    model.fit(feature[:100], all_person_label)
    t = model.predict(feature)

    # 随访记录作为无标签数据
    # 先获取所有随访记录
    unlabeled_feature = []
    for p in range(160):
        person_id = 'sub{:03}'.format(p + 1)
        for n in range(1, repeat_times_list[p]):
            try:
                temp_feature1 = table_1[p, 4:]
                temp_feature2 = table_2_dict[person_id][n][1:]
                temp_feature3 = table_3_Hemo_dict[table_p1_dict[person_id][n]]
                temp_feature = np.concatenate([temp_feature1, temp_feature2, temp_feature3])
                unlabeled_feature.append(temp_feature)
            except:
                continue
    unlabeled_feature = np.array(unlabeled_feature)
    unlabeled_feature1 = unlabeled_feature[:, :19]
    unlabeled_feature2 = unlabeled_feature[:, 19:41]
    unlabeled_feature3 = unlabeled_feature[:, 41:]
    # 对unlabeled_feature1处理：
    for i in range(unlabeled_feature1.shape[0]):
        if unlabeled_feature1[i, 1] == '男':
            unlabeled_feature1[i, 1] = 1
        else:
            unlabeled_feature1[i, 1] = 0

    blood_press = unlabeled_feature1[:, 11]
    blood_press_h = []
    blood_press_l = []
    for b in blood_press:
        temp_h, temp_l = str(b).split('/')
        blood_press_h.append(temp_h)
        blood_press_l.append(temp_l)
    blood_press = np.concatenate([np.array(blood_press_h).reshape(-1, 1), np.array(blood_press_l).reshape(-1, 1)],
                                 axis=1)
    unlabeled_feature1 = np.delete(unlabeled_feature1, 11, 1)
    unlabeled_feature1 = np.concatenate([blood_press, unlabeled_feature1], axis=1)
    # 对unlabeled_feature2处理：
    unlabeled_feature2 = np.delete(unlabeled_feature2, table2_remove_list, axis=1)  # 根据热力图和阈值删除特征
    # 对unlabeled_feature3处理：
    unlabeled_feature3 = np.delete(unlabeled_feature3, Hemo_remove_list, axis=1)  # 根据热力图和阈值删除特征

    unlabeled_feature1 = np.array(unlabeled_feature1, dtype=float)
    unlabeled_feature2 = np.array(unlabeled_feature2, dtype=float)
    unlabeled_feature3 = np.array(unlabeled_feature3, dtype=float)
    unlabeled_feature = np.concatenate([unlabeled_feature1, unlabeled_feature2, unlabeled_feature3], axis=1)
    unlabeled_feature = unlabeled_feature[~np.isnan(unlabeled_feature).any(axis=1)]
    unlabeled_feature = minmax_scale(unlabeled_feature)
    # 使用刚刚基于首次训练得到的LR给没有标签的回访记录做预测，输出概率
    confidence_threshold = 0.8
    pre_proba = model.predict_proba(unlabeled_feature)
    pre_y = model.predict(unlabeled_feature)
    # 概率比预先设置的阈值高则认为是高置信度的伪标签
    mask = pre_proba > confidence_threshold
    mask = np.sum(mask, axis=1)
    mask = np.array(mask, dtype=bool)
    # 将高置信度的伪标签和对应的数据插入训练集，重新训练LR
    new_x = unlabeled_feature[mask]
    new_y = pre_y[mask]
    new_x = np.concatenate([feature[:100, :], new_x], )
    new_y = np.concatenate([all_person_label, new_y])
    # 最后用重新训练后的LR，预测一共160个人的mRS
    model.fit(new_x, new_y)
    final_pre_160 = model.predict(feature)

    final_pre_all = model.predict(unlabeled_feature)

    # Q3c
    # 得到所有，即首次、随访的记录，所对应的mRS值
    # 扩充了数据集，因此更好的用于分析第三题
    all_feature = np.concatenate([feature, unlabeled_feature])
    all_label = np.concatenate([all_person_label, model.predict(feature[100:]), final_pre_all])
    all_data = np.concatenate([all_feature, all_label.reshape(-1, 1)], axis=1)

    # 现在一共540条样本-标签，因此可以使用深度网络
    train_size = 440

    class MyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(all_feature.shape[1], 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 7),
                nn.Softmax(-1)
            )

        def forward(self, x):
            return self.net(x)

    class MyDataset(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __getitem__(self, item):
            return self.x[item], self.y[item]

        def __len__(self):
            return len(self.y)

    batch = 10
    epoch = 800
    net = MyNet()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    loss_his = []
    validate_acc = []
    for e in range(epoch):
        np.random.shuffle(all_data)
        train_feature = all_data[:train_size, :-1]
        train_label = all_data[:train_size, -1]
        validate_feature = all_data[train_size:, :-1]
        validate_label = all_data[train_size:, -1]
        dataset = MyDataset(train_feature, train_label)
        loader = DataLoader(dataset, batch, shuffle=True)
        temp_loss = []
        for x, y in loader:
            opt.zero_grad()
            x = x.to(torch.float)
            y = nn.functional.one_hot(y.to(torch.long), 7).to(torch.float)
            out = net(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            opt.step()
            temp_loss.append(loss.item())
        loss_his.append(np.mean(temp_loss))
        val_pre = torch.argmax(net(torch.FloatTensor(validate_feature)), -1).tolist()
        temp_acc = accuracy_score(val_pre, validate_label)
        validate_acc.append(temp_acc)
        print(f'Epoch {e}/{epoch}, Validate Acc {validate_acc[-1]}, Loss {loss_his[-1]}')
    plt.plot(loss_his)
    plt.savefig('./result/3c_loss.png', dpi=800)
    plt.clf()
    plt.plot(validate_acc)
    plt.savefig('./result/3c_val_acc.png', dpi=800)
    print(net)
    print('Saving Weight of First Layer')
    np.savetxt('./result/first_layer_weight.txt', net.state_dict()['net.0.weight'].tolist())
    feature_weight = np.loadtxt('./result/first_layer_weight.txt')
    feature_importance = np.sum(np.abs(feature_weight), axis=0)
    feature_importance = feature_importance / np.sum(feature_importance)

    return t, final_pre_160, np.argsort(feature_importance)[::-1]


# print('Q_1a, Person index with label 1, and gap time:\n')
# pprint.pprint(Q_1a())

# print('Q_1b, Cross validate result, best model and predict probability:\n')
# pprint.pprint(Q1b())

# print('Q_2a, residual result:')
# pprint.pprint(Q2a())

# for n in [3, 4, 5]:
#     print(f'Q_2b, Cluster number {n} result:')
#     pprint.pprint(Q_2b(n))
#
# # Q_2b(4)

# print('Q_2c, Threaten method to ED:')
# pprint.pprint(Q_2c('ED'))

# print('Q_2d, Threaten method to Hemo:')
# pprint.pprint(Q2c('Hemo'))
#
# print('Q_2d, Spearmanr of ED and Hemom:', Q_2d())

pprint.pprint(Q_3a3b3c())
