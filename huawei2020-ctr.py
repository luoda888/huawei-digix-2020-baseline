'''
	Torch == 1.4.0

	CTR
		/inputs
		/models
			/vector
			**model.py
		/submit
'''

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import *
from sklearn.model_selection import StratifiedKFold

from deepctr_torch.models import xDeepFM 
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

import torch
import gc
import os
import json
from joblib import *

import warnings
warnings.filterwarnings("ignore")

def model_feed_dict(df):
    model = {name: df[name] for name in tqdm(feature_name)}
    return model

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
from sklearn.metrics import roc_auc_score

if not os.path.exists("../inputs/train_data.pickle"):
    print("原始数据读入")
    train = pd.read_csv("../inputs/train_data.csv",sep="|")
    test = pd.read_csv("../inputs/test_data_A.csv",sep="|")
    train.to_pickle("../inputs/train_data.pickle")
    test.to_pickle("../inputs/test_data_A.pickle")
    df = pd.concat([train, test],ignore_index=True)
    df.to_pickle("../inputs/all_data.pickle")
else:
    print("缓存数据读入")
    df = pd.read_pickle("../inputs/all_data.pickle")
    df = df[df['pt_d'].isin([6,7,8])]

# 特征工程
'''
	1. 对ID特征进行Count Encoder，Nunique Encoder, Target Encoder
	2. 类别与数值特征的交叉信息
	3. Word2Vec特征
'''

from gensim.models import *

def w2v_id_feature(df, key1, key2, mode='group',
                   embedding_size=64, window_size=20, iter=10, workers=20, min_count=0,
                   func=['mean','std','max'], use_cache=True):
    
    df = df[[key1, key2]]
    if mode == 'group':
        lbl = LabelEncoder()
        try:
            df[key2] = lbl.fit_transform(df[key2])
        except:
            df[key2] = lbl.fit_transform(df[key2].astype(str))
        sentences = df[[key1, key2]].groupby([key1])[key2].apply(list)
    else:
        sentences = df[[key1, key2]].groupby([key1])[key2].apply(lambda x:list(x)[0])
    
    if (os.path.exists("./vector/{}_{}_{}_{}.model".format(key1, key2, embedding_size, window_size))) & (use_cache):
        model = Word2Vec.load("./vector/{}_{}_{}_{}.model".format(key1, key2, embedding_size, window_size))
    else:
        model = Word2Vec(df[[key1, key2]].groupby([key1])[key2].apply(lambda x:[str(i) for i in x]).values.tolist(), 
                         size=embedding_size, window=window_size, 
                         min_count=min_count, sg=1, seed=seed,iter=iter, workers=workers)
        model.save("./vector/{}_{}_{}_{}.model".format(key1, key2, embedding_size, window_size))
    
    embedding = pd.DataFrame()
    embedding[key2] = model.wv.vocab.keys()
    embedding['embedding'] = [model[i] for i in embedding[key2].values]
    embedding[key2] = embedding[key2].astype(int)
    embedding = embedding.sort_values(by=[key2],ascending=True)
    embedding[key2] = lbl.inverse_transform(embedding[key2])
    emb_matrix = np.array([i for i in embedding['embedding'].values])
    emb_mean = []
    for i in tqdm(sentences.values.tolist()):
        emb_mean.append(np.mean(emb_matrix[i], axis=0))
    
    emb_feature = np.asarray(emb_mean)
    mean_col = ['{}(MainKEY)_{}_MEAN_Window{}_{}'.format(key1, key2, window_size, i) for i in range(embedding_size)]
    
    emb_feature = pd.DataFrame(emb_feature, 
                               columns=mean_col)
    
    emb_feature[key1] = sentences.index

    # deal embedding
    embeddings = np.concatenate(embedding['embedding'].values).reshape(-1, embedding_size)
    embeddings = pd.DataFrame(embeddings, columns=["{}_{}(MainKEY)_Window{}_{}".format(key1, key2, window_size, i) for i in range(embedding_size)])
    embedding[embeddings.columns] = embeddings
    del embedding['embedding']

    return emb_feature.reset_index(drop=True), embedding.reset_index(drop=True)

def kfold_stats_feature(train, test, feats, k):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)  

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in tqdm(feats):
        nums_columns = ['label']
        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                tmp_trn = train.iloc[trn_idx]
                order_label = tmp_trn.groupby([feat])[f].mean()
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = train[f].mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            test[colname] = None
            order_label = train.groupby([feat])[f].mean()
            test[colname] = test[feat].map(order_label)
            # fillna
            global_mean = train[f].mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
    del train['fold']
    return train, test


# Part 1

to_count = [['uid'], ['task_id'], ['adv_id'], ['creat_type_cd'], ['adv_prim_id'], 
            ['dev_id'], ['inter_type_cd'], ['slot_id'], ['spread_app_id'], ['tags'], ['app_first_class'],
            ['app_second_class'], ['age'], ['city'], ['city_rank'], ['device_name'], ['device_size'],
            ['career'], ['gender'], ['net_type'], ['residence'], ['his_app_size'], ['his_on_shelf_time'],
            ['app_score'], ['emui_dev'], ['list_time'], ['device_price'], ['up_life_duration'], ['up_membership_grade'],
            ['membership_life_duration'], ['consume_purchase'], ['communication_onlinerate'], ['communication_avgonline_30d'],
            ['indu_name']] 

for i in tqdm(to_count):
    df["{}_count".format("_".join(i))] = df[i].groupby(i)[i].transform('count') 
    # df["{}_rank".format("_".join(i))] = df["{}_count".format("_".join(i))].rank(method='min')

to_group = [
    ['uid','task_id'], ['uid','adv_id'], ['uid','adv_prim_id'], ['uid','dev_id'], ['uid','slot_id'],
    ['uid','spread_app_id'], ['uid','app_first_class'], ['uid','city'], ['uid','device_name'], ['uid', 'net_type'],
    ['uid','communication_onlinerate'], ['uid','list_time']
]

feature = pd.DataFrame()
for i in tqdm(to_group):
    feature["STAT_{}_nunique_1".format("_".join(i))] = df[i].groupby(i[1])[i[0]].transform('nunique')
    feature["STAT_{}_nunique_2".format("_".join(i))] = df[i].groupby(i[0])[i[1]].transform('nunique')
    feature["COUNT-2order_{}".format("_".join(i))] = df[i].groupby(i)[i[0]].transform("count")

# Part 2
to_group = [
        ['task_id'], ['dev_id'], ['adv_prim_id'], ['adv_id'], 
        ['inter_type_cd'], ['slot_id'], ['tags'], ['app_first_class'],
    ]

to_inter = [
    'age',
    'city_rank',
    'career',
    'his_app_size',
    'his_on_shelf_time',
    'app_score',
    'emui_dev',
    'device_price',
    'up_life_duration',
    'communication_avgonline_30d',
]

to_calc = [
    'std',
    'mean',
    'min',
    'max',
    lambda x:np.std(np.fft.fft(x)),
]

for i in tqdm(to_group):
    for j in to_inter:
        for k in to_calc:
            feature["STAT_{}_{}_{}".format("_".join(i),j,k)] = df[i + [j]].groupby(i)[j].transform(k)

choose = df['pt_d']!=8
train, test = df[choose].reset_index(drop=True), df[~choose].reset_index(drop=True)
target_encode_cols = ['uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 
                  	  'dev_id', 'inter_type_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class',]
train, test = kfold_stats_feature(train, test, target_encode_cols, 5)
df = pd.concat([train, test], ignore_index=True)

# Part 3

merge_features = []
embedding_size = 32
tmp = w2v_id_feature(df, 'uid', 'task_id', embedding_size=embedding_size)
merge_features.append(['uid', tmp[0]])
merge_features.append(['task_id', tmp[1]])

tmp = w2v_id_feature(df, 'uid', 'adv_id', embedding_size=embedding_size)
merge_features.append(['uid', tmp[0]])
merge_features.append(['adv_id', tmp[1]])

tmp = w2v_id_feature(df, 'uid', 'slot_id', embedding_size=embedding_size)
merge_features.append(['uid', tmp[0]])
merge_features.append(['slot_id', tmp[1]])

tmp = w2v_id_feature(df, 'uid', 'tags', embedding_size=embedding_size)
merge_features.append(['uid', tmp[0]])
merge_features.append(['tags', tmp[1]])

merges = []
for key,fea in tqdm(merge_features):
    tmp = df[[key]].merge(fea, how='left', on=key)
    merges.append(tmp)

feature.reset_index(drop=True, inplace=True)
df[feature.columns] = feature

for fea in tqdm(merges):
    fea = fea.reset_index(drop=True)
    df[fea.columns] = fea

'''
	数据预处理 & 模型
'''

drop_feature = ['label','id','pt_d']
feature_name = [i for i in df.columns if i not in drop_feature]
print(len(feature_name))

sparse_feature = ['uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 
                  'dev_id', 'inter_type_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class',
                  'app_second_class', 'age', 'city', 'city_rank', 'device_name', 'device_size',
                  'career', 'gender', 'net_type', 'residence', 'his_app_size', 'his_on_shelf_time',
                  'app_score', 'emui_dev', 'list_time', 'device_price', 'up_life_duration', 'up_membership_grade',
                  'membership_life_duration', 'consume_purchase', 'communication_onlinerate', 'communication_avgonline_30d',
                  'indu_name']

dense_feature = [i for i in feature_name if i not in sparse_feature]

for i in tqdm(sparse_feature):
    lbl = LabelEncoder()
    try:
        df[i] = lbl.fit_transform(df[i])
    except:
        continue
        df[i] = lbl.fit_trasnform(df[i].astype('str'))
        
for i in tqdm(dense_feature):
    try:
        df[i] = MinMaxScaler().fit_transform(df[i].fillna(-1).values.reshape(-1,1))
    except:
        feature_name.remove(i)
        dense_feature.remove(i)
        print("Remove", i)

train = df[df['pt_d'].isin([1,2,3,4,5,6])]
valid = df[df['pt_d']==7]
test = df[df['pt_d']==8]

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=8) for feat in sparse_feature] +\
                         [DenseFeat(feat, 1, ) for feat in dense_feature]

dnn_feature_columns = fixlen_feature_columns 
linear_feature_columns = fixlen_feature_columns 

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

X_train = model_feed_dict(train[feature_name])
X_valid = model_feed_dict(valid[feature_name])
X_test = model_feed_dict(test[feature_name])

torch.cuda.empty_cache()

use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'
    
# torch.autograd.set_detect_anomaly(True)

model = xDeepFM(linear_feature_columns, dnn_feature_columns, device=device)
model.compile("adam", 
              'binary_crossentropy',
              ["auc"])
model.fit(X_train, Y, batch_size=4096, epochs=1, 
          validation_data=(X_valid, valid_Y), verbose=1, )

model.fit(X_valid, valid_Y, batch_size=4096)
answer = model.predict(X_test, batch_size=8192)
submit = pd.DataFrame()
submit['id'] = test['id'].astype(int)
submit['probability'] = np.round(answer.flatten(), 6)
submit.to_csv("../submit/xDeepFM-deepctr-baseline.csv",index=False)
