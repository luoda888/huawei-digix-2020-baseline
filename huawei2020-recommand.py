import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import xgboost as xgb
import catboost as cbt
import gc
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"

from sklearn.metrics import mean_squared_error as mse
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

train = pd.read_csv("../inputs/train_dataset.csv",sep="\t",names=['label','query_id','doc_id'] + ["feature_{}".format(i) for i in range(362)])
test = pd.read_csv("../inputs/test_dataset_A.csv",sep='\t',names=['query_id','doc_id'] + ["feature_{}".format(i) for i in range(362)])
df = pd.concat([train, test], ignore_index=True)

feature_name = [i for i in df.columns if 'feature' in i]
drop_col = []
for i in tqdm(feature_name):
    if df[i].std()==0:
        feature_name.remove(i)
print(len(feature_name))

target = 'label'

nfold = 5
kf = KFold(n_splits=nfold, shuffle=True, random_state=2020)

oof = np.zeros((len(train), ))
predictions = np.zeros((len(test), ))
fi = []

ITERATIONS = 100000
EARLY_STOP = 500
VERBOSE = 500

i = 0
for train_index, valid_index in kf.split(train, train[target].astype(int).values):
    print("\nFold {}".format(i + 1))
    X_train, label_train = train.iloc[train_index][feature_name],train.iloc[train_index][target].astype(int).values
    X_valid, label_valid = train.iloc[valid_index][feature_name],train.iloc[valid_index][target].astype(int).values
    
    clf = cbt.CatBoostRegressor(iterations = ITERATIONS, learning_rate = 0.1, depth = 10, 
                                l2_leaf_reg = 10, loss_function = 'RMSE', eval_metric= "RMSE",
                                task_type = 'GPU',devices="0:1",simple_ctr = 'FeatureFreq', combinations_ctr = 'FeatureFreq',)
    clf.fit(X_train, label_train, eval_set = [(X_valid, label_valid)], 
            early_stopping_rounds=EARLY_STOP, verbose=VERBOSE*10)
    x1 = clf.predict(X_valid)
    y1 = clf.predict(test[feature_name])
    
    clf = xgb.XGBRegressor(learning_rate=0.1, max_depth=7, 
                           subsample=0.5, colsample_bytree=0.5, n_estimators=ITERATIONS,
                           eval_metric = 'rmse', tree_method='gpu_hist')
    clf.fit(X_train, label_train, eval_set = [(X_valid, label_valid)], 
            early_stopping_rounds=EARLY_STOP, verbose=VERBOSE)
    x2 = clf.predict(X_valid)
    y2 = clf.predict(test[feature_name])
    
    oof[valid_index] = (x1+x2) / 2#clf.predict(X_valid)
    
    predictions += ((y1+y2)/2) / nfold
    i += 1
    
print(rmse(oof, train[target]))

submit = test[['query_id','doc_id']].reset_index(drop=True)
submit['predict_label'] = predictions 
submit.columns = ['queryid','documentid','predict_label']
submit.to_csv("../submit/baseline.csv",index=False)
