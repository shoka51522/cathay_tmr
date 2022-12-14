import optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import gc
from datetime import datetime

import pickle
import configure
import data_processing as dp
CONF = configure.Configure()

from optuna.integration import LightGBMPruningCallback
from optuna.samplers import TPESampler
from optuna.samplers import RandomSampler

CONF = configure.Configure()
with open("col_list.pkl", "rb") as file:
    col_list = pickle.load(file)

################################################################################################################## model 1
df_tr = pd.read_csv(f'data/model1/FINAL_TABLE.csv', encoding='big5')
df_tr = df_tr.rename(columns = {'專案成立時間' : 'market_time', 'NEW_CONTACT_Y': 'CONTACT_Y', 'CONTACT_Y': 'old_CONTACT_Y'})

df_tr = df_tr[col_list]

##################################################################################################################
df_tr = df_tr[df_tr['DEL_ID'] == 'KEEP']
need_cols = [col for col in df_tr.columns if col not in ['自變數取樣時間', 'old_CONTACT_Y', 'MAX_SEC', 'DEL_ID']]
df_tr = df_tr[need_cols]

df_talk = pd.read_csv(f'data/model1/TMR_SELF_RECORD.csv', encoding='big5')
df_talk = df_talk.rename(columns = {'專案成立時間' : 'market_time'})

with open("endcodelist_sale_model1.pkl", "rb") as file:
    endcodelist = pickle.load(file)
with open("level_score_sale_model1.pkl", "rb") as file:
    endcode_ratio_level = pickle.load(file)
with open("score_range_1_sale_model1.pkl", "rb") as file:
    endcode_ratio = pickle.load(file)

df_talk['END_CODE'] = df_talk.apply(lambda end_code: dp.new_end_code(end_code), axis = 1)

df_talk_stat = dp.create_vars_records(df_talk, endcodelist, endcode_ratio, endcode_ratio_level) 

# bind into ABT
df_tr = df_tr.merge(df_talk_stat, on=['ID_SAS', 'market_time'], how='left')
del df_talk, df_talk_stat

df_tr = dp.feature_engineering(df_tr) 
cols_cate = [col for col in list(df_tr.columns) if (df_tr[col].dtype == 'object') & (col != 'ID_SAS')] 
dict_factorize = dict()
for col in cols_cate:
    dict_col = dp.factorize_categoricals(df_tr, col)
    dict_factorize[col] = dict_col
    df_tr[col].replace(dict_col, inplace=True)
df_tr = df_tr.drop(['CONTACT_Y'], axis = 1)
df_tr['Y'].fillna(value = 0, inplace = True)
##################################################################################################################

print('data preparation: i love big booooobs')
features = [col for col in df_tr.columns if col not in ['ID_SAS', CONF.training_y, 'market_time']]

def objective(trial, fold = 5):
    params = {
        "objective" : "binary",
        "metric" : "auc",
        "boosting": 'gbdt',
        "verbosity": -1,
        #"max_depth" : -1,
        "seed": 5271,
        "n_estimators": trial.suggest_int("n_estimators", 1000,120000), 
        "learning_rate": trial.suggest_float("learning_rate", 0.00005, 0.015), #0.0002
        "num_leaves": trial.suggest_int("num_leaves", 2, 1000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.5, step=0.1
        ),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 50),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.1, 0.95, step=0.05
        ),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 500),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "boost_from_average": "false",
    }


    oof = df_tr[['ID_SAS', 'Y']].copy()
    oof['predict'] = 0
    model = [0]*fold

    cv = StratifiedKFold(n_splits = fold, shuffle=True, random_state=801217)

    cv_scores = [0]
    gc.collect()

    for fold, (trn_idx, val_idx) in enumerate(cv.split(df_tr, df_tr['Y'])):
        
        X_train, y_train = df_tr.iloc[trn_idx][features], df_tr.iloc[trn_idx]['Y']   
        X_valid, y_valid = df_tr.iloc[val_idx][features], df_tr.iloc[val_idx]['Y']
        
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        # evals_result = {}
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")

        print(f'fold: {fold}')
        model[fold] = lgb.train(
            params,
            trn_data,
            valid_sets=[val_data],
            verbose_eval = False,
            early_stopping_rounds= 3000,
            callbacks=[
                pruning_callback
            ],  # Add a pruning callback
            )
        
        

        oof['predict'].iloc[val_idx] = model[fold].predict(X_valid)
    cv_scores = roc_auc_score(oof['Y'], oof['predict'])
    print(f'auc:{cv_scores}')
    gc.collect()

    return cv_scores

time1 = datetime.now()
print(f'start training: {time1}')

study = optuna.create_study(direction="maximize", study_name="into pussy", sampler=TPESampler(seed=992330))
def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
study.optimize(objective, n_trials=87, n_jobs=2, callbacks=[print_best_callback, lambda study, trial: gc.collect()])

print(f'training complete: {datetime.now()}')
print(f'training time: {datetime.now() - time1}')

print(f"\tBest value (auc): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")

joblib.dump(study, "optuna tune/study_model1.pkl")

del df_tr, features, study, time1, objective
gc.collect()

################################################################################################################## model 4
df_tr = pd.read_csv(f'data/model4/FINAL_TABLE.csv', encoding='big5')
df_tr = df_tr.rename(columns = {'專案成立時間' : 'market_time', 'NEW_CONTACT_Y': 'CONTACT_Y', 'CONTACT_Y': 'old_CONTACT_Y'})

df_tr = df_tr[col_list]

##################################################################################################################
df_tr = df_tr[df_tr['DEL_ID'] == 'KEEP']
need_cols = [col for col in df_tr.columns if col not in ['自變數取樣時間', 'old_CONTACT_Y', 'MAX_SEC', 'DEL_ID']]
df_tr = df_tr[need_cols]

df_talk = pd.read_csv(f'data/model4/TMR_SELF_RECORD.csv', encoding='big5')
df_talk = df_talk.rename(columns = {'專案成立時間' : 'market_time'})

with open("endcodelist_sale_model4.pkl", "rb") as file:
    endcodelist = pickle.load(file)
with open("level_score_sale_model4.pkl", "rb") as file:
    endcode_ratio_level = pickle.load(file)
with open("score_range_1_sale_model4.pkl", "rb") as file:
    endcode_ratio = pickle.load(file)

df_talk['END_CODE'] = df_talk.apply(lambda end_code: dp.new_end_code(end_code), axis = 1)

df_talk_stat = dp.create_vars_records(df_talk, endcodelist, endcode_ratio, endcode_ratio_level) 

# bind into ABT
df_tr = df_tr.merge(df_talk_stat, on=['ID_SAS', 'market_time'], how='left')
del df_talk, df_talk_stat

df_tr = dp.feature_engineering(df_tr) 
cols_cate = [col for col in list(df_tr.columns) if (df_tr[col].dtype == 'object') & (col != 'ID_SAS')] 
dict_factorize = dict()
for col in cols_cate:
    dict_col = dp.factorize_categoricals(df_tr, col)
    dict_factorize[col] = dict_col
    df_tr[col].replace(dict_col, inplace=True)
df_tr = df_tr.drop(['CONTACT_Y'], axis = 1)
df_tr['Y'].fillna(value = 0, inplace = True)
##################################################################################################################

print('data preparation: i love big pussyyyyyyyyyyy')
features = [col for col in df_tr.columns if col not in ['ID_SAS', CONF.training_y, 'market_time']]

def objective2(trial, fold = 5):
    params = {
        "objective" : "binary",
        "metric" : "auc",
        "boosting": 'gbdt',
        "verbosity": -1,
        #"max_depth" : -1,
        "seed": 5271,
        "n_estimators": trial.suggest_int("n_estimators", 1000,120000), 
        "learning_rate": trial.suggest_float("learning_rate", 0.00005, 0.015), #0.0002
        "num_leaves": trial.suggest_int("num_leaves", 2, 1000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.5, step=0.1
        ),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 50),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.1, 0.95, step=0.05
        ),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 500),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "boost_from_average": "false",
    }


    oof = df_tr[['ID_SAS', 'Y']].copy()
    oof['predict'] = 0
    model = [0]*fold

    cv = StratifiedKFold(n_splits = fold, shuffle=True, random_state=801217)

    cv_scores = [0]
    gc.collect()

    for fold, (trn_idx, val_idx) in enumerate(cv.split(df_tr, df_tr['Y'])):
        
        X_train, y_train = df_tr.iloc[trn_idx][features], df_tr.iloc[trn_idx]['Y']   
        X_valid, y_valid = df_tr.iloc[val_idx][features], df_tr.iloc[val_idx]['Y']
        
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        # evals_result = {}
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")

        print(f'fold: {fold}')
        model[fold] = lgb.train(
            params,
            trn_data,
            valid_sets=[val_data],
            verbose_eval = False,
            early_stopping_rounds= 3000,
            callbacks=[
                pruning_callback
            ],  # Add a pruning callback
            )
        
        

        oof['predict'].iloc[val_idx] = model[fold].predict(X_valid)
    cv_scores = roc_auc_score(oof['Y'], oof['predict'])
    print(f'auc:{cv_scores}')
    gc.collect()

    return cv_scores

time1 = datetime.now()
print(f'start training: {time1}')

study2 = optuna.create_study(direction="maximize", study_name="into pussy", sampler=TPESampler(seed=992330))
def print_best_callback(study2, trial):
    print(f"Best value: {study2.best_value}, Best params: {study2.best_trial.params}")
study2.optimize(objective2, n_trials=87, n_jobs=2, callbacks=[print_best_callback, lambda study2, trial: gc.collect()])

print(f'training complete: {datetime.now()}')
print(f'training time: {datetime.now() - time1}')

print(f"\tBest value (auc): {study2.best_value:.5f}")
print(f"\tBest params:")

for key, value in study2.best_params.items():
    print(f"\t\t{key}: {value}")

joblib.dump(study2, "optuna tune/study_model4.pkl")

del df_tr, features, study2, time1, objective2
gc.collect()

################################################################################################################## model 2
df_tr = pd.read_csv(f'data/model2/FINAL_TABLE.csv', encoding='big5')
df_tr = df_tr.rename(columns = {'專案成立時間' : 'market_time', 'NEW_CONTACT_Y': 'CONTACT_Y', 'CONTACT_Y': 'old_CONTACT_Y'})

df_tr = df_tr[col_list]

##################################################################################################################
df_tr = df_tr[df_tr['DEL_ID'] == 'KEEP']
need_cols = [col for col in df_tr.columns if col not in ['自變數取樣時間', 'old_CONTACT_Y', 'MAX_SEC', 'DEL_ID']]
df_tr = df_tr[need_cols]

df_talk = pd.read_csv(f'data/model2/TMR_SELF_RECORD.csv', encoding='big5')
df_talk = df_talk.rename(columns = {'專案成立時間' : 'market_time'})

with open("endcodelist_sale_model2.pkl", "rb") as file:
    endcodelist = pickle.load(file)
with open("level_score_sale_model2.pkl", "rb") as file:
    endcode_ratio_level = pickle.load(file)
with open("score_range_1_sale_model2.pkl", "rb") as file:
    endcode_ratio = pickle.load(file)

df_talk['END_CODE'] = df_talk.apply(lambda end_code: dp.new_end_code(end_code), axis = 1)

df_talk_stat = dp.create_vars_records(df_talk, endcodelist, endcode_ratio, endcode_ratio_level) 

# bind into ABT
df_tr = df_tr.merge(df_talk_stat, on=['ID_SAS', 'market_time'], how='left')
del df_talk, df_talk_stat

df_tr = dp.feature_engineering(df_tr) 
cols_cate = [col for col in list(df_tr.columns) if (df_tr[col].dtype == 'object') & (col != 'ID_SAS')] 
dict_factorize = dict()
for col in cols_cate:
    dict_col = dp.factorize_categoricals(df_tr, col)
    dict_factorize[col] = dict_col
    df_tr[col].replace(dict_col, inplace=True)
df_tr = df_tr.drop(['CONTACT_Y'], axis = 1)
df_tr['Y'].fillna(value = 0, inplace = True)
##################################################################################################################

print('data preparation: i touch my self')
features = [col for col in df_tr.columns if col not in ['ID_SAS', CONF.training_y, 'market_time']]

def objective3(trial, fold = 5):
    params = {
        "objective" : "binary",
        "metric" : "auc",
        "boosting": 'gbdt',
        "verbosity": -1,
        #"max_depth" : -1,
        "seed": 5271,
        "n_estimators": trial.suggest_int("n_estimators", 1000,120000), 
        "learning_rate": trial.suggest_float("learning_rate", 0.00005, 0.015), #0.0002
        "num_leaves": trial.suggest_int("num_leaves", 2, 1000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.5, step=0.1
        ),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 50),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.1, 0.95, step=0.05
        ),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 500),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "boost_from_average": "false",
    }


    oof = df_tr[['ID_SAS', 'Y']].copy()
    oof['predict'] = 0
    model = [0]*fold

    cv = StratifiedKFold(n_splits = fold, shuffle=True, random_state=801217)

    cv_scores = [0]
    gc.collect()

    for fold, (trn_idx, val_idx) in enumerate(cv.split(df_tr, df_tr['Y'])):
        
        X_train, y_train = df_tr.iloc[trn_idx][features], df_tr.iloc[trn_idx]['Y']   
        X_valid, y_valid = df_tr.iloc[val_idx][features], df_tr.iloc[val_idx]['Y']
        
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        # evals_result = {}
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")

        print(f'fold: {fold}')
        model[fold] = lgb.train(
            params,
            trn_data,
            valid_sets=[val_data],
            verbose_eval = False,
            early_stopping_rounds= 3000,
            callbacks=[
                pruning_callback
            ],  # Add a pruning callback
            )
        
        

        oof['predict'].iloc[val_idx] = model[fold].predict(X_valid)
    cv_scores = roc_auc_score(oof['Y'], oof['predict'])
    print(f'auc:{cv_scores}')
    gc.collect()

    return cv_scores

time1 = datetime.now()
print(f'start training: {time1}')

study3 = optuna.create_study(direction="maximize", study_name="into pussy", sampler=TPESampler(seed=992330))
def print_best_callback(study3, trial):
    print(f"Best value: {study3.best_value}, Best params: {study3.best_trial.params}")
study3.optimize(objective3, n_trials=87, n_jobs=2, callbacks=[print_best_callback, lambda study3, trial: gc.collect()])

print(f'training complete: {datetime.now()}')
print(f'training time: {datetime.now() - time1}')

print(f"\tBest value (auc): {study3.best_value:.5f}")
print(f"\tBest params:")

for key, value in study3.best_params.items():
    print(f"\t\t{key}: {value}")

joblib.dump(study3, "optuna tune/study_model2.pkl")

del df_tr, features, study3, time1, objective3
gc.collect()
