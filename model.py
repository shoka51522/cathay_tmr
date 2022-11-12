import pandas as pd
import pickle
import joblib

PATH = 'prod/model'

# load model
def load_model(path, fold=5):
    """load model pickle file
    Args :
        path : string
            file path
        fold : integer
            the training set is split into k smaller sets for cross validation
    Return :
        lgb : lightgbm.Booster
            lgbm trained model
    """
    lgb = [0]*fold
    for i in range(fold):
        lgb[i] = joblib.load('%(filename)s%(num)03d.pkl' % {'filename':path,'num': i+1})
    
    return lgb

# predict results
def predict_result_sale(data, features, fold):
    """predict data by model
    Args :
        data : pandas.dataframe
            the data you want to predict
        features : list
            columns of the training model variables
        model : lightgbm.Booster
            lgbm trained model
        fold : int
            fold number
    Return :
        predictions : pandas.dataframe
            the predicted result
    """

    model = load_model(f'{PATH}/sale/tmr_lgbm')

    data = data.drop(['CONTACT_Y'], axis=1)

    predictions = data[['ID_SAS', 'market_time', 'Y']] 
    X_test1 = data[features].values
    for i in range(fold):
        predictions['fold{}'.format(i+1)] = model[i].predict(X_test1)
        print(i)
    # predictions['mean'] = predictions[['fold1', 'fold2', 'fold3', 'fold4', 'fold5']].sum(axis=1)/fold
    for i in range(fold):
        predictions['mean'] += predictions[f'fold{i+1}']/fold
    return predictions

def predict_result_con(data, features, fold):
    """predict data by model
    Args :
        data : pandas.dataframe
            the data you want to predict
        features : list
            columns of the training model variables
        model : lightgbm.Booster
            lgbm trained model
        fold : int
            fold number
    Return :
        predictions : pandas.dataframe
            the predicted result
    """

    model = load_model(f'{PATH}/contact/tmr_lgbm_con')

    data = data.drop(['Y'], axis=1)

    predictions = data[['ID_SAS', 'market_time', 'CONTACT_Y']] 
    X_test1 = data[features].values
    for i in range(fold):
        predictions['fold{}'.format(i+1)] = model[i].predict(X_test1)
        print(i)
    # predictions['mean'] = predictions[['fold1', 'fold2', 'fold3', 'fold4', 'fold5']].sum(axis=1)/fold
    predictions['mean'] = predictions[[f'fold{i+1}' for i in range(fold)]].sum(axis=1)/fold
    
    return predictions

def list_score(pred_result_sale, pred_result_contact, rank1 = 196, rank2 = 151, rank3 = 96):
    """from model results to list ranking orders
    Args :
        pred_result_sale : pandas.dataframe
            predict results by sale model 
        pred_result_contact : pandas.dataframe
            predict results by contact model
        rank1, rank2, rank3 : constant
            threshold of each rank from EDA
    Return :
        predictions : pandas.dataframe
            list scores for update
    """

    with open(f"{PATH}/listorder/listorder_sale.pkl", "rb") as file:
        listorder_sale = pickle.load(file)
    with open(f"{PATH}/listorder/listorder_con.pkl", "rb") as file:
        listorder_con = pickle.load(file)

    pred_result_contact.rename({'mean': 'mean_con'}, axis=1, inplace=True)

    df_score = pred_result_sale.iloc[:,[0,1,8]].merge(pred_result_contact.iloc[:,[0,1,8]], how='left', on=['ID_SAS','market_time'])

    df_score['sale_pr'] = pd.cut(df_score['mean'], 
                                 bins=list(listorder_sale['min_sale']) + ['1'], 
                                 labels=listorder_sale['sale_pr'])
    df_score['contact_pr'] = pd.cut(df_score['mean_con'], 
                                    bins=list(listorder_con['min_contact']) + ['1'], 
                                    labels=listorder_con['contact_pr'])

    # create score
    df_score['sale_pr'] = df_score.sale_pr.astype(float)
    df_score['contact_pr'] = df_score.contact_pr.astype(float)
    df_score['score'] = df_score['sale_pr']*2 + df_score['contact_pr']

    # create rank
    df_score.loc[(df_score['score'] >= rank1), 'rank'] = '1'
    df_score.loc[(df_score['score'] >= rank2) & (df_score['score'] < rank1), 'rank'] = '2'
    df_score.loc[(df_score['score'] >= rank3) & (df_score['score'] < rank2), 'rank'] = '3'
    df_score.loc[(df_score['score'] < rank3), 'rank'] = '4'

    return df_score


