"""
<your project name>
    Author: XXX
        XXX
    Date: 2022/9/2
        <your proj date. ex: YYYY/MM/DD>
    Model Version: TMR model XXX
        <your model version. ex: 1.0.0>
    Model Type: lightgbm
        <your model type. ex: XGB>
    Introduction: XXX
        <project introduction>
"""

# -- System Package import
import pandas as pd

from prod.func import data_processing as dp 
from prod.func import model as md
from prod import configure

CONF = configure.Configure()
PATH_DATA_INPUT = 'prod/tests/input'
PATH_DATA_OUTPUT = 'prod/tests/output'
TYPE = "" #types: 
MONTH = 2022XX

# load csv
df_all = pd.read_csv(f"{PATH_DATA_INPUT}/{TYPE}/{MONTH}/XXX.csv", encoding='big5')
df_talk = pd.read_csv(f"{PATH_DATA_INPUT}/{TYPE}/{MONTH}/XXX.csv",
                      encoding='big5')

##### -- model prediction -- #####
def predict(df_all, df_talk):
    ''' the predict procedure
    Args :
        df_all : pandas.dataframe
            the input ABT
        df_talk : pandas.dataframe
            input features of endcode, with merge with df_all through the process
    Return :
        pred_sale : pandas.dataframe
            model results of sale model
        pred_con : pandas.dataframe
            model results of contact model
    '''
    print('start process')

    # reorder FINAL_TABLE.csv
    df_all = dp.reorder_csv(df_all)
    print(f'shape of data: {df_all.shape}')

    # feature engineering - end code mean encoding
    df_all_sale = dp.end_code_features(df_all, df_talk, "sale")
    df_all_contact = dp.end_code_features(df_all, df_talk, "contact")

    # feature engineering - other variables
    df_all_sale = dp.feature_engineering(df_all_sale)
    df_all_contact = dp.feature_engineering(df_all_contact)

    # predict
    print("Start to get the predicted result of sale model")
    pred_sale = md.predict_result_sale(df_all_sale, CONF.features, 5)

    print("Start to get the predicted result of contact model")
    pred_con = md.predict_result_con(df_all_contact, CONF.features, 5)

    return pred_sale, pred_con


##### -- list score -- #####
pred_sale, pred_con = predict(df_all, df_talk)
df_score = md.list_score(pred_sale, pred_con)

print("people in each rank")
print(df_score['rank'].value_counts())


# save results
pred_sale.to_csv(f"{PATH_DATA_OUTPUT}/{TYPE}/{MONTH}/sale_score_{MONTH}.csv", index=False)
pred_con.to_csv(f"{PATH_DATA_OUTPUT}/{TYPE}/{MONTH}/contact_score_{MONTH}.csv", index=False)
df_score.to_csv(f"{PATH_DATA_OUTPUT}/{TYPE}/{MONTH}/listorder_score_{MONTH}.csv", index=False)
