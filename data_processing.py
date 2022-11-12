import pandas as pd
import numpy as np
import pickle
from prod import configure

CONF = configure.Configure()
PATH_DATA = 'prod/model/features'
#PATH_OUTPUT = ''

def reorder_csv(df):
    """reorder and tidy ABT
    Args :
        df : pandas.dataframe
            data of ABT
    Return :
        df : pandas.dataframe
            data of ABT after reorder col and col names
    """
    with open(f"{PATH_DATA}/col_list.pkl", "rb") as file:
        col_list = pickle.load(file)

    col_list = [col for col in col_list if col not in ['DEL_ID', 'MAX_SEC', 'old_CONTACT_Y']]

    df.rename({'': 'market_time'}, axis='columns', inplace=True)
    df = df[col_list] # ensure col order is the same as testing data
    df = df.drop([''], axis=1)

    return df

def new_end_code(end_code):
    """dict for transfer non-use end codes into current use end codes
    Args :
        end_code : pandas.dataframe
            data of TMR records within a col of end code that contains some non-use end codes
    Return :
        end_code : pandas.dataframe
            data of TMR records that contains a col of end code all transfered into current use end codes
    """
    if end_code['END_CODE']  in ['', '', '', '', ')']:
        return ''
    elif end_code['END_CODE']  in ['', '', '', '']:
        return '忙碌(開會、電話中)'
    elif end_code['END_CODE']  in ['', '', '', '', '']:
        return ''
    elif end_code['END_CODE']  == '':
        return ''
    elif end_code['END_CODE']  in ['', '']:
        return ''
    elif end_code['END_CODE']  == '':
        return ''
    elif end_code['END_CODE']  in ['', '']:
        return ''
    elif end_code['END_CODE']  == '':
        return ''
    elif end_code['END_CODE']  == '':
        return ''
    elif end_code['END_CODE']  in ['', '', '', '', '', '',
                                '', '', '']:
        return 'not'
    else:
        return end_code['END_CODE']


def create_vars_records(df_talk, endcodelist, endcode_ratio, endcode_ratio_level, aggregate_output = ['sum', 'mean', 'min', 'max', 'std']):
    """feature engineering turn end codes into 
    Args :
        df_talk : pandas.dataframe
            data of TMR records
        endcodelist : list
            list of unique end codes
        endcode_ratio : list
            list of mean encoding of each end code
        save_output : endcode_ratio_level
            list of mean encoding level of each end code, ranked in 4 levels, 4 is the highest
    Return :
        df_stat : pandas.dataframe
            data contains new variables from end code ratio & endcode_ratio_level after feature engineering
    """
    
    df_talk = df_talk.rename(columns = {'' : 'market_time'})

    df_talk = df_talk[df_talk["END_CODE"].isin(endcodelist)].copy()

    df_talk['END_CODE_1'] = df_talk["END_CODE"].replace(endcodelist, endcode_ratio)
    df_talk['END_CODE_level_1'] = df_talk["END_CODE"].replace(endcodelist, endcode_ratio_level)

    df_talk = df_talk.astype({'END_CODE_level_1': 'int64', 'END_CODE_1': 'float64'})

    df_stat_endcode1 = df_talk.groupby(['ID_SAS', 'market_time']).agg({'END_CODE_1': aggregate_output})
    df_stat_endcode1.columns = df_stat_endcode1.columns.map(lambda x: f'{x[1]}_end_code1')
    df_stat_endcodelevel = df_talk.groupby(['ID_SAS', 'market_time']).agg({'END_CODE_level_1': aggregate_output})
    df_stat_endcodelevel.columns = df_stat_endcodelevel.columns.map(lambda x: f'{x[1]}_end_code')
    df_stat = df_stat_endcode1.reset_index().merge(df_stat_endcodelevel.reset_index(), on=['ID_SAS', 'market_time'])

    df_talk = df_talk.merge(df_stat, on=['ID_SAS', 'market_time'], how='left')

    return df_talk #df_talk df_stat



def end_code_features(df_all, df_talk, model_type):
    """do feature engineering of df_talk with condition of sale/contact model, then bind into ABT dataframe
    Args :
        df_all : pandas.dataframe
            data of ABT
        df_talk : pandas.dataframe
            data of TMR records
        model_type : string
            for sale model or contact model
    Return :
        df_all : pandas.dataframe
            data of ABT contains TMR records features after feature engineering from create_vars_records function
    """    

    df_talk['END_CODE'] = df_talk.apply(lambda end_code: new_end_code(end_code), axis = 1)

    if model_type == "sale":
        df_talk = create_vars_records(df_talk,
                                        CONF.endcodelist_sale,
                                        CONF.endcode_ratio_sale,
                                        CONF.endcode_ratio_level_sale)
        
        df_all = df_all.merge(df_talk, on=['ID_SAS', 'market_time'], how='left')

    elif model_type == "contact":
        df_talk = create_vars_records(df_talk,
                                        CONF.endcodelist_contact,
                                        CONF.endcode_ratio_contact,
                                        CONF.endcode_ratio_level_contact)
        
        df_all = df_all.merge(df_talk, on=['ID_SAS', 'market_time'], how='left')
    
    else:
        pass
    
    return df_all

def factorize_categoricals(df, col):
    """do label encoding
    Args :
        df : pandas.dataframe
            data you want to do feature engineering
        col : str
            column name
    Return :
        dict_col : dict
            dictinoary that save the column encoding
    """
    col_value = df[col].unique()
    col_factorize, _ = pd.factorize(col_value)
    dict_col = dict(zip(col_value, col_factorize))

    return dict_col


def feature_engineering(df):
    """do some feature engineering
    Args :
        df : pandas.dataframe
            create new features, label encoding
    Return :
        df : pandas.dataframe
            data after feature engineering
    """
    
    # create new features
    df = df.copy()

    df['c_unit_pay'] = df['BILL_AMT'] / df['BILL_TIMES']
    df['c_prem_rate'] = (df['SUM_AUTO_PREM'] / df['BILL_AMT']) 
    df['c_credit_rate'] = df['CREDIT_AMT'] / df['BILL_AMT'] 
    df['c_trans_rate'] = df['TRANS_AMT'] / df['BILL_AMT'] 
    df['c_inperson_rate'] = df['PAY_INPERSON_AMT'] / df['BILL_AMT'] 
    df['c_survival_rate'] = df['SURVIVAL_MONEY_AMT'] / df['BILL_AMT'] 
    df['c_borrow'] = np.log1p(df['SUM_LOAN'] + df['SUM_AUTO_PREM'])
    df['c_stop_rate'] = df['STOP_NO'] / df['POLICY_NO_COUNT']
    df['c_payoff_rate'] = df['PAYOFF_NO'] / df['POLICY_NO_COUNT']
    df['c_cancel_rate'] = df['CANCEL_NO'] / df['POLICY_NO_COUNT']

    # label encoding
    cols_cate = [col for col in list(df.columns) if (df[col].dtype == 'object') & (col != 'ID_SAS')]

    dict_factorize = dict()
    for col in cols_cate:
        dict_col = factorize_categoricals(df, col)
        dict_factorize[col] = dict_col
        df[col].replace(dict_col, inplace=True)

    
    return df