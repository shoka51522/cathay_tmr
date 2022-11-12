import os
import sys
import pickle

ROOT = 'D:\code review'
PATH = 'prod/model'

os.chdir(ROOT)
sys.path.insert(0, ROOT)
print(os.getcwd())
print(sys.path)

class Configure():
    def __init__(self):
        #load feature
        with open(f"{PATH}/features/features.pkl", "rb") as file:
            self.features = pickle.load(file)

        #load sale end code list: results of mean encoding of end codes
        with open(f"{PATH}/features/endcodelist_sale.pkl", "rb") as file:
            self.endcodelist_sale = pickle.load(file)
        with open(f"{PATH}/features/level_score_sale.pkl", "rb") as file:
            self.endcode_ratio_level_sale = pickle.load(file)
        with open(f"{PATH}/features/score_range_1_sale.pkl", "rb") as file:
            self.endcode_ratio_sale = pickle.load(file)
        
        #load contact end code list: results of mean encoding of end codes
        with open(f"{PATH}/features/endcodelist_contact.pkl", "rb") as file:
            self.endcodelist_contact = pickle.load(file)
        with open(f"{PATH}/features/level_score_contact.pkl", "rb") as file:
            self.endcode_ratio_level_contact = pickle.load(file)
        with open(f"{PATH}/features/score_range_1_contact.pkl", "rb") as file:
            self.endcode_ratio_contact = pickle.load(file)