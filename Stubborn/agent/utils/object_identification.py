from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pickle

import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import numpy as np


NB = False
with open('./Stubborn/log_path_16_middle.pickle', 'rb') as handle:
    b = pickle.load(handle)
with open('./Stubborn/log_path_17_conf_middle.pickle', 'rb') as handle:
    b2 = pickle.load(handle)
for i in range(len(b)):
    lg = b[i]['goal_log']
    for j in range(len(lg)):
        b[i]['goal_log'][j]['conflict'] = b2[i]['goal_log'][j]['conflict']

if not NB:
    params = {
        1: (1, 1, 200),
        2: (0, 0, 70),
        3: (0, 0, 70),
        4: (0, 1, 70),
        5: (0, 0, 70),
        6: None,
        7: (0, 0, 200),
        8: (0, 0, 200),
        9: (1, 1, 200),
        10: (0, 0, 200),
        11: None,
        12: (0, 0, 200),
        13: None,
        14: None,  # NA
        15: (1, 0, 200),
        16: (1, 1, 200),  # 0.0003
        17: (1, 1, 200),
        18: (1, 1, 200),
        19: None,  # NA
        20: (1, 1, 200),
        21: None,  # NA
    }
else:
    params = {
        1: (0, 2, 200),
        2: (0, 2, 70),
        3: (0, 2, 70),
        4: (0, 2, 70),
        5: (0, 2, 70),
        6: None,
        7: (0, 2, 200),
        8: (0, 2, 200),
        9: (0, 2, 200),
        10: (0, 2, 200),
        11: None,
        12: (0, 2, 200),
        13: None,
        14: None,  # NA
        15: (0, 2, 200),
        16: (0, 2, 200),  # 0.0003
        17: (0, 2, 200),
        18: (0, 2, 200),
        19: None,  # NA
        20: (0, 2, 200),
        21: None,  # NA
    }

def item2feature(item):
    cf = item['conflict']
    if NB:
        return [item['total']['cumu'], item['cumu'][0], item['total']['ratio'],
                item['total']['score'], cf['normal']
                ]
    else:
        return [item['total']['cumu'],item['total']['ratio'],item['total']['score'],cf['normal']]

def get_feature_for(b,interest,feature_mode,rg = None):

    x = []
    y = []
    for i in range(len(b)):
        if feature_mode == 0 and b[i]['goal'] != interest:
            continue
        if rg is not None and (rg[0] <= i and i <= rg[1]):
            continue
        lg = b[i]['goal_log']
        for item in lg:
            x.append(item2feature(item))
            y.append(float(item['suc']))
    if len(x) == 0:
        return None,None
    return np.array(x),np.array(y)


def get_oracle(b,interest,rg = None):
    param = params[interest]
    if param is None:
        return None
    feature_mode = param[0]
    if param[1] == 0:
        classifier = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10,max_depth=4),n_estimators=10,learning_rate=0.6)
    elif param[1] == 1:
        classifier = RandomForestClassifier(n_estimators=30, max_depth=9)
    else:
        classifier = MultinomialNB()
    x,y = get_feature_for(b,interest,feature_mode,rg)
    if x is None:
        return None
    return classifier.fit(x, y)

predictors = {}
for i in range(1,22):
    predictors[i] = get_oracle(b,i)

def recal_predictors(rg):
    print("recal",rg)
    global predictors
    for i in range(1, 22):
        predictors[i] = get_oracle(b, i, rg)

def get_prediction(item,goal):
    if params[goal] is None or item['step']>params[goal][2]:
        return True
    if predictors[goal] is None:
        return True
    sc = np.array([item2feature(item)])
    score = predictors[goal].predict(sc)
    return score > 0.5
