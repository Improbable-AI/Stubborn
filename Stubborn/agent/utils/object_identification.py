from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pickle

import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import numpy as np


NB = True
with open('./Stubborn/obj_id_data.pickle', 'rb') as handle:
    b = pickle.load(handle)

stepsize = 200

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
        1: (0, 2, stepsize),
        2: (0, 2, stepsize),
        3: (0, 2, stepsize),
        4: (0, 2, stepsize),
        5: (0, 2, stepsize),
        6: (0, 2, stepsize),
        7: (0, 2, stepsize),
        8: (0, 2, stepsize),
        9: (0, 2, stepsize),
        10: (0, 2, stepsize),
        11: (0, 2, stepsize),
        12: (0, 2, stepsize),
        13: (0,2,stepsize),
        14: (0,2,stepsize),  # NA
        15: (0, 2, stepsize),
        16: (0, 2, stepsize),  # 0.0003
        17: (0, 2, stepsize),
        18: (0, 2, stepsize),
        19: (0, 2, stepsize),  # NA
        20: (0, 2, stepsize),
        21: (0, 2, stepsize),  # NA
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
