import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, \
    cross_validate, cross_val_predict, KFold, GridSearchCV


def best_estimator(model):
    '''Returns the best estimator of/ C/ penalty
    '''
       
    C_best = model.best_estimator_.C
    max_iter_best = model.best_estimator_.max_iter
    penalty_best = model.best_estimator_.penalty
        
    print("Best estimator of C is : " + str(C_best))
    print("Best estimator of max_iterations is : " + str(max_iter_best))
    print("Best estimator of penalty is : " + str(penalty_best))

    return C_best , max_iter_best, penalty_best
