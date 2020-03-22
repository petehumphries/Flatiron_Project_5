import pandas as pd
import numpy as np
#import itertools
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
from sklearn.metrics import SCORERS, precision_score, recall_score, \
    accuracy_score, f1_score, roc_curve, auc, confusion_matrix, roc_auc_score


def delay_cutoff_plot(late_cutoff):
    '''Calculates class imbalnce for given level of timeliness,
       and returns a simple bar plot'''
    
    df = pd.read_pickle('./dataframe.pkl')
    df['DELAYED_TARGET'] = np.where(df['DEPARTURE_DELAY'] > late_cutoff, 1, 0)
    
    fig = plt.figure(figsize=(20,3))
    sns.countplot(y='DELAYED_TARGET', data=df)
    plt.ylabel('Delayed (1=Yes, 0=No)', fontsize=14)
    plt.xlabel('Count of delays > ' + str(late_cutoff) + ' min late', fontsize=14)
    plt.xticks(fontsize=14)
    plt.show()
    
    total = len(df)
    delayed = (df['DELAYED_TARGET']).sum()
    ontime = total - delayed
    delayed_per = round(100 * delayed / total,3)
    ontime_per =round(100 * ontime / total,3)
    
    print ('##########################################')
    print ('Total flights   :  ' + str(total))
    print ('Flights delayed :  ' + str(delayed))
    print ('Flights ontime  :  ' + str(ontime))
    print ('##########################################')
    print ('Flights delayed :  ' + str(delayed_per) + ' %')
    print ('Flights ontime  :  ' + str(ontime_per) + ' %')
    print ('##########################################')
    
def classifier_plot(col):
    '''Calculates class imbalnce for cancelled vs non-cancelled flights,
       and returns a simple bar plot'''
    
    df = pd.read_pickle('./dataframe.pkl')
    
    fig = plt.figure(figsize=(20,3))
    sns.countplot(y=col, data=df).set_title(str(col)) #, color='blue'
    plt.ylabel(str(col) + ' (1=Yes, 0=No)', fontsize=14)
    plt.xlabel('Count of ' + str(col), fontsize=14)
     
    plt.xticks(fontsize=14)
    plt.show()
    
    total = len(df)
    positive = (df[col]).sum()
    negative = total - positive
    positive_per = round(100 * positive / total,3)
    negative_per =round(100 * negative / total,3)
    
    print ('##########################################')
    print ('Total flights         :  ' + str(total))
    print ('Flights ' + str(col) + '      :   ' + str(positive))
    print ('Flights not ' + str(col) + '  :  ' + str(negative))
    print ('##########################################')
    print ('% Flights ' + str(col) + '    :  ' + str(positive_per) + ' %')
    print ('% Flights not ' + str(col) + ': ' + str(negative_per) + ' %')
    print ('##########################################')
    
     
    
    
def fill_col_na(col):
    df = pd.read_pickle('./dataframe.pk2')
    df[col] = df[col].replace('nan', np.nan).fillna(0)


def best_estimator(model):
    '''Returns the best estimator of/ C/ penalty
    '''
       
    score_best = model.best_score_
    C_best = model.best_estimator_.C
    max_iter_best = model.best_estimator_.max_iter
    penalty_best = model.best_estimator_.penalty
    
    
    print("Model best score : " + str(score_best))
    print("Best estimator of C is : " + str(C_best))
    print("Best estimator of max_iterations is : " + str(max_iter_best))
    print("Best estimator of penalty is : " + str(penalty_best))
    
    return score_best, C_best , max_iter_best, penalty_best

from sklearn.metrics import SCORERS, precision_score, recall_score, \
    accuracy_score, f1_score, roc_curve, auc, confusion_matrix, roc_auc_score



def calculate_scores(y_train_true, y_train_pred, y_valid_true, y_valid_pred, m, verbose = False):
    """
    calculates scores, updates dictionary with relevant scores
    if verbose is true, also prints out results
    returns a dictionary
    """
    
    scoring_dictionary = {}
    
    scoring_dictionary['train_accuracy'] = accuracy_score(y_train_true, y_train_pred)
    scoring_dictionary['validation_accuracy'] = accuracy_score(y_valid_true, y_valid_pred)
    scoring_dictionary['train_f1'] = f1_score(y_train_true, y_train_pred)
    scoring_dictionary['validation_f1'] = f1_score(y_valid_true, y_valid_pred)
    scoring_dictionary['train_auc'] = roc_auc_score(y_train_true, y_train_pred)
    scoring_dictionary['validation_auc'] = roc_auc_score(y_valid_true, y_valid_pred)
    scoring_dictionary['train_zweigcampbell'] = \
        calculate_zweig_campbell_score(y_train_true, y_train_pred, m)
    scoring_dictionary['validation_zweigcampbell'] = \
        calculate_zweig_campbell_score(y_valid_true, y_valid_pred, m)
    
    if verbose:
        print("Train accuracy : " + str(scoring_dictionary['train_accuracy']))
        print("Validation accuracy : " + str(scoring_dictionary['validation_accuracy']))
        print("Train F1 : " + str(scoring_dictionary['train_f1']))
        print("Validation F1 : " + str(scoring_dictionary['validation_f1']))
        print("Train AUC : " + str(scoring_dictionary['train_auc']))
        print("Validation AUC : " + str(scoring_dictionary['validation_auc']))
        print("Train Zweig-Campbell : " + str(scoring_dictionary['train_zweigcampbell']))
        print("Validation Zweig-Campbell : " + str(scoring_dictionary['validation_zweigcampbell']))
        
    return scoring_dictionary
    
    
def calculate_zweig_campbell_score(y_true, y_pred, m):
    """
    calculates tpr - m * fpr
    """
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    return tpr - fpr * m



