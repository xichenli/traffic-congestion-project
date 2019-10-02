import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import lightgbm as lgb 
import os, sys 
import category_encoders as ce
from sklearn.metrics import accuracy_score

train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv')
test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')
print("Train size",train.shape,"Test size",test.shape)
submission = pd.read_csv('../input/bigquery-geotab-intersection-congestion/sample_submission.csv')
#----------- What type of road -----------------#
road_encoding = {
    'Road': 1,
    'Street': 2,
    'Avenue': 2,
    'Drive': 3,
    'Broad': 3,
    'Boulevard': 4
}
def encode(x):
    if pd.isna(x):
        return 0
    for road in road_encoding.keys():
        if road in x:
            return road_encoding[road]
    return 0
train['EntryType'] = train['EntryStreetName'].apply(encode)
train['ExitType'] = train['ExitStreetName'].apply(encode)
test['EntryType'] = test['EntryStreetName'].apply(encode)
test['ExitType'] = test['ExitStreetName'].apply(encode)
#-------------------Direction encodeing -----------------------#
directions = {
    'N': 0,
    'NE': 1/4,
    'E': 1/2,
    'SE': 3/4,
    'S': 1,
    'SW': 5/4,
    'W': 3/2,
    'NW': 7/4
}
train['EntryHeading'] = train['EntryHeading'].map(directions)
train['ExitHeading'] = train['ExitHeading'].map(directions)

test['EntryHeading'] = test['EntryHeading'].map(directions)
test['ExitHeading'] = test['ExitHeading'].map(directions)
train['diffHeading'] = train['EntryHeading']-train['ExitHeading']
test['diffHeading'] = test['EntryHeading']-test['ExitHeading']
train.head()
#entering street == exit street?
train["same_street_exact"] = (train["EntryStreetName"] ==  train["ExitStreetName"]).astype(int)
test["same_street_exact"] = (test["EntryStreetName"] ==  test["ExitStreetName"]).astype(int)
#----------Intersection encoding--------------#
train["Intersection"] = train["IntersectionId"].astype(str) + train["City"]
test["Intersection"] = test["IntersectionId"].astype(str) + test["City"]

encoder = LabelEncoder()
encoder.fit(pd.concat([train["Intersection"],test["Intersection"]]).drop_duplicates().values)
train["Intersection"] = encoder.transform(train["Intersection"])
test["Intersection"] = encoder.transform(test["Intersection"])
#--------- Weather factor---------------------#
#Add temperature (°F) of each city by month
#Reference: https://www.kaggle.com/brokenfulcrum/geotab-baseline

monthly_av = {'Atlanta1': 43, 'Atlanta5': 69, 'Atlanta6': 76, 'Atlanta7': 79, 'Atlanta8': 78, 'Atlanta9': 73,
              'Atlanta10': 62, 'Atlanta11': 53, 'Atlanta12': 45, 'Boston1': 30, 'Boston5': 59, 'Boston6': 68,
              'Boston7': 74, 'Boston8': 73, 'Boston9': 66, 'Boston10': 55,'Boston11': 45, 'Boston12': 35,
              'Chicago1': 27, 'Chicago5': 60, 'Chicago6': 70, 'Chicago7': 76, 'Chicago8': 76, 'Chicago9': 68,
              'Chicago10': 56,  'Chicago11': 45, 'Chicago12': 32, 'Philadelphia1': 35, 'Philadelphia5': 66,
              'Philadelphia6': 76, 'Philadelphia7': 81, 'Philadelphia8': 79, 'Philadelphia9': 72, 'Philadelphia10': 60,
              'Philadelphia11': 49, 'Philadelphia12': 40}
train['city_month'] = train["City"] + train["Month"].astype(str)
test['city_month'] = test["City"] + test["Month"].astype(str)
# Creating a new column by mapping the city_month variable to it's corresponding average monthly temperature
train["average_temp"] = train['city_month'].map(monthly_av)
test["average_temp"] = test['city_month'].map(monthly_av)
#Add rainfall (inches) of each city by month

monthly_rainfall = {'Atlanta1': 5.02, 'Atlanta5': 3.95, 'Atlanta6': 3.63, 'Atlanta7': 5.12, 'Atlanta8': 3.67, 'Atlanta9': 4.09,
              'Atlanta10': 3.11, 'Atlanta11': 4.10, 'Atlanta12': 3.82, 'Boston1': 3.92, 'Boston5': 3.24, 'Boston6': 3.22,
              'Boston7': 3.06, 'Boston8': 3.37, 'Boston9': 3.47, 'Boston10': 3.79,'Boston11': 3.98, 'Boston12': 3.73,
              'Chicago1': 1.75, 'Chicago5': 3.38, 'Chicago6': 3.63, 'Chicago7': 3.51, 'Chicago8': 4.62, 'Chicago9': 3.27,
              'Chicago10': 2.71,  'Chicago11': 3.01, 'Chicago12': 2.43, 'Philadelphia1': 3.52, 'Philadelphia5': 3.88,
              'Philadelphia6': 3.29, 'Philadelphia7': 4.39, 'Philadelphia8': 3.82, 'Philadelphia9':3.88 , 'Philadelphia10': 2.75,
              'Philadelphia11': 3.16, 'Philadelphia12': 3.31}
# Creating a new column by mapping the city_month variable to it's corresponding average monthly rainfall
train["average_rainfall"] = train['city_month'].map(monthly_rainfall)
test["average_rainfall"] = test['city_month'].map(monthly_rainfall)
train.drop('city_month', axis=1, inplace=True)
test.drop('city_month', axis=1, inplace=True)

#---------------- Label encoder for City----------------------#
#HotEncoding of cities
#train = pd.concat([train,pd.get_dummies(train["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)
#test = pd.concat([test,pd.get_dummies(test["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)

encoder = LabelEncoder()
encoder.fit(pd.concat([train["City"],test["City"]]).drop_duplicates().values)
train["City"] = encoder.transform(train["City"])
test["City"] = encoder.transform(test["City"])

#------------- Transform response to a two stage model--------------#
for perc in ['20','50','80']:
    train['Time_p'+perc+'_NotZero'] = (train['TotalTimeStopped_p'+perc]>0).astype(int)
    train['Time_p'+perc+'_log'] = np.log(train['TotalTimeStopped_p'+perc]+1)
    train['Dist_p'+perc+'_NotZero'] = (train['DistanceToFirstStop_p'+perc]>=0).astype(int)
    train['Dist_p'+perc+'_log'] = np.log(train['DistanceToFirstStop_p'+perc]+1)
    print(perc,np.sum(train['Time_p'+perc+'_NotZero']),np.sum(train['Dist_p'+perc+'_NotZero']))
print(train.columns)
#====================================================================#
cat_feat = ['Intersection','Hour', 'Weekend','Month', 'same_street_exact',
       'City', 'EntryType', 'ExitType']

param_cols = ['Intersection','Hour','Weekend','Month','same_street_exact','City', 
            'EntryType', 'ExitType','EntryHeading','ExitHeading','diffHeading',
           'average_temp','average_rainfall']
target_zero_cols = ['Time_p20_NotZero','Time_p50_NotZero','Time_p80_NotZero','Dist_p20_NotZero','Dist_p50_NotZero','Dist_p80_NotZero']
target_log_cols = ['Time_p20_log','Time_p50_log','Time_p80_log','Dist_p20_log','Dist_p50_log','Dist_p80_log']

#----------------- Model1 lightGBM ---------------------------------#
#Step 1 classify zero and non-zero
def gbm_classification(X_train,X_valid,y_train,y_valid):
    y_train = (y_train>0).astype(int)
    y_valid = (y_valid>0).astype(int)
    dtrain = lgb.Dataset(data=X_train, label=y_train,categorical_feature=cat_feat)
    dvalid = lgb.Dataset(data=X_valid, label=y_valid,categorical_feature=cat_feat,reference=dtrain)
    params = {'num_leaves': 200,
         'objective': 'binary',
         'metric':'cross_entropy',
         'boosting_type': 'gbdt',
         'learning_rate': 0.01,
         'feature_fraction': 0.9,
         'bagging_fraction': 0.8,
         'bagging_freq': 5,
         'verbose': 0}
    gbm_binary = lgb.train(params,
                dtrain,
                num_boost_round=200,
                valid_sets=dvalid,
                early_stopping_rounds=5)
    y_pred_notzero = gbm_binary.predict(X_valid, num_iteration=gbm_binary.best_iteration)
    return y_pred_notzero

#Step2 Regression
def gbm_regression(X_train,X_valid,y_train,y_valid):
    #filter out those that are zero
    y_train_nonzero = y_train[y_train>0]
    y_valid_nonzero = y_valid[y_valid>0]
    y_train_nonzero = np.log(y_train_nonzero+1)
    y_valid_nonzero = np.log(y_valid_nonzero+1)
    dtrain = lgb.Dataset(data=X_train, label=y_train_nonzero,categorical_feature=cat_feat)
    dvalid = lgb.Dataset(data=X_valid, label=y_valid_nonzero,categorical_feature=cat_feat,reference=dtrain)
    
    def hyp_lgbm(num_leaves, feature_fraction, bagging_fraction, max_depth, min_split_gain, min_child_weight, lambda_l1, lambda_l2):
        params = {'application':'regression','num_iterations': 400,
                  'learning_rate':0.01,
                  'metric':'rmse'} # Default parameters
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['lambda_l1'] = lambda_l1
        params['lambda_l2'] = lambda_l2
        cv_results = lgb.cv(params, dtrain, nfold=5, seed=17,categorical_feature=cat_feat, stratified=False,
                            verbose_eval =None)
        print(cv_results)
        return -np.min(cv_results['rmse-mean'])
# Domain space-- Range of hyperparameters
    pds = {'num_leaves': (120, 230),
          'feature_fraction': (0.1, 0.5),
          'bagging_fraction': (0.8, 1),
           'lambda_l1': (0,3),
           'lambda_l2': (0,5),
          'max_depth': (8, 19),
          'min_split_gain': (0.001, 0.1),
          'min_child_weight': (1, 20)
          }

    # Surrogate model
    optimizer = BayesianOptimization(hyp_lgbm,pds,random_state=7)

    # Optimize
    optimizer.maximize(init_points=5, n_iter=10)
    #optimizer.max
    p = optimizer.max['params']
    param = {'num_leaves': int(round(p['num_leaves'])),
         'feature_fraction': p['feature_fraction'],
         'bagging_fraction': p['bagging_fraction'],
         'max_depth': int(round(p['max_depth'])),
         'lambda_l1': p['lambda_l1'],
         'lambda_l2':p['lambda_l2'],
         'min_split_gain': p['min_split_gain'],
         'min_child_weight': p['min_child_weight'],
         'learning_rate':0.05,
         'objective': 'regression',
         'boosting_type': 'gbdt',
         'verbose': 1,
         'metric': {'rmse'}
        }
    clf = lgb.train(param, dtrain, 10000, valid_sets=dvalid,
                         verbose_eval=100, early_stopping_rounds = 200)
    y_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
    return np.exp(y_pred)-1

from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
outfile = open("error.dat",'w')
for target in target_cols:
    X_train,X_valid,y_train,y_valid=train_test_split(train[param_cols],train[target], test_size=0.2, random_state=42)
    y_pred_nonzero = gbm_classification(X_train,X_valid,y_train,y_valid)
    y_pred = gbm_regression(X_train,X_valid,y_train,y_valid)
    
    #Check results
    y_valid = y_valid.to_numpy()
    true_nonzero = (y_valid>0).astype(int)
    for ith in range(9):
        threshold = ith/10.0
        pred_nonzero = (y_pred_nonzero>threshold).astype(int)
        combined = pred_nonzero*y_pred
        tn, fp, fn, tp = confusion_matrix(true_nonzero,pred_nonzero).ravel()
        mse = mean_squared_error(combined,y_valid)
        outfile.write(target+"\t"+str(threshold)+"\t"+str(tn)+"\t"+str(fp)+"\t"+str(fn)+"\t"+str(tp)+"\t"+str(mse)+"\n")
outfile.close()
