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
import math
train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv')
test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')
print("Train size",train.shape,"Test size",test.shape)
submission = pd.read_csv('../input/bigquery-geotab-intersection-congestion/sample_submission.csv')
target_cols = ['TotalTimeStopped_p20','TotalTimeStopped_p50','TotalTimeStopped_p80','DistanceToFirstStop_p20','DistanceToFirstStop_p50','DistanceToFirstStop_p80']
#----------- What type of road -----------------#
road_encoding = {'Street': 0, 'St': 0, 'Avenue': 1, 'Ave': 1, 'Boulevard': 2, 'Blvd': 2,'Road': 3,'Rd':3,
                'Drive': 4, 'Dr':4,'Lane': 5, 'Tunnel': 6, 'Highway': 7,'Hwy':7,'Express':8,'Expy':8}

def encode(x):
    if pd.isna(x):
        return -1
    for road in road_encoding.keys():
        if road in x:
            return road_encoding[road]
    return -1

for par in [train, test]:
    par['EntryType'] = par['EntryStreetName'].apply(encode)
    par['ExitType'] = par['ExitStreetName'].apply(encode)
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

monthly_snowfall = {'Atlanta1': 0.6, 'Atlanta5': 0, 'Atlanta6': 0, 'Atlanta7': 0, 'Atlanta8': 0, 'Atlanta9': 0, 
                    'Atlanta10': 0, 'Atlanta11': 0, 'Atlanta12': 0.2, 'Boston1': 12.9, 'Boston5': 0, 'Boston6': 0, 
                    'Boston7': 0, 'Boston8': 0, 'Boston9': 0, 'Boston10': 0,'Boston11': 1.3, 'Boston12': 9.0, 
                    'Chicago1': 11.5, 'Chicago5': 0, 'Chicago6': 0, 'Chicago7': 0, 'Chicago8': 0, 'Chicago9': 0, 
                    'Chicago10': 0,  'Chicago11': 1.3, 'Chicago12': 8.7, 'Philadelphia1': 6.5, 'Philadelphia5': 0, 
                    'Philadelphia6': 0, 'Philadelphia7': 0, 'Philadelphia8': 0, 'Philadelphia9':0 , 'Philadelphia10': 0, 
                    'Philadelphia11': 0.3, 'Philadelphia12': 3.4}
# Creating a new column by mapping the city_month variable to it's corresponding average monthly snowfall
train["average_snowfall"] = train['city_month'].map(monthly_rainfall)
test["average_snowfall"] = test['city_month'].map(monthly_rainfall)
train.drop('city_month', axis=1, inplace=True)
test.drop('city_month', axis=1, inplace=True)

#--------------- Distance to City Center---------------------#
# distance from the center of the city
def add_distance(df):
    
    df_center = pd.DataFrame({"Atlanta":[33.753746, -84.386330],
                             "Boston":[42.361145, -71.057083],
                             "Chicago":[41.881832, -87.623177],
                             "Philadelphia":[39.952583, -75.165222]})
    
    df["CenterDistance"] = df.apply(lambda row: math.sqrt((df_center[row.City][0] - row.Latitude) ** 2 +
                                                          (df_center[row.City][1] - row.Longitude) ** 2) , axis=1)

add_distance(train)
add_distance(test)
#---------------- Label encoder for City----------------------#
#HotEncoding of cities
#train = pd.concat([train,pd.get_dummies(train["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)
#test = pd.concat([test,pd.get_dummies(test["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)

encoder = LabelEncoder()
encoder.fit(pd.concat([train["City"],test["City"]]).drop_duplicates().values)
train["City"] = encoder.transform(train["City"])
test["City"] = encoder.transform(test["City"])
#------------------Target encoder for Weekend and Hour------------------------------#
# This function encode time information (time and weekend) according to a specific target
def Time_encoder(train_df,test_df,target_str):
    train_df['Time'] = train_df['Hour'].astype(str)+"_"+train_df['Weekend'].astype(str)
    test_df['Time'] = test_df['Hour'].astype(str)+"_"+test_df['Weekend'].astype(str)
    encoder = ce.TargetEncoder()
    encoder.fit(train_df['Time'],train[target_str])
    train_df['Time'] = encoder.transform(train_df['Time'])
    test_df['Time'] = encoder.transform(test_df['Time'])
    return train_df,test_df
#-----------------------------------------------------------------------------------#
cat_feat = ['Intersection','Month','City', 'EntryType', 'ExitType','Weekend','Hour']
#'Hour', 'Weekend','same_street_exact',
param_cols = ['Intersection','Month','City', 'EntryType', 'ExitType','Weekend','Hour',
              'EntryHeading','ExitHeading','diffHeading','average_temp','average_rainfall','average_snowfall','CenterDistance']
print(train[param_cols].head(3))
print(test[param_cols].head(3))

#----------------- Model1 CatBoost ---------------------------------#
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier,Pool
def cat_classification(X_train,X_valid,y_train,y_valid=None):
    y_train = (y_train>0).astype(int)
    number_of_positive = y_train.sum()
    number_of_negative = y_train.shape[0]-number_of_positive
    if y_valid is not None:
        y_valid = (y_valid>0).astype(int)
        eval_data = Pool(data=X_valid,label=y_valid,cat_features=cat_feat)
    else:
        eval_data = None
        
    model = CatBoostClassifier(iterations=300,loss_function='Logloss',custom_metric=['Logloss','AUC:hints=skip_train~false'],class_weights=[1,float(number_of_negative)/number_of_positive])
    model.fit(X_train,y_train,cat_features=cat_feat,eval_set=eval_data)
    y_pred_notzero = model.predict(X_valid)
    return y_pred_notzero
    
def cat_regression(X_train,X_valid,y_train,y_valid=None):
    y_train_nonzero = y_train[y_train>0]
    y_train_nonzero = np.log(y_train_nonzero+1)
    dtrain = Pool(data=X_train[y_train>0], label=y_train_nonzero,cat_features=cat_feat)

    if y_valid is not None:
        y_valid_nonzero = y_valid[y_valid>0]
        y_valid_nonzero = np.log(y_valid_nonzero+1)
        dvalid = Pool(data=X_valid[y_valid>0], label=y_valid_nonzero,cat_features=cat_feat)
    else:
        dvalid = None
        
    model = CatBoostRegressor(iterations=300,loss_function='RMSE',custom_metric=['RMSE'])
    model.fit(X_train[y_train>0],y_train_nonzero,cat_features=cat_feat,eval_set=dvalid)
    y_pred = model.predict(X_valid)
    return np.exp(y_pred)-1
#----------------- Model2 lightGBM ---------------------------------#
#Step 1 classify zero and non-zero
def gbm_classification(X_train,X_valid,y_train,y_valid=None):
    y_train = (y_train>0).astype(int)
    number_of_positive = y_train.sum()
    number_of_negative = y_train.shape[0]-number_of_positive
    dtrain = lgb.Dataset(data=X_train, label=y_train,categorical_feature=cat_feat)
    params = {'num_leaves': 200,
         'objective': 'binary',
         'metric':'cross_entropy',
         'boosting_type': 'gbdt',
         'learning_rate': 0.01,
         'feature_fraction': 0.9,
         'bagging_fraction': 0.8,
         'bagging_freq': 5,
         'verbose': 0,
         'scale_pos_weight':float(number_of_negative)/number_of_positive
             }
    if y_valid is not None:
        y_valid = (y_valid>0).astype(int)
        dvalid = lgb.Dataset(data=X_valid, label=y_valid,categorical_feature=cat_feat,reference=dtrain)
    else:
        dvalid = None
        
    gbm_binary = lgb.train(params,
                dtrain,
                num_boost_round=300,
                valid_sets=dvalid,
                early_stopping_rounds=5)
    y_pred_notzero = gbm_binary.predict(X_valid, num_iteration=gbm_binary.best_iteration)
    return y_pred_notzero

#Step2 Regression
def gbm_regression(X_train,X_valid,y_train,y_valid):
    #filter out those that are zero
    y_train_nonzero = y_train[y_train>0]
    y_train_nonzero = np.log(y_train_nonzero+1)
    dtrain = lgb.Dataset(data=X_train[y_train>0], label=y_train_nonzero,categorical_feature=cat_feat,free_raw_data=False)
    
    if y_valid is not None:
        y_valid_nonzero = y_valid[y_valid>0]
        y_valid_nonzero = np.log(y_valid_nonzero+1)
        dvalid = lgb.Dataset(data=X_valid[y_valid>0], label=y_valid_nonzero,categorical_feature=cat_feat,reference=dtrain,free_raw_data=False)
    else:
        dvalid = None
        
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
#        print(cv_results)
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
         'verbose': 0,
         'metric': {'rmse'}
        }
    clf = lgb.train(param, dtrain, 10000, valid_sets=dvalid,
                         verbose_eval=100, early_stopping_rounds = 300)
    y_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
    return np.exp(y_pred)-1

from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
#outfile = open("error.dat",'w')

get_test_results = True
tot_result = []
for it,target in enumerate(target_cols):
#   train,test = Time_encoder(train,test,target)
    if not get_test_results:
        X_train,X_valid,y_train,y_valid=train_test_split(train[param_cols],train[target], test_size=0.2, random_state=42)
    else:
        X_train = train[param_cols]
        y_train = train[target]
        X_valid = test[param_cols]
        y_valid = None
        
    nonzero_bgm = gbm_classification(X_train,X_valid,y_train,y_valid)
    predict_bgm = gbm_regression(X_train,X_valid,y_train,y_valid)
#    np.savetxt("validation.dat",np.c_[,y_valid.to_numpy()[:,np.newaxis]])
    nonezero_cat = cat_classification(X_train,X_valid,y_train,y_valid)
    predict_cat = cat_regression(X_train,X_valid,y_train,y_valid)
    if not get_test_results:
        np.savetxt(target+".dat",np.c_[nonezero_bgm[:,np.newaxis],predict_bgm[:,np.newaxis],nonezero_cat[:,np.newaxis],predict_cat[:,np.newaxis],y_valid.to_numpy()[:,np.newaxis]])
    else:
        threshold = 0.3
        if target[-2:]=="50": threshold = 0.6
        result = 0.7*(nonzero_bgm>threshold).astype(int)*predict_bgm+0.3*predict_cat*nonzero_cat
        submission[]
#    true_nonzero = (y_valid>0).astype(int)
#    for ith in range(9):
#        threshold = ith/10.0
#        pred_nonzero = (y_pred_nonzero>threshold).astype(int)
#        combined = pred_nonzero*y_pred
#        tn, fp, fn, tp = confusion_matrix(true_nonzero,pred_nonzero).ravel()
#       mse = mean_squared_error(combined,y_valid)
#        outfile.write(target+"\t"+str(threshold)+"\t"+str(tn)+"\t"+str(fp)+"\t"+str(fn)+"\t"+str(tp)+"\t"+str(mse)+"\n")
#outfile.close()











