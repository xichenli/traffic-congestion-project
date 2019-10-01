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
train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv')
test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')
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
    train['Time_p'+perc+'_IsZero'] = (train['TotalTimeStopped_p'+perc]==0).astype(int)
    train['Time_p'+perc+'_log'] = np.log(train['TotalTimeStopped_p'+perc]+1)
    train['Dist_p'+perc+'_IsZero'] = (train['DistanceToFirstStop_p'+perc]==0).astype(int)
    train['Dist_p'+perc+'_log'] = np.log(train['DistanceToFirstStop_p'+perc]+1)
    print(perc,np.sum(train['Time_p'+perc+'_IsZero']),np.sum(train['Dist_p'+perc+'_IsZero']))
print(train.columns)
#====================================================================#
cat_feat = ['Intersection','Hour', 'Weekend','Month', 'same_street_exact',
       'City', 'EntryType', 'ExitType']

param_cols = ['Intersection','Hour','Weekend','Month','same_street_exact','City', 
            'EntryType', 'ExitType','EntryHeading','ExitHeading','diffHeading',
           'average_temp','average_rainfall']
target_zero_cols = ['Time_p20_IsZero','Time_p50_IsZero','Time_p80_IsZero','Dist_p20_IsZero','Dist_p50_IsZero','Dist_p80_IsZero']
target_log_cols = ['Time_p20_log','Time_p50_log','Time_p80_log','Dist_p20_log','Dist_p50_log','Dist_p80_log']
#----------------- Model1 lightGBM ---------------------------------#

#Step 1 classify zero and non-zero
X_train,X_test,y_train,y_test=train_test_split(train[param_cols],train['Time_p80_IsZero'], test_size=0.2, random_state=42)
dtrain = lgb.Dataset(data=X_train, label=y_train,categorical_feature=cat_feat)
dvalid = lgb.Dataset(data=X_test, label=y_test,categorical_feature=cat_feat,reference=dtrain)
params = {'num_leaves': 200,
         'objective': 'binary',
         'metric':'cross_entropy',
         'boosting_type': 'gbdt',
         'learning_rate': 0.05,
         'feature_fraction': 0.9,
         'bagging_fraction': 0.8,
         'bagging_freq': 5,
         'verbose': 0}
gbm = lgb.train(params,
                dtrain,
                num_boost_round=200,
                valid_sets=dvalid,
                early_stopping_rounds=5)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test,y_pred)
print("auc:",auc)
for threshold in range(9):
    y_pred_class = y_pred > threshold/10.0
    cm = confusion_matrix(y_test, y_pred_class)
    tn, fp, fn, tp = cm.ravel()
    print('The mistake of prediction is:', tn,fp,fn,tp,float(fp)/(fp+tn))
