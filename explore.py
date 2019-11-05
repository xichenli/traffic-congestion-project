import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import math,time,gc
start = time.time()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['UniqueID']=train["City"]+train['Latitude'].astype(str).str[4:7]+train['Longitude'].astype(str).str[4:7]
test['UniqueID']=test["City"]+test['Latitude'].astype(str).str[4:7]+test['Longitude'].astype(str).str[4:7]
target_cols = ['TotalTimeStopped_p20','TotalTimeStopped_p50','TotalTimeStopped_p80','DistanceToFirstStop_p20','DistanceToFirstStop_p50','DistanceToFirstStop_p80']
#for col in target_cols:
#  subtrain = train[train[col]>0]
#  plt.figure()
#  plt.boxplot(np.log(subtrain[col]+1))
#  plt.show()
# Remove outliers #
print("data shape:",train.shape,test.shape)
train = train[(train['TotalTimeStopped_p50']<139.7702496) & (train['DistanceToFirstStop_p50']<151.9573668)]
print("new train shape:",train.shape)
train.drop(['RowId','Path','IntersectionId','TotalTimeStopped_p40','TotalTimeStopped_p60','TimeFromFirstStop_p20','TimeFromFirstStop_p40','TimeFromFirstStop_p50','TimeFromFirstStop_p60','TimeFromFirstStop_p80','DistanceToFirstStop_p40','DistanceToFirstStop_p60'],axis=1,inplace=True)
gc.collect()


#----------- Convert Heading Direction to Angle------------------------------------#

directions = {
    'N': 0.5*math.pi,
    'NE': 0.25*math.pi,
    'E': 0.,
    'SE':-0.25*math.pi ,
    'S': -0.5*math.pi,
    'SW': -0.75*math.pi,
    'W': math.pi,
    'NW': 0.75*math.pi
}
train['EntryHeading'] = train['EntryHeading'].map(directions)
train['ExitHeading'] = train['ExitHeading'].map(directions)
train['diffHeading'] = train['EntryHeading']-train['ExitHeading']
train['diffHeading'] = train['diffHeading'] +(train['diffHeading']>math.pi)*(-2*math.pi)+(train['diffHeading']<-math.pi)*(2*math.pi)
train["same_street_exact"] = (train["EntryStreetName"] ==  train["ExitStreetName"]).astype(int)

test['EntryHeading'] = test['EntryHeading'].map(directions)
test['ExitHeading'] = test['ExitHeading'].map(directions)
test['diffHeading'] = test['EntryHeading']-test['ExitHeading']
test['diffHeading'] = test['diffHeading'] +(test['diffHeading']>math.pi)*(-2*math.pi)+(test['diffHeading']<-math.pi)*(2*math.pi)
test["same_street_exact"] = (test["EntryStreetName"] ==  test["ExitStreetName"]).astype(int)

#------------- Add number of Exit direction, number of Entry direction --------------#

cols = ['UniqueID','Latitude','Longitude','EntryHeading','ExitHeading','City']
df_all = pd.concat([train[cols],test[cols]])
df_all = df_all.drop_duplicates()
df_nexit = df_all.groupby(['UniqueID','EntryHeading'])['ExitHeading'].nunique().reset_index()
df_nexit.columns = ['UniqueID','EntryHeading','ExitHeading_Count']
df_nentry = df_all.groupby(['UniqueID','ExitHeading'])['EntryHeading'].nunique().reset_index()
df_nentry.columns = ['UniqueID','ExitHeading','EntryHeading_Count']

train = train.merge(df_nexit,on=['UniqueID','EntryHeading'],how='left',copy=False,suffixes=('', '_y'))
train = train.merge(df_nentry,on=['UniqueID','ExitHeading'],how='left',copy=False,suffixes=('', '_y'))

test = test.merge(df_nexit,on=['UniqueID','EntryHeading'],how='left',copy=False,suffixes=('', '_y'))
test = test.merge(df_nentry,on=['UniqueID','ExitHeading'],how='left',copy=False,suffixes=('', '_y'))

df_all = df_all.drop_duplicates(['UniqueID','ExitHeading'])
df_center = pd.DataFrame({"Atlanta":[33.753746, -84.386330],
                           "Boston":[42.361145, -71.057083],
                           "Chicago":[41.881832, -87.623177],
                           "Philadelphia":[39.952583, -75.165222]})
df_all["CityCenterDistance"] = df_all.apply(lambda row: math.sqrt((df_center[row.City][0] - row.Latitude) ** 2 +
                                                        (df_center[row.City][1] - row.Longitude) ** 2), axis=1)
    # cos\theta value towards city center
df_all["ExitToCenterAngle"] = df_all.apply(lambda row: math.cos(row.ExitHeading)*(df_center[row.City][0] - row.Latitude)/row.CityCenterDistance
                                          +math.sin(row.ExitHeading)*(df_center[row.City][1] - row.Longitude)/row.CityCenterDistance,axis=1)
train = train.merge(df_all,on=['UniqueID','ExitHeading'],how='left',copy=False,suffixes=('', '_y'))
test = test.merge(df_all,on=['UniqueID','ExitHeading'],how='left',copy=False,suffixes=('', '_y'))

train.drop(list(train.filter(regex='_y$')), axis=1, inplace=True)
test.drop(list(test.filter(regex='_y$')), axis=1, inplace=True)
del [df_all,df_nexit,df_nentry]



#----------- What type of road -----------------#
road_encoding = {'Street': 1, 'St': 1, 'Avenue': 1, 'Ave': 1, 'Boulevard': 2, 'Blvd': 2,'Road': 3,'Rd':3,
                'Drive': 4, 'Dr':4,'Lane': 5, 'Tunnel': 8, 'Highway': 7,'Hwy':7,'Express':8,'Expy':8}
def encode(x):
    if pd.isna(x):
        return 0
    for road in road_encoding.keys():
        if road in x:
            return road_encoding[road]
    return 0
train['EntryType'] = train['EntryStreetName'].apply(encode)
test['EntryType'] = test['EntryStreetName'].apply(encode)
train['ExitType'] = train['ExitStreetName'].apply(encode)
test['ExitType'] = test['ExitStreetName'].apply(encode)


#--------- Weather factor---------------------#
#Add temperature (°F) of each city by month
#Reference: https://www.kaggle.com/brokenfulcrum/geotab-baseline
monthly_temp = {'Atlanta1': 43, 'Atlanta5': 69, 'Atlanta6': 76, 'Atlanta7': 79, 'Atlanta8': 78, 'Atlanta9': 73,
              'Atlanta10': 62, 'Atlanta11': 53, 'Atlanta12': 45, 'Boston1': 30, 'Boston5': 59, 'Boston6': 68,
              'Boston7': 74, 'Boston8': 73, 'Boston9': 66, 'Boston10': 55,'Boston11': 45, 'Boston12': 35,
              'Chicago1': 27, 'Chicago5': 60, 'Chicago6': 70, 'Chicago7': 76, 'Chicago8': 76, 'Chicago9': 68,
              'Chicago10': 56,  'Chicago11': 45, 'Chicago12': 32, 'Philadelphia1': 35, 'Philadelphia5': 66,
              'Philadelphia6': 76, 'Philadelphia7': 81, 'Philadelphia8': 79, 'Philadelphia9': 72, 'Philadelphia10': 60,
              'Philadelphia11': 49, 'Philadelphia12': 40}
monthly_rainfall = {'Atlanta1': 5.02, 'Atlanta5': 3.95, 'Atlanta6': 3.63, 'Atlanta7': 5.12, 'Atlanta8': 3.67, 'Atlanta9': 4.09,
              'Atlanta10': 3.11, 'Atlanta11': 4.10, 'Atlanta12': 3.82, 'Boston1': 3.92, 'Boston5': 3.24, 'Boston6': 3.22,
              'Boston7': 3.06, 'Boston8': 3.37, 'Boston9': 3.47, 'Boston10': 3.79,'Boston11': 3.98, 'Boston12': 3.73,
              'Chicago1': 1.75, 'Chicago5': 3.38, 'Chicago6': 3.63, 'Chicago7': 3.51, 'Chicago8': 4.62, 'Chicago9': 3.27,
              'Chicago10': 2.71,  'Chicago11': 3.01, 'Chicago12': 2.43, 'Philadelphia1': 3.52, 'Philadelphia5': 3.88,
              'Philadelphia6': 3.29, 'Philadelphia7': 4.39, 'Philadelphia8': 3.82, 'Philadelphia9':3.88 , 'Philadelphia10': 2.75,
              'Philadelphia11': 3.16, 'Philadelphia12': 3.31}
monthly_snowfall = {'Atlanta1': 0.6, 'Atlanta5': 0, 'Atlanta6': 0, 'Atlanta7': 0, 'Atlanta8': 0, 'Atlanta9': 0, 
                    'Atlanta10': 0, 'Atlanta11': 0, 'Atlanta12': 0.2, 'Boston1': 12.9, 'Boston5': 0, 'Boston6': 0, 
                    'Boston7': 0, 'Boston8': 0, 'Boston9': 0, 'Boston10': 0,'Boston11': 1.3, 'Boston12': 9.0, 
                    'Chicago1': 11.5, 'Chicago5': 0, 'Chicago6': 0, 'Chicago7': 0, 'Chicago8': 0, 'Chicago9': 0, 
                    'Chicago10': 0,  'Chicago11': 1.3, 'Chicago12': 8.7, 'Philadelphia1': 6.5, 'Philadelphia5': 0, 
                    'Philadelphia6': 0, 'Philadelphia7': 0, 'Philadelphia8': 0, 'Philadelphia9':0 , 'Philadelphia10': 0, 
                    'Philadelphia11': 0.3, 'Philadelphia12': 3.4}
# Creating a new column by mapping the city_month variable to it's corresponding average monthly snowfall
train['city_month'] = train["City"] + train["Month"].astype(str)
train["Temp"] = train['city_month'].map(monthly_temp)
train["Rainfall"] = train['city_month'].map(monthly_rainfall)
train["Snowfall"] = train['city_month'].map(monthly_snowfall)
train.drop('city_month', axis=1, inplace=True)
test['city_month'] = test["City"] + test["Month"].astype(str)
test["Temp"] = test['city_month'].map(monthly_temp)
test["Rainfall"] = test['city_month'].map(monthly_rainfall)
test["Snowfall"] = test['city_month'].map(monthly_snowfall)
test.drop('city_month', axis=1, inplace=True)


# ------------ Business of the Neighbor Area -------------------------#
subtrain = train.drop_duplicates(['UniqueID'])
lat = subtrain['Latitude'].to_numpy()
lon = subtrain['Longitude'].to_numpy()
uid = subtrain['UniqueID'].to_numpy()
sigma = 1e-5
weights = np.exp(-1.0/sigma*(np.power((lat[:,np.newaxis]-lat[np.newaxis,:]),2)+np.power((lon[:,np.newaxis]-lon[np.newaxis,:]),2)))
normalization = np.sum(weights,axis=1).reshape((-1,1))
df_dist = pd.DataFrame(data=weights)
df_dist['UniqueID']=uid
ncol_dist = int(df_dist.shape[1]) 
train_midday = train[train['Hour'].isin([10,11,12,13,14]) & train['Weekend']==0]
self_mean = train_midday.groupby(['UniqueID'])[target_cols].mean().reset_index()
df_dist = df_dist.merge(self_mean,on='UniqueID',how='left',copy=False,suffixes=('', '_y'))
df_dist = df_dist.to_numpy()
results = np.dot(df_dist[:,:(ncol_dist-1)],df_dist[:,ncol_dist:])/normalization
neighbor_df = pd.DataFrame(results,columns=['NB_T20','NB_T50','NB_T80','NB_D20','NB_D50','NB_D80'])
neighbor_df['UniqueID'] = uid
train = train.merge(neighbor_df, on='UniqueID',how='left',copy=False,suffixes=('', '_y'))
test = test.merge(neighbor_df, on='UniqueID',how='left',copy=False,suffixes=('', '_y'))
del [train_midday,self_mean,df_dist,subtrain]

#-------------- Encoder for Categorical Cols: City,Month----------------- #
         # OneHot Encoder, for XgBoost
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(train[['City','Month','EntryType','ExitType']])
train = pd.concat([train,
                   pd.DataFrame(enc.transform(train[['City','Month','EntryType','ExitType']]).toarray(),
                               index=train.index,
                               columns=enc.get_feature_names().tolist())],axis=1)
test = pd.concat([test,
                   pd.DataFrame(enc.transform(test[['City','Month','EntryType','ExitType']]).toarray(),
                               index=test.index,
                               columns=enc.get_feature_names().tolist())],axis=1)


#-------------- Encoder for Categorical Cols: Hour ----------------- #
  # First add a baseline for hour
train['MorningPeak'] = ((train['Hour']>=7) & (train['Hour']<=9)).astype(int)
train['AfternoonPeak'] = ((train['Hour']>=15) & (train['Hour']<=17)).astype(int)
train['IsDay'] = ((train['Hour']>=6) & (train['Hour']<=20)).astype(int)
test['MorningPeak'] = ((test['Hour']>=7) & (test['Hour']<=9)).astype(int)
test['AfternoonPeak'] = ((test['Hour']>=15) & (test['Hour']<=17)).astype(int)
test['IsDay'] = ((test['Hour']>=6) & (test['Hour']<=20)).astype(int)
tmp = train.groupby(['Weekend','Hour','City'])[target_cols].mean().reset_index()
tmp.columns = ['Weekend','Hour','City','HourAve_T20','HourAve_T50','HourAve_T80','HourAve_D20','HourAve_D50','HourAve_D80']
test = test.merge(tmp, on=['Weekend','City','Hour'],how='left',copy=False,suffixes=('', '_y'))
train = train.merge(tmp, on=['Weekend','City','Hour'],how='left',copy=False,suffixes=('', '_y'))
train.drop(list(train.filter(regex='_y$')), axis=1, inplace=True)
test.drop(list(test.filter(regex='_y$')), axis=1, inplace=True)



#-------------- The percentage of certain intersection --------------#
def get_percentage_of_known_data(train,test):
  train['Count1'] = train.groupby(['UniqueID'])['IntersectionId'].transform('count')
  test['Count2'] = test.groupby(['UniqueID'])['IntersectionId'].transform('count')
  print("grouped")
  train = train.drop_duplicates(['UniqueID'])
  test = test.drop_duplicates(['UniqueID'])
  print("dropped")
  train = train.merge(test[['UniqueID','Count2']],on=['UniqueID'],how='left',copy=False,suffixes=('', '_y'))
  print("merged")
  train['CountPercent'] = train['Count1']/(train['Count1']+train['Count2'])*100
  train.fillna(0)
  train_count = train[['UniqueID','Latitude','Longitude','CountPercent']].to_numpy()
  print(train_count[:,3])
  plt.figure()
  #plt.scatter(train_count[:,1].tolist(),train_count[:,2].tolist(),s=train_count[:,3].tolist(),color='b',alpha=0.5)
  plt.plot(range(train_count.shape[0]),train_count[:,3].tolist())
  plt.show()
