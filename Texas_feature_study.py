import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re

# import master_df csv
master_df = pd.read_csv('./Data for Datathon/Structured Data/Texas_master_df_v1.csv')

# create training and validation subsets
training_df = master_df[master_df['Set']=='Training']

# remove [, ], <
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
training_df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in training_df.columns.values]

train1 = training_df.drop(['UWI','Set','True_Temp_BHT(F)','True_TD(F)','Static_Temp (F)','TrueTemp'], axis=1)
# Encoding Categorical features, if any.
cat_columns = train1.select_dtypes(include=['object']).columns
for column in tqdm(cat_columns):
    le = LabelEncoder()
    train1[column] = le.fit_transform(train1[column])
#print(list(train1))
#print(train1.head(10))​
xgr = xgb.XGBRegressor(random_state=42)

reg_xgr = xgr.fit(train1, training_df['TrueTemp'].values.reshape(-1, 1))
n_features = train1.shape[1]
plt.figure(figsize=(10,8))
plt.barh(range(n_features), xgr.feature_importances_, align='center')
plt.yticks(np.arange(n_features), list(train1))
plt.xlabel("Feature Importance For TrueTemp")
plt.ylabel("Feature")
plt.show()

reg_xgr = xgr.fit(train1, training_df['True_TD(F)'].values.reshape(-1, 1))
n_features = train1.shape[1]
plt.figure(figsize=(10,8))
plt.barh(range(n_features), xgr.feature_importances_, align='center')
plt.yticks(np.arange(n_features), list(train1))
plt.xlabel("Feature Importance For True_TD")
plt.ylabel("Feature")
plt.show()