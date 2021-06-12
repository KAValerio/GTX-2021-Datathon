from re import sub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from plot_wells_df import plot_wells_df as pwdf
from compile_data import compile_Eaglebine_data


master_df = compile_Eaglebine_data(export_csv=True)
# Check relationship of the temp difference versus TSC
# _ = sns.regplot(x='dTemp(F)',y = 'TSCorORT(hrs)',data = master_df)
# plt.show()

# plot temperature versus depth data
fig1, ax1 = plt.subplots(1,3,sharey=True,sharex=True)
_ = sns.scatterplot(x=master_df['True_Temp_BHT(F)'],y = master_df['BHT_SS(ft)'],alpha = 0.5,ax=ax1[0])
_ = sns.scatterplot(x=master_df['BHTorMRT(F)'],y = master_df['BHT_SS(ft)'],alpha = 0.7,ax=ax1[0])
_ = sns.scatterplot(x=master_df['Temp_Waples_calc'],y = master_df['BHT_SS(ft)'],alpha = 0.7,ax=ax1[0])
plt.legend(['True','Raw','Waple'])
#_ = sns.scatterplot(x=training_df['True2(C)'],y = training_df['Pressure Depth (m)'],alpha = 0.6,ax=ax1[1])
_ = sns.scatterplot(x=master_df['BHTorMRT(F)'],y = master_df['BHT_SS(ft)'],hue =master_df['Set'],palette = 'Set1',alpha = 0.6,ax=ax1[1])
legend3 = ['True','Static']
_ = sns.scatterplot(x=master_df['True_TD(F)'],y = master_df['TD (ft)'],alpha = 0.6,ax=ax1[2])
_ = sns.scatterplot(x=master_df['Static_Temp (F)'],y = master_df['TD (ft)'],alpha = 0.8,ax=ax1[2])
plt.legend(legend3)
ax1[0].invert_yaxis()
plt.show()

pwdf(master_df, 'SurfLat', 'SurfLong','UWI','TD (ft)','True_TD(F)','BHT_SS(ft)','True_Temp_BHT(F)',
                mapname='./Data for Datathon/Structured Data/Eaglebine_Data')