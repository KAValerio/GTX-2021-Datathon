from re import sub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
from plot_wells_df import plot_wells_df as pwdf
from extract_true import extract_true as et
from extract_median import extract_median as em
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load static temperature data, considered highest confidence bottom hole temperature
Static_log_temp = pd.read_csv('./Data for Datathon/Data_static_logs.csv')
# split static by field, remove depth values. NB! Depth values will be replaced by TD depths
Static_Eagle_1 = Static_log_temp[Static_log_temp['Field']=='Eaglebine'].reset_index()
static_temp = Static_Eagle_1.rename(columns={'Well_ID':'UWI','Temp (degC)':'Static_Temp (F)'})
# convert Celsius to F
static_temp['Static_Temp (F)'] = static_temp['Static_Temp (F)']*9/5 + 32
static4merge = static_temp[['UWI','Static_Temp (F)']]

# Load BHT data. This is the measured data that will be used to predict bottom hole temperature
BHT_temp = pd.read_excel('./Data for Datathon/Eaglebine/Eaglebine BHT TSC data for SPE April 21 2020.xlsx')
master_df = BHT_temp.rename(columns={'BHT_below sea level (ft)':'BHT_SS(ft)','BHTorMRT (maximum recorded temperature) oF':'BHTorMRT(F)',
                                    'TSC or ORT (time since circulation or original recorded time in hours)':'TSCorORT(hrs)',})
master_df['UWI'] = master_df['UWI'].astype(str)

# Merge dataframes
master_df = master_df.merge(static4merge, on='UWI',how='left')

# Load set assignment file
set_assign = pd.read_csv('./Data for Datathon/set_assign.csv')
# merge set assigment with the well list
master_df = master_df.merge(set_assign, on='UWI', how='left').reset_index()

# Load prod data.
#Eagle_prod = pd.read_excel('./Data for Datathon/Eaglebine/EagleBine Casing production summary for SPE April21 2020.xlsx')

# Load casing data.
#Eagle_casing = pd.read_excel('./Data for Datathon/Eaglebine/EagleBine Casing production summary for SPE April21 2020.xlsx')

# Calculate TD in SS for TrueTemp extraction
master_df['TD SS(ft)'] = master_df['TD (ft)'] - master_df['GL(ft)']

# Load temperature-depth profiles, considered second source for true formation temperature
True_temp = pd.read_excel('./Data for Datathon/Eaglebine/Eaglebine TrueTemp_Train2.xlsx')
True_temp['UWI'] = True_temp['UWI'].astype(str)
True_temp = True_temp.rename(columns={'Depth sub-sea (feet)':'True_depth_SS(ft)','True Temperature   (oF)':'True_Temp(F)'})

# Extract True temperature at TD and BHTorMRT depths in SubSea
temp_list = [['True_Temp_BHT(F)','BHT_SS(ft)'],['True_TD(F)','TD SS(ft)']]
for temp in temp_list:
    master_df[temp[0]] = np.nan
    et(master_df,True_temp,'UWI','True_depth_SS(ft)',temp[1],temp[0])

# Load mudweight-depth profiles
MW_temp = pd.read_excel('./Data for Datathon/Eaglebine/Eaglebine mud weight SPE April 21 2021.xlsx')

# Extract em Mud weight within depth window around BHTorMRT depths
fig, ax = plt.subplots()
_ = sns.scatterplot(x='Mud Wt',y = 'MW@Depth(KB)',data=MW_temp, palette = 'binary',alpha=0.4)
rg_list = [500,1000,2000]
for range in rg_list:
    out = 'MW@BHT_'+str(range)
    master_df[out] = np.nan
    img = 0
    em(master_df,'BHT_ subsurface (ft)',MW_temp,'Mud Wt','MW@Depth(KB)',range,out)
    _ = sns.scatterplot(x=out,y = 'BHT_ subsurface (ft)',data=master_df)
    img +=1
ax.invert_yaxis()
rg_list.insert(0,'orig')
plt.legend(rg_list)
plt.show()

# create TrueTemp column from Static (primary) and True TD (secondary)
master_df['TrueTemp'] = master_df['Static_Temp (F)'].fillna(master_df['True_TD(F)'])

# calculate Waples corrected temps
Temp_surf = 70 # average annual surface temp estimation
# calculate f and see if it has a depth trend
master_df['f_Waples_calc'] = (master_df['True_Temp_BHT(F)'] - Temp_surf)/(master_df['BHTorMRT(F)'] - Temp_surf)
print(master_df.describe())

fig, ax = plt.subplots()
_ = sns.scatterplot(x='f_Waples_calc',y ='BHT_ subsurface (ft)',data=master_df, palette = 'binary',alpha=0.4)
rg_list = [100,500,1000]
for range in rg_list:
    out = 'Waples_'+str(range)
    master_df[out] = np.nan
    img = 0
    em(master_df,'BHT_ subsurface (ft)',master_df,'f_Waples_calc','BHT_ subsurface (ft)',range,out)
    _ = sns.scatterplot(x=out,y ='BHT_ subsurface (ft)',data=master_df)
    img +=1
ax.invert_yaxis()
rg_list.insert(0,'orig')
plt.legend(rg_list)
plt.show()

master_df['Temp_Waples_calc'] = master_df['Waples_1000']*(master_df['BHTorMRT(F)'] - Temp_surf) +  Temp_surf
# calculate temp diff of Waples calc and measured temp
master_df['dTemp(F)'] = master_df['Temp_Waples_calc']-master_df['BHTorMRT(F)']
print(master_df.columns)

# reorganize master df and export csv file 
export = master_df[['UWI', 'SurfLat', 'SurfLong','GL(ft)','TD (ft)','TD SS(ft)', # well info
                    'BHT_SS(ft)','BHT_ subsurface (ft)','BHTorMRT(F)','TSCorORT(hrs)', # raw data
                    'MW@BHT_2000','Temp_Waples_calc','dTemp(F)',  # calculated data
                    'Set','True_Temp_BHT(F)','True_TD(F)','Static_Temp (F)','TrueTemp']] # setclass and true data

export.to_csv('./Data for Datathon/Structured Data/Texas_master_df_v1.csv')

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

#print(Static_Duvernay_out.head())
pwdf(master_df, 'SurfLat', 'SurfLong','UWI','TD (ft)','True_TD(F)','BHT_SS(ft)','True_Temp_BHT(F)',
                mapname='./Data for Datathon/Structured Data/Texas_Data')