import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
from plot_wells_df import plot_wells_df as pwdf
from extract_true import extract_true as et
from extract_median import extract_median as em

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
#MW_temp['UWI'] = MW_temp['UWI'].astype(str)

# Extract average Mud weight within depth window around BHTorMRT depths
fig, ax = plt.subplots()
_ = sns.scatterplot(x='Mud Wt',y = 'MW@Depth(KB)',data=MW_temp, palette = 'binary',alpha=0.4)
rg_list = [500,1000,1500]
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

# calculate temp diff of temp measurements relative to Time since circulation
master_df['dTemp(F)'] = master_df['True_Temp_BHT(F)']-master_df['BHTorMRT(F)']
print(master_df.columns)

# create training and validation subsets
training_df = master_df[master_df['Set']=='Training']
validation_df = master_df[master_df['Set']=='Validation_Testing']

# Check relationship of the temp difference versus TSC
_ = sns.regplot(x='dTemp(F)',y = 'TSCorORT(hrs)',data = master_df)
plt.show()

# plot temperature versus depth data
fig1, ax1 = plt.subplots(1,3,sharey=True,sharex=True)
_ = sns.scatterplot(x=training_df['True_Temp_BHT(F)'],y = training_df['BHT_SS(ft)'],alpha = 0.6,ax=ax1[0])
_ = sns.scatterplot(x=training_df['BHTorMRT(F)'],y = training_df['BHT_SS(ft)'],palette ='Paired' ,alpha = 0.6,ax=ax1[0])
#_ = sns.scatterplot(x=training_df['True2(C)'],y = training_df['Pressure Depth (m)'],alpha = 0.6,ax=ax1[1])
_ = sns.scatterplot(x=master_df['BHTorMRT(F)'],y = master_df['BHT_SS(ft)'],hue =master_df['Set'],palette = 'Set1',alpha = 0.6,ax=ax1[1])
_ = sns.scatterplot(x=master_df['True_TD(F)'],y = master_df['TD (ft)'],hue = master_df['Set'],palette ='Set2',alpha = 0.6,ax=ax1[2])
_ = sns.scatterplot(x=training_df['Static_Temp (F)'],y = training_df['TD (ft)'],alpha = 0.6,ax=ax1[2])
ax1[0].invert_yaxis()
plt.show()

#print(Static_Duvernay_out.head())
pwdf(master_df, 'SurfLat', 'SurfLong','UWI','TD (ft)','True_TD(F)','BHT_SS(ft)','True_Temp_BHT(F)',
                mapname='Texas_Data')