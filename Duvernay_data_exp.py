import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
from plot_wells_df import plot_wells_df as pwdf
from extract_true import extract_true as et

# Load static temperature data, considered highest confidence bottom hole temperature
Static_log_temp = pd.read_csv('./Data for Datathon/Data_static_logs.csv')
# split static by field, remove depth values. NB! Depth values will be replaced by TD depths
Static_Duvernay_1 = Static_log_temp[Static_log_temp['Field']=='Duvernay'].reset_index()
static_temp = Static_Duvernay_1.rename(columns={'Well_ID':'UWI','Temp (degC)':'Static_Temp (C)'})
static4merge = static_temp[['UWI','Static_Temp (C)','Field']]

# Load well headers for x and y coordinates of the wells
Duvernay_well = pd.read_excel('./Data for Datathon/Duvernay/Duvernay/Duvernay well headers SPE April 21 2021 .xlsx')
wells4merge = Duvernay_well[['UWI ','SurfaceLatitude_NAD83','SurfaceLongitude_NAD83','Elevation Meters','TD meters ']]
wells4merge = wells4merge.rename(columns={'UWI ':'UWI','SurfaceLatitude_NAD83':'Lat','SurfaceLongitude_NAD83':'Long',
                                                          'Elevation Meters':'Elevation (m)','TD meters ':'TD (m)'})
# Add static temps to the well hdr dataframe
master_df = wells4merge.merge(static4merge, on='UWI',how='left')
master_df['Field'] = 'Duvernay'

# Load set assignment file
set_assign = pd.read_csv('./Data for Datathon/set_assign.csv')
# merge set assigment with the well list
master_df = master_df.merge(set_assign, on='UWI')

# Load Drill Stem Test data. This is the measured data that will be used to predict bottom hole temperature
Duvernay_DST = pd.read_excel('./Data for Datathon/Duvernay/Duvernay/Duvernay DST BHT for SPE April 20 2021.xlsx')
DST_temp = Duvernay_DST[['UWI','DST Start Depth (MD) (m)','DST End Depth (MD) (m)','DST Bottom Hole Temp. (degC)','Formation DSTd']]
DST_temp = DST_temp.rename(columns={'DST Start Depth (MD) (m)':'DST Start (m)','DST End Depth (MD) (m)':'DST End (m)','DST Bottom Hole Temp. (degC)':'DST Temp (C)',
                                'Formation DSTd':'DST Formation'})
DST_temp['DST_mid_depth'] = (DST_temp['DST End (m)']+DST_temp['DST Start (m)'])/2

# merge DST data with well file
master_df = master_df.merge(DST_temp, on='UWI')
# calculate Sub sea depth for true temp extraction
master_df['DST_mid_depth_SS'] = master_df['DST_mid_depth'] - master_df['Elevation (m)']

# Load prod data.
Duvernay_prod = pd.read_excel('./Data for Datathon/Duvernay/Duvernay/SPE Duvernay production summary April 20 2021.xlsx')
prod_temp = Duvernay_prod[['API   ','Spud Date   ','Last Production Month   ','Yield Total Average   ']].rename(columns={'API   ':'UWI',
                               'Spud Date   ':'Spud Date','Last Production Month   ':'Last Prod Year','Yield Total Average   ':'Mean Yield'})
master_df = master_df.merge(prod_temp, on='UWI')

# Load dst pressure and temperature data.
Duvernay_press = pd.read_excel('./Data for Datathon/Duvernay/Duvernay/Duvernay DST Pressures SPE May 2 2021.xlsx')
press_temp = Duvernay_press[['Well ID','Initial Hydrostatic Pressure (kPa)','Pressure Recorder Depth (m)',
                            'DST Bottom Hole Temp. (degC)','Formation DSTd']].rename(columns={'Well ID':'UWI', 'Pressure Recorder Depth (m)':'Pressure Depth (m)',
                            'Initial Hydrostatic Pressure (kPa)':'Init HS Press (kPa)',
                            'DST Bottom Hole Temp. (degC)':'DST_Temp2 (C)','Formation DSTd':'DST Formation2'})
master_df = master_df.merge(press_temp, on='UWI')

# Calculate Pressure depth in SS for TrueTemp extraction
master_df['Pressure Depth SS (m)'] = master_df['Pressure Depth (m)'] - master_df['Elevation (m)']
# Calculate TD in SS for TrueTemp extraction
master_df['TD SS(m)'] = master_df['TD (m)'] - master_df['Elevation (m)']

# Calculate the differences for DST depth and temperature
master_df['temp_diff'] = master_df['DST Temp (C)'] - master_df['DST_Temp2 (C)']
master_df['DST_depth_diff'] = master_df['DST_mid_depth'] - master_df['Pressure Depth (m)']

# Load temperature-depth profiles, considered second source for true formation temperature
Duvernay_True = pd.read_excel('./Data for Datathon/Duvernay/Duvernay/Duvenay TrueTemp_Train.xlsx')

# Extract True temperature at TD and DST depths in SubSea
temp_list = [['True1(C)','DST_mid_depth_SS'],['True2(C)','Pressure Depth SS (m)'],['True_TD(C)','TD SS(m)']]
for temp in temp_list:
    master_df[temp[0]] = np.nan
    et(master_df,Duvernay_True,'UWI','Depths subsea (m)',temp[1],temp[0])

# create TrueTemp column from Static (primary) and True_TD (Secondary)
master_df['TrueTemp'] = master_df['Static_Temp (C)'].fillna(master_df['True_TD(C)'])
print(master_df.describe())

# reorganize master df and export csv file 
export = master_df[['UWI','Lat','Long','Elevation (m)','Spud Date','TD (m)','TD SS(m)', # well info
                    'DST_mid_depth','DST_mid_depth_SS','DST Temp (C)','DST Formation', # DST Temp raw data
                    'Pressure Depth (m)','Pressure Depth SS (m)','DST_Temp2 (C)','DST Formation2', # DST Pressure raw data
                    'Last Prod Year', 'Mean Yield', 'Init HS Press (kPa)', # Production raw data
                     'Set','True1(C)','True2(C)','True_TD(C)','Static_Temp (C)','TrueTemp']] #setclass and true data

#export.to_csv('./Data for Datathon/Structured Data/Canada_master_df_v1.csv')

# create training and validat
training_df = master_df[master_df['Set']=='Training']
validation_df = master_df[master_df['Set']=='Validation_Testing']


# plot
fig1, ax1 = plt.subplots(1,3,sharey=True,sharex=True)

_ = sns.scatterplot(x=training_df['True1(C)'],y = training_df['DST_mid_depth'],alpha = 0.6,ax=ax1[0])
_ = sns.scatterplot(x=training_df['DST Temp (C)'],y = training_df['DST_mid_depth'],palette ='Paired' ,alpha = 0.6,ax=ax1[0])
#_ = sns.scatterplot(x=training_df['True2(C)'],y = training_df['Pressure Depth (m)'],alpha = 0.6,ax=ax1[1])
_ = sns.scatterplot(x=master_df['DST_Temp2 (C)'],y = master_df['Pressure Depth (m)'],hue =master_df['Set'],palette = 'Set1',alpha = 0.6,ax=ax1[1])
legend3 = ['True','Static']
_ = sns.scatterplot(x=master_df['True_TD(C)'],y = master_df['TD (m)'],alpha = 0.6,ax=ax1[2])
_ = sns.scatterplot(x=master_df['Static_Temp (C)'],y = master_df['TD (m)'],alpha = 0.8,ax=ax1[2])
plt.legend(legend3)
ax1[0].invert_yaxis()
plt.show()

#print(Static_Duvernay_out.head())
pwdf(master_df, 'Lat', 'Long','UWI','TD (m)','True_TD(C)','DST_mid_depth','True1(C)',
                'Pressure Depth (m)','True2(C)', mapname='./Data for Datathon/Structured Data/Canada_data')