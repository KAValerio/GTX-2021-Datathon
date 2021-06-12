import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns

from helpers import extract_true as et
from helpers import extract_median as em

def compile_Duvernay_data(export_csv=False):

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

    if export_csv:
        export.to_csv('./Data for Datathon/Structured Data/Duvernay_master_df_v1.csv')
    
    return master_df




def compile_Eaglebine_data(export_csv=False):

    # Load static temperature data, considered highest confidence bottom hole temperature
    Static_log_temp = pd.read_csv('./Data for Datathon/Data_static_logs.csv')
    # split static by field, remove depth values. NB! Depth values will be replaced by TD depths
    Static_Eagle_1 = Static_log_temp[Static_log_temp['Field']=='Eaglebine'].reset_index()
    static_temp = Static_Eagle_1.rename(columns={'Well_ID':'UWI','Temp (degC)':'Static_Temp (F)'})
    # convert Celsius to F
    static_temp['Static_Temp (F)'] = static_temp['Static_Temp (F)']*9/5 + 32
    static4merge = static_temp[['UWI','Static_Temp (F)']]

    # Load BHT data. This is the measured data that will be used to predict bottom hole temperature
    BHT_temp = pd.read_excel('./Data for Datathon/Eaglebine/Eaglebine/Eaglebine BHT TSC data for SPE April 21 2020.xlsx')
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
    True_temp = pd.read_excel('./Data for Datathon/Eaglebine/Eaglebine/Eaglebine TrueTemp_Train2.xlsx')
    True_temp['UWI'] = True_temp['UWI'].astype(str)
    True_temp = True_temp.rename(columns={'Depth sub-sea (feet)':'True_depth_SS(ft)','True Temperature   (oF)':'True_Temp(F)'})

    # Extract True temperature at TD and BHTorMRT depths in SubSea
    temp_list = [['True_Temp_BHT(F)','BHT_SS(ft)'],['True_TD(F)','TD SS(ft)']]
    for temp in temp_list:
        master_df[temp[0]] = np.nan
        et(master_df,True_temp,'UWI','True_depth_SS(ft)',temp[1],temp[0])

    # Load mudweight-depth profiles
    MW_temp = pd.read_excel('./Data for Datathon/Eaglebine/Eaglebine/Eaglebine mud weight SPE April 21 2021.xlsx')

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

    if export_csv: 
        export.to_csv('./Data for Datathon/Structured Data/Eaglebine_master_df_v1.csv')
    
    return master_df