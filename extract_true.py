def extract_true(df1,df2,depth,true):
    '''
        Args: 
        df1(pandas.dataframe) - master data frame with depths to extract true temperature
        df2(pandas.dataframe) - true temperature profiles
        depth(str) - column name for depth extraction
        true(str) -  column to write true temperature
    '''

    for ind,row in df1.iterrows():
        X = row[depth]
        df2_subset = df2[df2['UWI']==row['UWI']]
        try:
            df_high = df2_subset[df2_subset['Depths subsea (m)'] < X].iat[-1,1]
            df_low = df2_subset[df2_subset['Depths subsea (m)'] > X].iat[0,1]
        except:
            pass
        else:
            df_high = df2_subset[df2_subset['Depths subsea (m)'] < X]
            x1 = df_high.iat[-1,1]
            y1 = df_high.iat[-1,2]
            df_low = df2_subset[df2_subset['Depths subsea (m)'] > X]
            x2 = df_low.iat[0,1]
            y2 = df_low.iat[0,2]
            Y = y1 + (y2-y1)/(x2-x1)*(X-x1)
            df1.at[ind,true] = Y
    
    return(df1)
