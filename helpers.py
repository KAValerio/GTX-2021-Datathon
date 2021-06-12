import numpy as np

def extract_median(df1,df1_depth,df2,df2_attr,df2_depth,df2_range,attr_out):
    '''
        Function to extract attribute value based on a median of windowed values
        Args: 
        df1(pandas.dataframe) - master dataframe with depths to extract attribute at a desired depth |
        df1_depth(str) - column name for depths from df1 |
        df2(pandas.dataframe) - dataframe with attribute depth profiles |
        df2_attr(str) - column name for attribute from df2 |
        df2_depth(str) - column name for depth from df2 |
        df2_range(int) - user selected range for calculation |
        attr_out(str) - column in df1 to write calculated value |
    '''

    for ind,row in df1.iterrows():
        X = row[df1_depth]
        depth_min = X - df2_range/2
        depth_max = X + df2_range/2
        try:
            df2[(df2[df2_depth] > depth_min) & (df2[df2_depth] < depth_max)][df2_attr]
        except:
            pass
        else:
            df2_subset = df2[(df2[df2_depth] > depth_min) & (df2[df2_depth] < depth_max)]
            a = df2_subset[df2_attr]
            Y = np.NaN if np.all(a!=a) else np.nanmean(a)
            df1.at[ind,attr_out] = Y
                
    return(df1)


def extract_true(df1,df2,ref,depth1,depth2,true):
    '''
        function to extract true temperatures for a given depth for every well
        Args: 
        df1(pandas.dataframe) - master data frame with depths to extract true temperature
        df2(pandas.dataframe) - true temperature profiles
        depth1(str) - column name for true temperature depth profiles       
        depth2(str) - column name for depth extraction
        true(str) -  column to write true temperature
    '''

    for ind,row in df1.iterrows():
        X = row[depth2]
        df2_subset = df2[df2[ref]==row[ref]]
        try:
            df_high = df2_subset[df2_subset[depth1] < X].iat[-1,1]
            df_low = df2_subset[df2_subset[depth1] > X].iat[0,1]
        except:
            pass
        else:
            df_high = df2_subset[df2_subset[depth1] < X]
            x1 = df_high.iat[-1,1]
            y1 = df_high.iat[-1,2]
            df_low = df2_subset[df2_subset[depth1] > X]
            x2 = df_low.iat[0,1]
            y2 = df_low.iat[0,2]
            Y = y1 + (y2-y1)/(x2-x1)*(X-x1)
            df1.at[ind,true] = Y
    
    return(df1)