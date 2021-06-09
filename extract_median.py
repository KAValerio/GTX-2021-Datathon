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
    import numpy as np

    for ind,row in df1.iterrows():
        X = row[df1_depth]
        depth_min = X - df2_range/2
        depth_max = X + df2_range/2
        try:
            df2[(df2[df2_depth] > depth_min) & (df2[df2_depth] < depth_max)]
        except:
            pass
        else:
            df2_subset = df2[(df2[df2_depth] > depth_min) & (df2[df2_depth] < depth_max)]
            Y = np.median(df2_subset[df2_attr])
            df1.at[ind,attr_out] = Y
    
    return(df1)
