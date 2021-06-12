import matplotlib.pyplot as plt
import seaborn  as sns
from plot_wells_df import plot_wells_df as pwdf
from compile_data import compile_Duvernay_data

master_df = compile_Duvernay_data(export_csv = True)

# create training and validation
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

pwdf(master_df, 'Lat', 'Long','UWI','TD (m)','True_TD(C)','DST_mid_depth','True1(C)',
                'Pressure Depth (m)','True2(C)', mapname='./Data for Datathon/Structured Data/Duvernay_data')