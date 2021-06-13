# This file is based on the the Data_wrangling_GtX ipython notebook provided by the competition organizers
# There has been some refactoring to allow the file to run in this project file structure
import lasio
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json

# set paths of different subdirectories
dir_eaglebine = 'Data for Datathon/Eaglebine/Eaglebine/'
dir_duvernay = 'Data for Datathon/Duvernay/Duvernay/'
dir_wrangled_data = 'Data for Datathon/Wrangled Data/'
dir_model_data = 'Data for Datathon/Model Data/'

# Load all files at once into las and las_df to save time
folder = 'Data for Datathon/clean_las_files/Clean_LAS/'
all_files = os.listdir(folder)
n_files = len(all_files)
print("Number of LAS files found: ", n_files)


las = {}
las_df = {}
mnemonics = {}
curves = dict()

unit_mapping = {
    'M' : 'm',

    'FT': 'ft',
    'F':  'ft',

    'MM': 'mm',

    'OHMM':'ohmm',
    'MMho': 'MMHO',

    'KG':'Kg',

    'US/F':'US/FT',

    'V/':'V/V',

    'UG_RA-EQ/TON': 'UGRA-EQ/TON',
    'UG-RA-EQ/T': 'UGRA-EQ/TON',
    'UgRa-Eq/TON': 'UGRA-EQ/TON',
    'UgRa-Eq/T': 'UGRA-EQ/TON',
    'RAEQ/TON' : 'UGRA-EQ/TON',

    'K/M3' : 'KG/M3',
    'KG/M2' : 'KG/M3',

    'US/M' : 'us/m',
    'US' : 'us',

    'INI': 'IN',

    'LB' : 'lbs',
    'LBS': 'lbs',

    'DEC': 'dec',

    'N/A':'Unknown',
    '':'Unknown',
    '0':'Unknown',
    'UNITLESS':'Unknown',
    'none':'Unknown',
    'None':'Unknown'
    }

mnemonic_replacements = {
    'SPONTANEOUS POTENTIAL': {'SPED', 'SPR', 'SPR_1', 'SPM', 'SPRRED', 'SPSED', 'SP', 'SPR2', 'SPR2ED', 'SPRED', 'SPWS', 'SPR1ED', 'SPR_R', 'SPS', 'SPR1', 'SPMED', 'DTWS', 'SPR_1ED', 'RILMED'},
    'DENSITY POROSITY': {'DPOSSWS', 'DPHILS', 'PORZ', 'DPORSSED', 'DPHI_SS1', 'DPHI_SSED', 'DPHI_SS', 'DPOR', 'DPORSS', 'PORZ_LS', 'PORZ_SS', 'DPO_LS', 'PORD_SS', 'DPHZSS', 'PORZCLSED', 'DPHZ', 'CALS', 'DPHIED', 'PORDLSED', 'DPHIDMED', 'PORDED', 'DPHI_LSED', 'DPORLS', 'DPOR_DM', 'DPOR_LS', 'DPHI1ED', 'DPHZSSED', 'DPHZLSED', 'PORZLSED', 'NPHISS', 'DPHISSED', 'DPHISS', 'DPHI_DM', 'DPHI_LS_R', 'PORZSSED', 'DPHZED', 'CDLLSED', 'PORD_LS', 'DPHI_LS', 'DPHI', 'DPHZ_SS', 'PORZED', 'DPHILSRED', 'DPORLSED', 'DPOWS', 'PORD', 'CDL_LS', 'DPOR_SS', 'DPOLSED', 'DPHZLS', 'DPHILSED', 'DPOLSWS', 'CDL_SS', 'DPHI_1', 'PORDSSED', 'DPHIDM', 'DPHZ_LS', 'DPHIDLED', 'DPORDMED', 'CDL', 'PHND_LS', 'DPHISS1ED', 'DENWS', 'DPHI_SAN', 'CDLSSED'},
    'DENSITY CORRECTION': {'DCOR1', 'DRHO', 'ZCORED', 'DRH', 'DCORED', 'DRHO_R', 'DRHORED', 'ZCOR', 'DRHOED', 'DRHED', 'HDRA', 'DCORWS', 'DRHO1', 'CORRED', 'HDRAED', 'DCORR', 'DRHO1ED', 'CORR', 'DCOR'},
    'PHOTOELECTRIC FACTOR': {'PEFWS', 'PEF', 'PEFZED', 'PEED', 'PEFLED', 'PEFL', 'PE', 'PEFED', 'PEFZ'},
    'GAMMA RAY': {'SGRDDED', 'GRT', 'CGRDED', 'GRS1', 'GRKT', 'GRD1ED', 'GRS1ED', 'GRN2ED', 'GRN2', 'GRSRED', 'GR_1', 'GRS_R', 'GRD1', 'GRD_1', 'GRAXED', 'GRNED', 'GRN_2', 'GRTH', 'GR1ED', 'GRSED', 'GRR_1', 'GR_2', 'GRR_R', 'GRN1', 'SGRDD', 'GRDRED', 'GRDED', 'GRM', 'SGRDED', 'GRKTED', 'GRR1', 'GRMED', 'GRRRED', 'GRR', 'GRN1ED', 'GRTED', 'GRKUTED', 'GRTHED', 'GRN_1', 'GRKUT', 'GRR_1ED', 'GRN3', 'GRD', 'GRS_1', 'GRD_R', 'SGRD', 'GRR1ED', 'GRWS', 'GRN', 'GRED', 'GR_D3', 'GRS', 'GRAX', 'GR', 'GRRED'},
    'NEUTRON POROSITY': {'CNCED', 'NPORLS', 'CNCLSED', 'NPHI_SS', 'NPHI_DM', 'NPHI_LS_R', 'NPOWS', 'CNCF', 'NPORSSED', 'CNCSSED', 'NPORSS', 'NPHI_SS_1', 'NPHILSRED', 'NPHILS', 'CNC', 'CNLLSED', 'CNCFED', 'NPHISSED', 'NPOLSWS', 'NPHIED', 'NPOR', 'NEUT1', 'NPHISS', 'NPHI_LSED', 'CNC_SS', 'CNS', 'NPODLWS', 'NPOR_SS', 'CNSLSED', 'CNSSSED', 'CN', 'CNLSED', 'NPHI', 'NPHI_LS', 'CNSSED', 'NPHI_SS1', 'CNC_LS', 'RHOB', 'NPOR_DM', 'CNCFLSED', 'NEUTWS', 'NPOR_LS', 'NPHIDM', 'CNL_LS', 'CNS_LS', 'NPORDMED', 'NPHILSED', 'NPHI_SSED', 'NPHISS1ED', 'CNCF_LS', 'NPHIDMED', 'NPORLSED', 'NPOR_LIM', 'NPOSSWS', 'NPHIDLED'},
    'TENSION': {'TENR1ED', 'TENMED', 'TENR', 'TENS1ED', 'TENED', 'TENRRED', 'TENSRED', 'TENR_1', 'TENM', 'TEN', 'TENT', 'TENS_R', 'TEND', 'TENTED', 'TENDED', 'TENS', 'TENSED', 'TENR_R', 'TENRED'},
    'BULK DENSITY': {'DRHO', 'RHOB1ED', 'DEN', 'ZDEN', 'DRH', 'DENED', 'DRHOED', 'RHOZED', 'ZDENED', 'DRHED', 'RHOZ', 'DCORWS', 'RHOB_R', 'RHOBED', 'DRHO1', 'DRHO_1', 'RHOB', 'RHOBRED', 'RHOB1', 'DENWS'},
    }

mnemonic_mapping = {
    'GRNED' : 'GRWS',
}

for key in mnemonic_replacements.keys():
    mnemonic_mapping = { **mnemonic_mapping, **{value : key for value in mnemonic_replacements[key]}}

expressions=['SPONTANEOUS POTENTIAL', 'DENSITY POROSITY', 'DENSITY CORRECTION','PHOTOELECTRIC FACTOR', 'GAMMA RAY', 'NEUTRON POROSITY','TENSION','BULK DENSITY']
expression_dict = {x : set() for x in expressions}


i = 0
for filename in os.listdir(folder):
    i = i + 1
    if filename.endswith(".LAS"):
        las[filename] = lasio.read(folder + '/' + filename)

        for curve in las[filename].curves:
            unit = curve.unit
            unit = unit if unit not in unit_mapping else unit_mapping[unit]
            mnemonic = curve.mnemonic
            mnemonic = mnemonic if mnemonic not in mnemonic_mapping else mnemonic_mapping[mnemonic]
            descr = curve.descr

            for expression in expression_dict.keys():
                if descr.startswith(expression): expression_dict[expression].update([curve.mnemonic])


            if unit in curves:
                if mnemonic in curves[unit]:
                    if descr not in curves[unit][mnemonic]:
                        curves[unit][mnemonic].append(descr)
                else:
                    curves[unit][mnemonic] = [descr]
            else:
                curves[unit] = {mnemonic : [descr]}

        las_df[filename] = las[filename].df()
        las_df[filename]['well_name'] = filename.split('_')[0]
        mnemonics[filename] = las_df[filename].columns


# find out which well curves/logs are in each las file
listDF = []
for filename in all_files:
    df = pd.DataFrame(columns=list(mnemonics[filename]), data=np.ones(
        (1, len(mnemonics[filename]))))
    df['well_name'] = filename.split('_')[0]
    listDF.append(df)


log_table = pd.concat(listDF)

# Here we can see which logs are in each well
print(log_table.head())
print(log_table.describe())
print(sorted(curves.keys()))


with open(dir_wrangled_data + 'curves.json','w') as outfile:
    json.dump(curves, outfile, sort_keys=True, indent=4)

# see what are the most common log types
sumT = log_table.drop(columns=['well_name']).sum()
print(sumT.sort_values(ascending=False))

# make a table of the log types available per well
for filename in all_files:
    las_df[filename] = las_df[filename].rename_axis('DEPT').reset_index()
