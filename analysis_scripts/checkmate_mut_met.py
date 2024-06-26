import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums

if __name__ == "__main__":
    # DNA Maf data for CheckMate 009, CheckMate 010, and CheckMate 025
    # Plasma metabolics data for CheckMate 009 and CheckMate 025
    # --> so Matched DNA and MET data only for CheckMate 009 and CheckMate 025
    data_path = "/juno/work/reznik/xiea1/MetabolicModel/data/CheckMate"
    data = pd.read_excel(f'{data_path}/checkmate_cohorts_maf.xlsx',
                         index_col=0, header=1)
    # exclude CM-010 cohort since there are no matched plasma metabolics data for this cohort
    data = data[data['Cohort'] != 'CM-010']
    data['Tumor_Sample_Barcode'] = np.nan
    data.loc[data['Cohort'] == 'CM-009', 'Tumor_Sample_Barcode'] = data.loc[data['Cohort'] == 'CM-009',
                                                                            'Matched_Norm_Sample_Barcode'].str[:-16]
    'MAF_Tumor_ID'



    # --------------------------------Start From Here---------------------------------------
    # Load Braunetal which includes CheckMate 009, 010, 025 clinical data (including BAP1 mutation status
    data_path = "/juno/work/reznik/xiea1/MetabolicModel/data/CheckMate"
    checkmate_clinical = pd.read_excel('/juno/work/reznik/xiea1/MIRTH/Other_RNA/clinical data/BraunEtAl.xlsx',
                                           sheet_name='Braun_et_al.FOLH1', index_col=0, header=0)
    checkmate_clinical = checkmate_clinical[checkmate_clinical['BAP1'].notna()]
    # subset to CheckMate 025 cohort
    checkmate_clinical_025 = checkmate_clinical[checkmate_clinical['Cohort'] == 'CM-025']
    # replace the string before '-' with 'CA209025'
    checkmate_clinical_025.index = checkmate_clinical_025.index.str.replace(r'(.*?)-', 'CA209025-', regex=True)
    # subset to CheckMate 009 cohort
    checkmate_clinical_009 = checkmate_clinical[checkmate_clinical['Cohort'] == 'CM-009']
    # replace the string before '_' with 'CA209009'
    checkmate_clinical_009.index = checkmate_clinical_009.index.str.replace(r'^.*?_', 'CA209009-', regex=True)


    # Load CheckMate 025 metabolomics data
    checkmate_025_met = pd.read_excel(f'{data_path}/checkmate_025_met.xlsx',
                                      sheet_name='baseline', index_col=0, header=0)
    # remove the string between the first and second '-' in the index
    checkmate_025_met.index = checkmate_025_met.index.str.replace(r'-(.*?)-', '-', regex=True)
    checkmate_025 = pd.concat([checkmate_clinical_025, checkmate_025_met], axis=1, join='inner')
    checkmate_025.to_csv(f'{data_path}/checkmate_025_baseline_met_mut.csv')

    # Load CheckMate 009 metabolomics data
    checkmate_009_met = pd.read_excel(f'{data_path}/checkmate_009_met.xlsx',
                                      sheet_name='baseline', index_col=0, header=0)
    # remove the string between the first and second '-' in the index
    checkmate_009_met.index = checkmate_009_met.index.str.replace(r'-(.*?)-', '-', regex=True)
    checkmate_009 = pd.concat([checkmate_clinical_009, checkmate_009_met], axis=1, join='inner')
    checkmate_009.to_csv(f'{data_path}/checkmate_009_baseline_met_mut.csv')

    # Check the level of plasma glucose in WT and MUT BAP1 patients
    # Boxplots and wilcoxon rank sum test
    df = checkmate_025
    ax = sns.boxplot(data=df, x="BAP1", y='glucose')
    ax = sns.stripplot(data=df, x="BAP1", y='glucose', jitter=True, alpha=0.5)
    plt.title('glucose level in WT and MUT BAP1 patients in CheckMate 025')
    plt.show()

    # no glucose, lactate, fumarate, citrate, malate data











