import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import scipy.stats
from statsmodels.stats.multitest import multipletests
import numpy as np
import statsmodels.api as sm
import seaborn as sns

if __name__ == '__main__':
    # ------------------------MITO ccRCC: isotopologue ~ metastasis------------------------
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/Isotope_raw_mito'
    data = pd.read_excel(f'{data_path}/isotope_glucose.xlsx', sheet_name='Tissue Isotopologues',
                         index_col=0, header=0)
    data = data.drop([np.nan])
    data = data.T
    data.index = [data['Patient ID'][i] + '_' + data['Tissue Type'][i] for i in range(data.shape[0])]
    # subset data to only ccRCC and ccRCC metastasis samples
    data = data.loc[data['Tissue Type'].isin(['Tumor_Primary', 'Tumor_Met'])]
    data = data.loc[~data['Subtype'].isin(['HLRCC', 'Papillary', 'Oncocytoma',
                                            'Chromophobe'])]
    data['metastasis'] = data['Other Notation'].apply(lambda x: 'Primary' if pd.isna(x) else 'Metastasis')
    # Normalize the original data by pyruvate m+3
    raw_data = data.iloc[:, 5:-1]
    raw_data = raw_data[raw_data['Pyruvate m+3'].notna()]
    normalized_data = pd.DataFrame(raw_data.to_numpy()/(raw_data['Pyruvate m+3'].to_numpy().reshape(raw_data.shape[0], -1)),
                                   index=raw_data.index, columns=raw_data.columns)
    mean_data = pd.concat([normalized_data, data[['Patient ID', 'Tissue Type',
                                                  'Subtype', 'Other Notation', 'metastasis']]],axis=1)

    # make box plots of isotopologue abundance by metastasis status
    # x: 'Subtype', y: 'Citrate m+2'
    mean_data = test.loc[test['Subtype'].isin(['CCRCC', 'Liver', 'Colon', 'Lung',
                                               'Brain', 'Adrenal', 'Lymph Node'])]
    mean_data['metastasis'] = mean_data['Subtype'].apply(lambda x: 'Metastasis' if x != 'CCRCC' else 'Primary')

    isotope = 'Citrate m+2'
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.boxplot(x="metastasis", y=isotope, data=mean_data, ax=ax)
    # differential abundance test of 2 groups (metastasis, non-metastasis)
    statistic, p_value = scipy.stats.ranksums(mean_data.loc[mean_data['metastasis'] == 'Metastasis', isotope],
                                               mean_data.loc[mean_data['metastasis'] == 'Primary', isotope])
    # add p values of the Kruskal test to the plot
    ax.text(0.05, 1, f'p={p_value:.2e}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    sns.stripplot(x="metastasis", y=isotope, data=mean_data, ax=ax, jitter=True, alpha=1, color='black')
    ax.set_title(isotope)
    plt.tight_layout()
    plt.show()




    # ------------------------Imputed ccRCC (clinical trials): isotopologue ~ metastasis------------------------
    # file_path = "/juno/work/reznik/xiea1/MetabolicModel"
    file_path = "/data1/resnike/xiea1/MetabolicModel"

    # (1) ---------------------------------------- BraunEtAl ----------------------------------------
    sub_dir = 'BraunEtAl'
    results_dir = f'{file_path}/results_RNA_imputation/{sub_dir}'
    embedding_dir = f'{results_dir}/embeddings'
    clinical_path = f'{file_path}/data/clinical_data'
    len_samples = 311
    # Load the predicted MET_RNA matrix
    imputed_mean_met = pd.read_csv(f'{results_dir}/{sub_dir}_isotope_imputed_mean_met_flipped.csv',
                                   index_col=0, header=0)
    # Load the clinical data
    clinical = pd.read_excel(f'{clinical_path}/{sub_dir}.xlsx', sheet_name="Braun_et_al.FOLH1", index_col="RNA_ID")
    clinical = clinical.iloc[:len_samples, :]
    # Merge the data
    data = pd.concat([imputed_mean_met, clinical], axis=1, join='inner')
    data = data.iloc[:len_samples, :]
    data = data[data['Tumor_Sample_Primary_or_Metastas'].notna()]
    # reorder the rows by metastasis status, having primary samples first
    data = data.sort_values(by=['Tumor_Sample_Primary_or_Metastas'], ascending=False)


    metastasis_col_name = 'Tumor_Sample_Primary_or_Metastas'
    primary_name = 'PRIMARY'
    metastasis_name = 'METASTASIS'

    # ---------------------------------------- IMmotion151 ----------------------------------------
    sub_dir = 'IMmotion151_isotope'
    results_dir = f'{file_path}/results_RNA_imputation/{sub_dir}'
    embedding_dir = f'{results_dir}/embeddings'
    clinical_path = f'{file_path}/data/clinical_data'
    len_samples = 823
    # Load the predicted MET_RNA matrix
    imputed_mean_met = pd.read_csv(f'{results_dir}/{sub_dir}_isotope_imputed_mean_met_flipped.csv',
                                   index_col=0, header=0)
    # Load the clinical data
    clinical = pd.read_csv(f'{clinical_path}/IMmotion151.csv', header=0, index_col='RNASEQ_SAMPLE_ID')
    clinical = clinical.iloc[:len_samples, :]
    # Merge the data
    data = pd.concat([imputed_mean_met, clinical], axis=1, join='inner')
    data = data.iloc[:len_samples, :]
    data.to_csv(f'{results_dir}/{sub_dir}_imputed_mean_met_flipped_clinical.csv')

    metastasis_col_name = 'PRIMARY_VS_METASTATIC'
    primary_name = 'PRIMARY'
    metastasis_name = 'METASTATIC'



    # make boxplot comparing the abundance of isotopologues between metastasis and primary
    isotope = 'Lactate m+3'
    fig, ax = plt.subplots(figsize=(5, 5))
    # change the order of the x-axis in the box plot
    sns.boxplot(x=metastasis_col_name, y=isotope, data=data, ax=ax)

    # differential abundance test of 2 groups (metastasis, non-metastasis)
    statistic, p_value = scipy.stats.ranksums(data.loc[data[metastasis_col_name] == metastasis_name, isotope],
                                               data.loc[data[metastasis_col_name] == primary_name, isotope])
    # add p values of the Kruskal test to the plot
    ax.text(0.05, 1, f'p={p_value:.2e}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    sns.stripplot(x=metastasis_col_name, y=isotope, data=data, ax=ax, jitter=True, alpha=1, color='black')
    ax.set_title(isotope)
    plt.tight_layout()
    plt.show()
