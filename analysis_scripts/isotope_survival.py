import pandas as pd
import scipy.stats
from statistics import mean
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests
from Performance_Benchmarking.scripts.utils import re_rank_2D

if __name__ == '__main__':
    def get_key(val):
        for key, value in met_map.items():
            if val == value:
                return key
        return "key doesn't exist"


    file_path = "/juno/work/reznik/xiea1/MetabolicModel"
    # Get the confident predicted isotopologues
    metabolites = pd.read_csv(f'{file_path}/results_RNA_MITO/MITO1/median_rho_feature.csv', header=0, index_col=0)
    #met = []
    #met.extend(metabolites['feature'])
    #met.extend(metabolites.loc[(metabolites['median_rho'] > 0) & (metabolites['sig_in_most'] == True), 'feature'])
    met = ['Glucose m+6', '3-Phosphoglycerate m+3', 'Lactate m+3', 'Alanine m+3',
                                'Citrate m+2', 'Succinate m+2', 'Malate m+2',
                                'Glutamate m+2', 'Aspartate m+2']
    met_map = {s: f'{i}a' for i, s in enumerate(met)}

    # All isotope labelling data are imputed by the 46 ccRCC samples from the MITO paper

    # ---------------------------------------- BraunEtAl ----------------------------------------
    sub_dir = 'BraunEtAl'
    results_dir = f'{file_path}/results_RNA_imputation/{sub_dir}'
    embedding_dir = f'{results_dir}/embeddings'
    clinical_path = f'{file_path}/data/clinical_data'
    len_samples = 311

    # Load the predicted MET_RNA matrix
    rank_hat_draws_mean = np.load(f'{embedding_dir}/rank_hat_draws_mean_met.npy')[:len_samples, :]
    # rerank metabolite ranks in the flipped way (now rank 0 means the smallest value, # 1 means the largest value)
    rank_hat_draws_mean = re_rank_2D(rank_hat_draws_mean)
    rank_hat_draws_mean = rank_hat_draws_mean / rank_hat_draws_mean.shape[0]  # transformed all ranks to [0,1)
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                  index_col=0, header=0).iloc[:len_samples, :rank_hat_draws_mean.shape[1]]
    imputed_mean_met = pd.DataFrame(rank_hat_draws_mean, index=normalized_data_met.index, columns=normalized_data_met.columns)
    imputed_mean_met.to_csv(f'{results_dir}/{sub_dir}_isotope_imputed_mean_met_flipped.csv')
    imputed_mean_met = imputed_mean_met.rename(columns=met_map)

    # Load the clinical data
    clinical = pd.read_excel(f'{clinical_path}/{sub_dir}.xlsx', sheet_name="Braun_et_al.FOLH1", index_col="RNA_ID")
    clinical = clinical.iloc[:len_samples, :]
    #clinical = clinical.loc[clinical['Tumor_Sample_Primary_or_Metastas'] == 'PRIMARY']
    #imputed_mean_met = imputed_mean_met.loc[clinical.index, :]
    #imputed_mean_met.to_csv(f'{results_dir}/{sub_dir}_isotope_imputed_mean_met_flipped_primary.csv')

    # Merge the data
    data = pd.concat([imputed_mean_met, clinical], axis=1, join='inner')
    data = data.iloc[:len_samples, :]
    data = data[data['Age'].notna()]
    #data = data[data['Tumor_Sample_Primary_or_Metastas'].notna()]
    data.loc[data['Sex']== 'Male','Sex'] = 'M'
    data.loc[data['Sex'] == 'MALE', 'Sex'] = 'M'
    data.loc[data['Sex'] == 'Female', 'Sex'] = 'F'
    data.loc[data['Sex'] == 'FEMALE', 'Sex'] = 'F'

    # Merge datasets together
    cols = []
    cols.extend(met_map.values())
    cols.extend(['Age', 'Sex', 'OS', 'OS_CNSR', 'PFS', 'PFS_CNSR', 'Arm'])
    survival_df = data[cols].copy()
    survival_df = survival_df.rename({'Age': 'AGE'}, axis=1)
    survival_df = survival_df.rename({'Sex': 'SEX'}, axis=1)
    survival_df.loc[survival_df['PFS_CNSR'] == 0, 'PFS_EVENT'] = 1
    survival_df.loc[survival_df['PFS_CNSR'] == 1, 'PFS_EVENT'] = 0
    survival_df.loc[survival_df['OS_CNSR'] == 0, 'OS_EVENT'] = 1
    survival_df.loc[survival_df['OS_CNSR'] == 1, 'OS_EVENT'] = 0
    survival_df = survival_df.rename({'OS': 'OS_MO'}, axis=1)
    survival_df = survival_df.rename({'PFS': 'PFS_MO'}, axis=1)
    survival_df = survival_df.drop(columns=['PFS_CNSR', 'OS_CNSR'])
    survival_df['Dataset'] = 'BraunEtAl'
    survival_df_add_T = survival_df.loc[survival_df['Arm']=='NIVOLUMAB']
    survival_df_add_C = survival_df.loc[survival_df['Arm']=='EVEROLIMUS']
    #survival_df_add_T.to_csv(f'{clinical_path}/treatment_arm/BraunEtAl.csv')
    #survival_df_add_C.to_csv(f'{clinical_path}/control_arm/BraunEtAl.csv')

    # ---------------------------------------- javelin_101 ----------------------------------------
    sub_dir = 'javelin_101'
    results_dir = f'{file_path}/results_RNA_imputation/{sub_dir}'
    embedding_dir = f'{results_dir}/embeddings'
    clinical_path = f'{file_path}/data/clinical_data'
    len_samples = 726

    # Load the predicted MET_RNA matrix
    rank_hat_draws_mean = np.load(f'{embedding_dir}/rank_hat_draws_mean_met.npy')[:len_samples, :]
    # rerank metabolite ranks in the flipped way (now rank 0 means the smallest value, # 1 means the largest value)
    rank_hat_draws_mean = re_rank_2D(rank_hat_draws_mean)
    rank_hat_draws_mean = rank_hat_draws_mean / rank_hat_draws_mean.shape[0]  # transformed all ranks to [0,1)
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                  index_col=0, header=0).iloc[:len_samples, :rank_hat_draws_mean.shape[1]]
    imputed_mean_met = pd.DataFrame(rank_hat_draws_mean, index=normalized_data_met.index, columns=normalized_data_met.columns)
    imputed_mean_met.to_csv(f'{results_dir}/{sub_dir}_isotope_imputed_mean_met_flipped.csv')
    imputed_mean_met = imputed_mean_met.rename(columns=met_map)

    # Load the clinical data
    clinical = pd.read_excel(f'{clinical_path}/{sub_dir}.xlsx', sheet_name="all_clinical_data",index_col='ID')

    # Merge the data
    data = pd.concat([imputed_mean_met, clinical], axis=1, join='inner')
    data = data.iloc[:len_samples, :]

    data = data.rename({'PFS_P':'PFS_MO'},axis=1)
    data.loc[data['PFS_P_CNSR'] == 0, 'PFS_EVENT'] = 1
    data.loc[data['PFS_P_CNSR'] == 1, 'PFS_EVENT'] = 0
    cols = []
    cols.extend(met_map.values())
    cols.extend(['AGE','SEX','PFS_MO','PFS_EVENT','TRT01P'])
    survival_df_add = data[cols].copy()
    survival_df_add['Dataset'] = 'javelin_101'
    #survival_df = pd.concat([survival_df, survival_df_add], axis=0)
    survival_df_add_T = survival_df_add.loc[survival_df_add['TRT01P']=='Avelumab+Axitinib']
    survival_df_add_C = survival_df_add.loc[survival_df_add['TRT01P']=='Sunitinib']
    #survival_df_add_T.to_csv(f'{clinical_path}/treatment_arm/javelin_101.csv')
    #survival_df_add_C.to_csv(f'{clinical_path}/control_arm/javelin_101.csv')

    # ---------------------------------------- IMmotion151 ----------------------------------------
    sub_dir = 'IMmotion151_isotope'
    results_dir = f'{file_path}/results_RNA_imputation/{sub_dir}'
    embedding_dir = f'{results_dir}/embeddings'
    clinical_path = f'{file_path}/data/clinical_data'
    len_samples = 823

    # Load the predicted MET_RNA matrix
    rank_hat_draws_mean = np.load(f'{embedding_dir}/rank_hat_draws_mean_met.npy')[:len_samples, :]
    # rerank metabolite ranks in the flipped way (now rank 0 means the smallest value, # 1 means the largest value)
    rank_hat_draws_mean = re_rank_2D(rank_hat_draws_mean)
    rank_hat_draws_mean = rank_hat_draws_mean / rank_hat_draws_mean.shape[0]  # transformed all ranks to [0,1)
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                      index_col=0, header=0).iloc[:len_samples, :rank_hat_draws_mean.shape[1]]
    imputed_mean_met = pd.DataFrame(rank_hat_draws_mean, index=normalized_data_met.index,
                                    columns=normalized_data_met.columns)
    imputed_mean_met.to_csv(f'{results_dir}/{sub_dir}_isotope_imputed_mean_met_flipped.csv')
    imputed_mean_met = imputed_mean_met.rename(columns=met_map)


    # Load the clinical data
    clinical = pd.read_csv(f'{clinical_path}/IMmotion151.csv', header=0, index_col='RNASEQ_SAMPLE_ID')
    clinical = clinical.iloc[:len_samples, :]
    #clinical = clinical.loc[clinical['PRIMARY_VS_METASTATIC'] == 'PRIMARY']
    #imputed_mean_met = imputed_mean_met.loc[clinical.index, :]
    #imputed_mean_met.to_csv(f'{results_dir}/{sub_dir}_isotope_imputed_mean_met_flipped_primary.csv')

    # Merge the data
    data = pd.concat([imputed_mean_met, clinical], axis=1, join='inner')
    data = data.iloc[:len_samples, :]

    data = data.rename({'PFS_MONTHS': 'PFS_MO'}, axis=1)
    data.loc[data['PFS_CENSOR'] == 0, 'PFS_EVENT'] = 1
    data.loc[data['PFS_CENSOR'] == 1, 'PFS_EVENT'] = 0
    #data = data[data['TMB'].notna()]
    cols = []
    cols.extend(met_map.values())
    cols.extend(['AGE', 'SEX', 'PFS_MO', 'PFS_EVENT','ARM'])
    survival_df_add = data[cols].copy()
    survival_df_add['Dataset'] = 'IMmotion151'
    #survival_df = pd.concat([survival_df, survival_df_add], axis=0)
    survival_df_add_T = survival_df_add.loc[survival_df_add['ARM']=='atezo_bev']
    survival_df_add_C = survival_df_add.loc[survival_df_add['ARM']=='sunitinib']
    #survival_df_add_T.to_csv(f'{clinical_path}/treatment_arm/IMmotion151_isotope.csv')
    #survival_df_add_C.to_csv(f'{clinical_path}/control_arm/IMmotion151_isotope.csv')

    # ---------------------------------------- Comparz ----------------------------------------
    sub_dir = 'Comparz'
    results_dir = f'{file_path}/results_RNA_imputation/{sub_dir}'
    embedding_dir = f'{results_dir}/embeddings'
    clinical_path = f'{file_path}/data/clinical_data'
    len_samples = 412

    # Load the predicted MET_RNA matrix
    rank_hat_draws_mean = np.load(f'{embedding_dir}/rank_hat_draws_mean_met.npy')[:len_samples, :]
    # rerank metabolite ranks in the flipped way (now rank 0 means the smallest value, # 1 means the largest value)
    rank_hat_draws_mean = re_rank_2D(rank_hat_draws_mean)
    rank_hat_draws_mean = rank_hat_draws_mean / rank_hat_draws_mean.shape[0]  # transformed all ranks to [0,1)
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                      index_col=0, header=0).iloc[:len_samples, :rank_hat_draws_mean.shape[1]]
    imputed_mean_met = pd.DataFrame(rank_hat_draws_mean, index=normalized_data_met.index,
                                    columns=normalized_data_met.columns)
    imputed_mean_met.to_csv(f'{results_dir}/{sub_dir}_isotope_imputed_mean_met_flipped.csv')
    imputed_mean_met = imputed_mean_met.rename(columns=met_map)


    # Load the clinical data
    clinical = pd.read_excel(f'{clinical_path}/{sub_dir}.xlsx', sheet_name="Comparz.FOLH1",index_col="RNASampleID")
    clinical = clinical.iloc[25:, :]

    # Merge the data
    data = pd.concat([imputed_mean_met, clinical], axis=1, join='inner')
    data = data.iloc[:len_samples, :]
    data = data[data['SRVCFLCD'].notna()]
    data = data[data['SRVMO'].notna()]

    data = data.rename({'SRVCFLCD': 'OS_EVENT'}, axis=1)
    data = data.rename({'SRVMO': 'OS_MO'}, axis=1)
    data = data.rename({'PFSCFLCD': 'PFS_EVENT'}, axis=1)
    data = data.rename({'PFSMO': 'PFS_MO'}, axis=1)
    #data = data[data['HscrGrp'].notna()]
    #data = data[data['CD8grp'].notna()]

    cols = []
    cols.extend(met_map.values())
    cols.extend(['AGE', 'SEX', 'PFS_MO', 'PFS_EVENT','OS_EVENT','OS_MO','TRTGRP'])
    survival_df_add = data[cols].copy()
    survival_df_add['Dataset'] = 'Comparz'
    #survival_df = pd.concat([survival_df, survival_df_add], axis=0)
    survival_df_add_T = survival_df_add.loc[survival_df_add['TRTGRP']=='pazopanib']
    survival_df_add_C = survival_df_add.loc[survival_df_add['TRTGRP']=='sunitinib']
    #survival_df_add_T.to_csv(f'{clinical_path}/treatment_arm/Comparz.csv')
    #survival_df_add_C.to_csv(f'{clinical_path}/control_arm/Comparz.csv')

    # ---------------------------------------- CheckMate214 ----------------------------------------
    sub_dir = 'CheckMate214'
    results_dir = f'{file_path}/results_RNA_imputation/{sub_dir}'
    embedding_dir = f'{results_dir}/embeddings'
    clinical_path = f'{file_path}/data/clinical_data'
    len_samples = 167

    # Load the predicted MET_RNA matrix
    rank_hat_draws_mean = np.load(f'{embedding_dir}/rank_hat_draws_mean_met.npy')[:len_samples, :]
    # rerank metabolite ranks in the flipped way (now rank 0 means the smallest value, # 1 means the largest value)
    rank_hat_draws_mean = re_rank_2D(rank_hat_draws_mean)
    rank_hat_draws_mean = rank_hat_draws_mean / rank_hat_draws_mean.shape[0]  # transformed all ranks to [0,1)
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                      index_col=0, header=0).iloc[:len_samples, :rank_hat_draws_mean.shape[1]]
    imputed_mean_met = pd.DataFrame(rank_hat_draws_mean, index=normalized_data_met.index,
                                    columns=normalized_data_met.columns)
    imputed_mean_met.to_csv(f'{results_dir}/{sub_dir}_isotope_imputed_mean_met_flipped.csv')
    imputed_mean_met = imputed_mean_met.rename(columns=met_map)


    # Load the clinical data
    clinical = pd.read_csv(f'{clinical_path}/{sub_dir}.csv', header=0, index_col=0)
    # rename index for checkmate 214
    new_index = []
    for i in range(clinical.shape[0]):
        new_index.append(clinical.index[i].replace('-','.'))
    clinical['new_index'] = new_index
    clinical = clinical.set_index('new_index')

    # Merge the data
    data = pd.concat([imputed_mean_met, clinical], axis=1, join='inner')
    data = data.iloc[:len_samples, :]

    data = data.rename({'Age':'AGE'},axis=1)
    data = data.rename({'Sex':'SEX'},axis=1)
    data = data.rename({'Progression Free Survival per Investigator Primary Definition (months) (PFSINV)':'PFS_MO'},axis=1)
    #data = data.rename({'Progression Free Survival per IRRC Primary Definition (months) (PFSIRC)':'PFS_MO'},axis=1)
    data.loc[data['PFSINV Censor'] == 0, 'PFS_EVENT'] = 1
    data.loc[data['PFSINV Censor'] == 1, 'PFS_EVENT'] = 0
    #data.loc[data['PFSIRC Censor'] == 0, 'PFS_EVENT'] = 1
    #data.loc[data['PFSIRC Censor'] == 1, 'PFS_EVENT'] = 0

    data = data.rename({'Overall Survival (months) (OS)':'OS_MO'},axis=1)
    data.loc[data['OS Censor'] == 0, 'OS_EVENT'] = 1
    data.loc[data['OS Censor'] == 1, 'OS_EVENT'] = 0
    #data = data.rename({'PD-L1 Expression from Validated Assay':'PDL1_Expression'},axis=1)
    #data = data[data['PDL1_Expression'].notna()]
    #data['BMI'] = data['Weight at Baseline (kg)']/((data['Height at Baseline (cm)'])**2)*10000
    #data = data[data['BMI'].notna()]
    data = data[data['AGE'].notna()]

    cols = []
    cols.extend(met_map.values())
    cols.extend(['AGE', 'SEX', 'PFS_MO', 'PFS_EVENT','OS_EVENT','OS_MO','Arm'])
    survival_df_add = data[cols].copy()
    survival_df_add['Dataset'] = 'CheckMate214'
    #survival_df = pd.concat([survival_df, survival_df_add], axis=0)
    survival_df_add_T = survival_df_add.loc[survival_df_add['Arm']=='Nivo+Ipi']
    survival_df_add_C = survival_df_add.loc[survival_df_add['Arm']=='Sunitinib']
    #survival_df_add_T.to_csv(f'{clinical_path}/treatment_arm/CheckMate214.csv')
    #survival_df_add_C.to_csv(f'{clinical_path}/control_arm/CheckMate214.csv')



    #---------------------- Multivariate Cox’s proportional hazard model ----------------------
    # PFS
    met_cox_df = pd.DataFrame(
        {'coef': [], 'exp(coef)': [], 'se(coef)': [], 'coef lower 95%': [], 'coef upper 95%': [],
         'exp(coef) lower 95%': [], 'exp(coef) upper 95%': [], 'cmp to': [], 'z': [], 'p': [], '-log2(p)': []})
    for i in list(met_map.values()):
        cph = CoxPHFitter()
        cph.fit(survival_df_add_C[survival_df_add_C['AGE'].notna()], duration_col='PFS_MO', event_col='PFS_EVENT',
                formula=f'{i}+AGE+SEX')
        met_cox_df.loc[i]=cph.summary.loc[i]
    met_cox_df.index = list(met_map.keys())
    met_cox_df['p_adj'] = multipletests(pvals=met_cox_df['p'], method="fdr_bh", alpha=0.1)[1]
    print('# Significant metabolites:', met_cox_df[met_cox_df['p_adj']< 0.1].shape[0])
    met_cox_df.to_csv(f'{results_dir}/survival/agesex_PFS_C_9met.csv')

    # OS
    met_cox_df = pd.DataFrame(
        {'coef': [], 'exp(coef)': [], 'se(coef)': [], 'coef lower 95%': [], 'coef upper 95%': [],
         'exp(coef) lower 95%': [], 'exp(coef) upper 95%': [], 'cmp to': [], 'z': [], 'p': [], '-log2(p)': []})
    for i in met_map.values():
        cph = CoxPHFitter()
        cph.fit(survival_df_add_C[survival_df_add_C['OS_MO'].notna()], duration_col='OS_MO', event_col='OS_EVENT',
                formula=f'{i}+AGE+SEX')
        met_cox_df.loc[i]=cph.summary.loc[i]
    met_cox_df.index = met_map.keys()
    met_cox_df['p_adj'] = multipletests(pvals=met_cox_df['p'], method="fdr_bh", alpha=0.1)[1]
    print('# Significant metabolites:', met_cox_df[met_cox_df['p_adj']< 0.1].shape[0])
    met_cox_df.to_csv(f'{results_dir}/survival/agesex_OS_C_9met.csv')



    ### Fisher's Method to combine p values
    sub_dir = 'CheckMate214'
    results_dir = f'results_MET_RNA_imputation/{sub_dir}'
    checkmate214 = pd.read_csv(f'{results_dir}/multi_cox_df_PFS_C_35.csv', header=0, index_col=0)
    sub_dir = 'Comparz'
    results_dir = f'results_MET_RNA_imputation/{sub_dir}'
    comparz = pd.read_csv(f'{results_dir}/multi_cox_df_PFS_C_35.csv', header=0, index_col=0)
    sub_dir = 'IMmotion151'
    results_dir = f'results_MET_RNA_imputation/{sub_dir}'
    immotion151 = pd.read_csv(f'{results_dir}/multi_cox_df_PFS_C_35.csv', header=0, index_col=0)
    sub_dir = 'javelin_101'
    results_dir = f'results_MET_RNA_imputation/{sub_dir}'
    javelin= pd.read_csv(f'{results_dir}/multi_cox_df_PFS_C_35.csv', header=0, index_col=0)
    combined_cox_df = pd.DataFrame( {'coef': [], 'p': []})
    for i in javelin.index:
        p_list = [javelin.loc[i,'p'],immotion151.loc[i,'p'],comparz.loc[i,'p'],checkmate214.loc[i,'p']]
        test_statistic, combined_p_value = scipy.stats.combine_pvalues(p_list,method='fisher',weights=None)
        ave_coef = mean([javelin.loc[i,'coef'],immotion151.loc[i,'coef'],comparz.loc[i,'coef'],checkmate214.loc[i,'coef']])
        combined_cox_df.loc[i,'coef']= ave_coef
        combined_cox_df.loc[i, 'p'] = combined_p_value
    combined_cox_df.to_csv(f'results_MET_RNA_imputation/multi_combined_PFS_CPH_C_35met.csv')


    p_list = [0.008857636, 0.243205604, 0.002519368,0.962589987]
    test_statistic, combined_p_value = scipy.stats.combine_pvalues(p_list, method='fisher', weights=None)

    ########### K-M plot
    # for 1-methylimidazoleacetate
    plt.hist(survival_df_add_C['26a'])
    plt.xlabel("Imputed level of 1-methylimidazoleacetate")
    plt.ylabel("Counts of samples")
    plt.title('Histogram of metabolite 1-methylimidazoleacetate')
    plt.show()
    print(survival_df_add_C['26a'].median()) #0.4711141678129299

    test = survival_df_add_C
    T1 = test[test['26a']<0.5211864406779662]['PFS_MO']
    T2 = test[test['26a']>=0.5211864406779662]['PFS_MO']
    E1 = test[test['26a']<0.5211864406779662]['PFS_EVENT']
    E2 = test[test['26a']>= 0.5211864406779662]['PFS_EVENT']
    kmf = KaplanMeierFitter(label="High 1-methylimidazoleacetate")
    kmf.fit(T2, E2)
    kmf.plot(show_censors=True,ci_show=False)
    print(kmf.median_survival_time_)
    kmf = KaplanMeierFitter(label="Low 1-methylimidazoleacetate")
    kmf.fit(T1, E1)
    kmf.plot(show_censors=True,ci_show=False)
    print(kmf.median_survival_time_)

    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    print(results.p_value)
    plt.xlabel("Progression Free Survival(months)")
    plt.ylabel("Survival Probability")
    plt.show()

    plt.savefig('results_MET_RNA_imputation/tcga/plots/KM_1-methylimidazoleacetate_COMPARZ.pdf')
    plt.close()







