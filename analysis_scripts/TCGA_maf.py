import pandas as pd
import requests
import gzip

if __name__ == "__main__":
    data_path = "/juno/work/reznik/xiea1/MetabolicModel/data/TCGA_downstream"
    # download TCGA mc3 maf file from the website: https://gdc.cancer.gov/about-data/publications/mc3-2017
    url = 'https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc'  # Replace with the URL of the file you want to download
    filename = 'mc3.v0.2.8.PUBLIC.maf.gz'  # Specify the filename you want to save the file as

    response = requests.get(url)
    response.raise_for_status()  # Check for any potential errors

    with open(f'{data_path}/{filename}', 'wb') as file:
        file.write(response.content)
    print(f"File '{filename}' downloaded successfully.")
    # unzip the file
    with gzip.open(f'{data_path}/{filename}', 'rb') as f_in:
        with open(f'{data_path}/mc3.v0.2.8.PUBLIC.maf', 'wb') as f_out:
            f_out.write(f_in.read())

    #----------------------------------------------------------------------------------------------------------------
    data = pd.read_csv(f'{data_path}/mc3.v0.2.8.PUBLIC.maf', sep='\t')
    # subset to TCGA KIRC samples
    tcga = pd.read_csv(f'/juno/work/reznik/xiea1/MetabolicModel/data/RNA_matched_imputation/matched_tpm_TCGA.csv', index_col=0, header=0)
    # only the participant and sample code matters
    data['Tumor_Sample_Barcode_short'] = data['Tumor_Sample_Barcode'].str[8:15]
    tcga['Sample_Barcode_short'] = tcga.index.str[8:15]
    data = data[data['Tumor_Sample_Barcode_short'].isin(tcga['Sample_Barcode_short'])]
    total_samples = set(data['Tumor_Sample_Barcode'])  # 368 total samples
    # subset to 15 ccRCC driver mutations
    driver_mutations = ['VHL', 'PBRM1', 'SETD2', 'BAP1', 'MTOR', 'KDM5C', 'PIK3CA', 'PIK3R1', 'PTEN', 'TP53', 'TSC1', 'TSC2', 'TCEB1', 'FH', 'SDHB']
    data = data[data['Hugo_Symbol'].isin(driver_mutations)]
    # excluded ‘silent’, ‘intron’, ‘3utr’, and ‘5 utr’, because they won’t bring a change to gene functions
    driver_mutation_types = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
                            'Splice_Site', 'Frame_Shift_Ins', 'In_Frame_Del',
                             'Translation_Start_Site', 'In_Frame_Ins', 'Nonstop_Mutation']
    data = data[data['Variant_Classification'].isin(driver_mutation_types)]


    # pivot table
    df_wide = pd.pivot_table(data[['Tumor_Sample_Barcode', 'Hugo_Symbol']], index='Tumor_Sample_Barcode', columns='Hugo_Symbol', aggfunc=len, fill_value=0)
    # Convert non-zero values to 1
    df_wide[df_wide > 0] = 1  # shape of df_wide: (290, 14)
    # There are 368 samples with matched RNA-seq and DNA seq, but only 290 samples with mutations
    # in the 15 driver genes, so I need to fill in the missing 78 samples with 0s
    missing_samples = list(total_samples - set(df_wide.index))
    df_missing = pd.DataFrame(0, index=missing_samples, columns=df_wide.columns)
    df = pd.concat([df_wide, df_missing], axis=0)
    df.index = df.index.str[8:15]
    df.to_csv(f'{data_path}/TCGA_KIRC_sample_mutation_matrix.csv')

    #------------------------------------GET KIPAN MAF----------------------------------
    data_path = "/juno/work/reznik/xiea1/MetabolicModel/data/TCGA_downstream"
    data = pd.read_csv(f'{data_path}/mc3.v0.2.8.PUBLIC.maf', sep='\t')
    # subset to TCGA KIPAN samples
    kipan = pd.read_csv(f'/juno/work/reznik/xiea1/MetabolicModel/data/RNA_matched_imputation/matched_tpm_KIPAN.csv', index_col=0, header=0)
    # only the participant and sample code matters
    data['Tumor_Sample_Barcode_short'] = data['Tumor_Sample_Barcode'].str[8:15]
    kipan['Sample_Barcode_short'] = kipan.index.str[8:15]
    data = data[data['Tumor_Sample_Barcode_short'].isin(kipan['Sample_Barcode_short'])]
    total_samples = set(data['Tumor_Sample_Barcode'])  # 715 total samples
    # subset to 15 ccRCC driver mutations
    driver_mutations = ['VHL', 'PBRM1', 'SETD2', 'BAP1', 'MTOR', 'KDM5C', 'PIK3CA', 'PIK3R1', 'PTEN', 'TP53', 'TSC1', 'TSC2', 'TCEB1', 'FH', 'SDHB']
    data = data[data['Hugo_Symbol'].isin(driver_mutations)]
    # excluded ‘silent’, ‘intron’, ‘3utr’, and ‘5 utr’, because they won’t bring a change to gene functions
    driver_mutation_types = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
                            'Splice_Site', 'Frame_Shift_Ins', 'In_Frame_Del',
                             'Translation_Start_Site', 'In_Frame_Ins', 'Nonstop_Mutation']
    data = data[data['Variant_Classification'].isin(driver_mutation_types)]


    # pivot table
    df_wide = pd.pivot_table(data[['Tumor_Sample_Barcode', 'Hugo_Symbol']], index='Tumor_Sample_Barcode', columns='Hugo_Symbol', aggfunc=len, fill_value=0)
    # Convert non-zero values to 1
    df_wide[df_wide > 0] = 1  # shape of df_wide: (391, 15)
    # There are 715 samples with matched RNA-seq and DNA seq, but only 391 samples with mutations
    # in the 15 driver genes, so I need to fill in the missing 78 samples with 0s
    missing_samples = list(total_samples - set(df_wide.index))
    df_missing = pd.DataFrame(0, index=missing_samples, columns=df_wide.columns)
    df = pd.concat([df_wide, df_missing], axis=0)
    df.index = df.index.str[8:15]
    df.to_csv(f'{data_path}/TCGA_KIPAN_sample_mutation_matrix.csv')
