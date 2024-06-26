import pandas as pd
import requests
import gzip

if __name__ == "__main__":
    data_path = "/juno/work/reznik/xiea1/MetabolicModel/data/IMmotion151"
    data = pd.read_csv(f'{data_path}/IMmotion151_clinical_genomic.csv', header=0, index_col='RNASEQ_SAMPLE_ID')
    # replace '-' with '.' in index to match the imputed metabolomics data
    data.index = data.index.str.replace('-', '.')
    data = data[data.index.notna()]  # drop samples with no RNA-seq data
    data.columns = data.columns.str.replace('FMI_', '')
    # only last 702 patients have noth DNA-seq and RNA_seq data
    data = data[data['SAMPLE_ID'].notna()]  # drop samples with no DNA-seq data
    # subset to 15 ccRCC driver mutations
    # 'TCEB1' is not in the IMmotion151 mutation data, so I will drop it
    driver_mutations = ['VHL', 'PBRM1', 'SETD2', 'BAP1', 'MTOR', 'KDM5C', 'PIK3CA', 'PIK3R1', 'PTEN', 'TP53', 'TSC1', 'TSC2', 'FH', 'SDHB']
    data = data[driver_mutations]  # shape of data: (702, 14)
    # if value is not na then 1, else 0
    data = data.notna().astype(int)

    data.to_csv(f'{data_path}/IMmotion151_sample_mutation_matrix.csv')
