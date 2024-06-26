import pandas as pd
import gzip
import os

# unzip the file
def extract_all_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.gz'):
                gz_path = os.path.join(root, file)
                with gzip.open(gz_path, 'rb') as gz_file:
                    extracted_path = os.path.splitext(gz_path)[0]  # Remove the .gz extension
                    with open(extracted_path, 'wb') as extracted_file:
                        extracted_file.write(gz_file.read())


def read_and_concatenate_dataframes(root_dir):
    dfs = []  # List to store individual dataframes

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.gz'):
                gz_path = os.path.join(root, file)
                with gzip.open(gz_path, 'rb') as gz_file:
                    # Read the uncompressed contents using pandas
                    df = pd.read_csv(gz_file, sep='\t', comment='#')
                    dfs.append(df)

    concatenated_df = pd.concat(dfs, ignore_index=True)  # Concatenate the dataframes

    return concatenated_df

if __name__ == "__main__":
    data_path = "/juno/work/reznik/xiea1/MetabolicModel/data/CPTAC_DNA"
    # Provide the root directory where your subfolders are located
    root_directory = f'{data_path}/cptac_gdc_snv'
    # Call the function to extract files from .gz archives in subfolders
    extract_all_files(root_directory)
    # Call the function to read and concatenate the dataframes
    data = read_and_concatenate_dataframes(root_directory)
    data.to_csv(f'{data_path}/cptac_gdc_snv.csv')

    #----------------------------------------------------------------------------------------------------------------
    data = pd.read_csv(f'{data_path}/cptac_gdc_snv.csv', index_col=0, header=0)
    sample = pd.read_csv(f'{data_path}/cptac_gdc_sample.tsv', sep='\t', index_col=0, header=0)
    # mapping case_id to case_submitter_id in the maf file
    case_dict = dict(zip(sample['case_id'], sample['case_submitter_id']))
    data['case_submitter_id'] = data['case_id'].map(case_dict)
    # subset to matching CPTAC KIRC samples
    cptac = pd.read_csv(f'/juno/work/reznik/xiea1/MetabolicModel/data/MET_matched_ccRCC/matched_Harmonized_Met_CPTAC.csv',
                        index_col=0, header=0)
    cptac_val = pd.read_csv(f'/juno/work/reznik/xiea1/MetabolicModel/data/MET_matched_ccRCC/matched_Harmonized_Met_CPTAC_val.csv',
                            index_col=0, header=0)

    # only the participant and sample code matters
    cptac['case_submitter_id_short'] = cptac.index.str[:-2]
    cptac_val['case_submitter_id_short'] = cptac_val.index.str[:-2]
    data_cptac = data[data['case_submitter_id'].isin(cptac['case_submitter_id_short'])]
    data_cptac_val = data[data['case_submitter_id'].isin(cptac_val['case_submitter_id_short'])]
    total_samples_cptac = set(data_cptac['case_submitter_id'])  # 50 total samples
    total_samples_cptac_val = set(data_cptac_val['case_submitter_id'])  # 56 total samples

    # subset to 15 ccRCC driver mutations
    driver_mutations = ['VHL', 'PBRM1', 'SETD2', 'BAP1', 'MTOR', 'KDM5C', 'PIK3CA', 'PIK3R1', 'PTEN', 'TP53', 'TSC1', 'TSC2', 'TCEB1', 'FH', 'SDHB']
    data_cptac = data_cptac[data_cptac['Hugo_Symbol'].isin(driver_mutations)]
    data_cptac_val = data_cptac_val[data_cptac_val['Hugo_Symbol'].isin(driver_mutations)]
    # excluded ‘silent’, ‘intron’, 'IGR', '5'Flank', 'RNA', '3'Flank ', ‘3utr’, and ‘5 utr’, because they won’t bring a change to gene functions
    driver_mutation_types = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
                            'Splice_Site', 'Frame_Shift_Ins', 'In_Frame_Del', 'Splice_Region',
                             'Translation_Start_Site', 'In_Frame_Ins', 'Nonstop_Mutation']
    data_cptac = data_cptac[data_cptac['Variant_Classification'].isin(driver_mutation_types)]
    data_cptac_val = data_cptac_val[data_cptac_val['Variant_Classification'].isin(driver_mutation_types)]


    # pivot tables for both cptac and cptac_val
    df_wide = pd.pivot_table(data_cptac_val[['case_submitter_id', 'Hugo_Symbol']], index='case_submitter_id', columns='Hugo_Symbol', aggfunc=len, fill_value=0)
    # Convert non-zero values to 1
    df_wide[df_wide > 0] = 1  # shape of df_wide: (48, 12) cptac; (50, 11) cptac_val
    # There are 50 samples with matched RNA-seq and DNA seq, but only 290 samples with mutations
    # in the 15 driver genes, so I need to fill in the missing 78 samples with 0s
    missing_samples = list(total_samples_cptac_val - set(df_wide.index))
    df_missing = pd.DataFrame(0, index=missing_samples, columns=df_wide.columns)
    df = pd.concat([df_wide, df_missing], axis=0)
    df.to_csv(f'{data_path}/CPTAC_val_KIRC_sample_mutation_matrix.csv')
