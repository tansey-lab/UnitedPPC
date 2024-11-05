import numpy as np
import pandas as pd

# path_main = "/data1/tanseyw/projects/whitej/UnitedPPC"
# path_data = f"{path_main}/data"

# file_sample_meta = "sample_metadata.csv"
# file_rna = "df_count_inner.csv"
# file_zscore = "mean_aucs.csv"

def generate_concatenated_ppc_data(
    data_path: str,
    rna_file: str,
    zscore_file: str,
    meta_file: str,
):
    """Generate concatenated data for UnitedPPC.
    
    Parameters
    ----------
    data_path : str
        Path to the data directory.
    rna_file : str
        RNA file name.
    zscore_file : str
        Z-score file name.
    meta_file : str
        Meta file name.

    Returns
    -------
    df_merge : pd.DataFrame
        Merged data.
    batch_index_vector : np.ndarray
        Batch index vector.
    
    """

    df_meta_sample = pd.read_csv(f'{data_path}/{meta_file}', header=0)
    df_rna = pd.read_csv(f'{data_path}/{rna_file}', header=0, index_col=0)

    df_zscore = pd.read_csv(f'{data_path}/{zscore_file}', header=0)
    df_zscore_pivot = df_zscore.pivot(
        index="sample_id", 
        columns="drug_id", 
        values="z-score",
    )

    df_merge = pd.merge(
        df_zscore_pivot, 
        df_rna, 
        left_index=True, 
        right_index=True, 
        how="outer",
    )

    # sort samply meta by study_id
    df_meta_sample = df_meta_sample.sort_values(by="study_id").reset_index(drop=True)
    # extract id_sample sorted by study_id and sort df_merge by sorted id_sample
    list_sample_sorted = df_meta_sample.loc[
        df_meta_sample["id_sample"].isin(df_merge.index), "id_sample"].tolist()
    df_merge = df_merge.loc[list_sample_sorted]
    # generate batch_index_vector by extracting study_id > convert to int > convert to numpy array
    list_study_name = df_meta_sample.loc[
        df_meta_sample["id_sample"].isin(df_merge.index), "study_id"].tolist()
    dict_study = {i: idx for idx, i in enumerate(df_meta_sample["study_id"].unique())}
    list_study_idx = [dict_study[i] for i in list_study_name]
    batch_index_vector = np.array(list_study_idx, dtype=int)

    # double check
    # len(df_zscore_pivot.index.intersection(df_rna.index)) + \
    # len(df_zscore_pivot.index.symmetric_difference(df_rna.index)) == \
    # df_merge.shape[0]
    # df_meta_sample["id_sample"].isin(df_merge.index).sum() == df_merge.shape[0]

    return df_merge, batch_index_vector