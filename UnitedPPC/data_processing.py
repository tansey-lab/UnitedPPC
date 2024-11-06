import numpy as np
import pandas as pd
import os

from UnitedPPC.utils import count_obs, order_and_rank

def generate_concatenated_ppc_data(
    rna_file: str,
    zscore_file: str,
    meta_file: str,
):
    """Generate concatenated data for UnitedPPC.
    
    Parameters
    ----------
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
    for file in [rna_file, zscore_file, meta_file]:
        if not os.path.exists(file):
            raise OSError(f"File {file} doesn't exist.")
        
    df_meta_sample = pd.read_csv(meta_file, header=0)
    
    df_rna = pd.read_csv(rna_file, header=0, index_col=0)
    n_rna = df_rna.shape[1]

    df_zscore = pd.read_csv(zscore_file, header=0)
    df_zscore_pivot = df_zscore.pivot(
        index="sample_id", 
        columns="drug_id", 
        values="z-score",
    )
    n_zscore = df_zscore_pivot.shape[1]

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
    n_batch = len(dict_study)

    # double check
    # len(df_zscore_pivot.index.intersection(df_rna.index)) + \
    # len(df_zscore_pivot.index.symmetric_difference(df_rna.index)) == \
    # df_merge.shape[0]
    # df_meta_sample["id_sample"].isin(df_merge.index).sum() == df_merge.shape[0]

    return df_merge, batch_index_vector, n_batch, n_rna, n_zscore

def file_system_init(results_dir):
    """Initialize the file system.

    Parameters
    ----------
    results_dir : str
        The directory to save the results.

    Returns
    -------
    plots_dir : str
        The plots directory.
    embedding_dir : str
        The embeddings directory.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # define sub-directories inside the results directory
    plots_dir = f"{results_dir}/plots"
    embedding_dir = f"{results_dir}/embeddings"

    for dir in [embedding_dir, plots_dir]:
        try:
            os.makedirs(dir)
        except FileExistsError:
            print(f"Directory {dir} already exists.")
    return (plots_dir,  embedding_dir)
