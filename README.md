# UnitedMet: A Bayesian probabilistic method to predict metabolic phenotypes from transcriptomic data
This package implements UnitedMet, a Bayesian probabilistic framework jointly modeling metabolomics/isotopologue data and RNA-seq data. Training target RNA-seq dataset with cancer-specific reference datasets with paired metabolics/isotopologue data and transcriptomic data, UnitedMet is capable of imputing whole pool sizes as well as the outcomes of isotope tracing experiments for the target dataset. 

This repository contains code for UnitedMet imputation, scripts to benchmark UnitedMet against Lasso and MIRTH, as well as a demonstration of imputation benchmarking.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11286535.svg)](https://doi.org/10.5281/zenodo.11286535)


## Directory structure
```
.
├── UnitedMet               # Main Python code to impute metabolomics/isotopologue data
├── data                    # Preprocessed data files for benchmarking
├── benchmarking_scripts    # Scripts to benchmark UnitedMet's performance
├── analysis_scripts        # Scripts to perform downstream biological analysis
├── figure_scripts          # Scripts to reproduce figures in the paper
├── UnitedMet_env.yaml      # Dependencies
├── impute_met_demo.ipynb   # Demo code to run UnitedMet and benchmark performance
└── README.md
```
## Installation and Dependencies

UnitedMet has been implemented in Python3 and can be installed into a conda environment using the following commands
```
git clone https://github.com/reznik-lab/UnitedMet.git
```
Install anaconda (at least Python v3.11): https://www.anaconda.com/download/

Activate the conda and Clone a conda environment according to UnitedMet_env.yml
```
source anaconda3/bin/activate
conda env create -f UnitedMet_env.yml
```
## Instructions for Imputing Metabolomics with UnitedMet

### Imputate metabolomics data with UnitedMet: 

Reference data (4 ccRCC datasets with paired metabolomics and RNA-seq data) are available here: https://doi.org/10.5281/zenodo.11286535

To run UnitedMet imputation:
1) **Input files**
 - `file_path`: set a main directory for input data, e.g. ~/data. Under this main directory:
    - `rna_matched_data_dir`: input directory for paired RNA-seq reference datasets
      - All RNA-seq files should be named 'matched_tpm_<dataset name>.csv'. Column names (the first row) should be gene names. Row names (the first column) should be sample names/IDs. Gene names should be harmonized across all datasets under consideration.
    - `met_matched_data_dir`: input directory for paired metabolomics reference datasets.
      - All metabolomics datasets should be saved as '.csv' files. Column names (the first row) should be metabolite names. Row names (the first column) should be sample names/IDs. Metabolite names should be harmonized across all datasets under consideration.
    - `rna_imputation_data_dir`: input directory for single-modality RNA-seq dataset
      - The RNA-seq file should be named 'matched_tpm_<dataset name>.csv'. Column names (the first row) should be gene names. Row names (the first column) should be sample names/IDs. Gene names should be harmonized across all datasets under consideration.
   - `MasterMapping_ccRCC.csv`: a sreadsheet containing matched RNA-seq and metabolomics sample IDs. If using our 4 ccRCC datasets as reference, place this file under the main directory as well.
2) **Run cross-validation**
 - Grid search to determine hyperparameter `ndims` (the number of latent dimensions for embedding matrices W and H)
   - If you are able to run job arrays on an HPC cluster, we recommend running cross-validation to determine the optimal hyperparameter for your datasets. Running cross-validation on a local computer could be time-consuming.
   - You can find bash scripts to run cross-validation on a HPC using SLURM here: `/UnitedMet/impute_met/jobs/job_cv_met`.
3) **Run imputation**:
- Running command lines as below
```
cd UnitedMet
conda activate UnitedMet
python3 -m UnitedMet.impute_met.local.impute.py --verbose_embeddings --n_steps 4000 --learning_rate 0.001 -fp ${file_path} -rna ${rna_matched_data_dir} -met ${met_matched_data_dir} -id ${rna_imputation_data_dir} -ck -rd ${results_dir} -n ${ndims}
```
  - Specify `file_path`, `rna_matched_data_dir`, `met_matched_data_dir`, `rna_imputation_data_dir`, `rresults_dir` according to your file path set up.
  - Remove `-ck` if not using our 4 ccRCC datasets as reference. 
  - Specify `ndims` with the optimal number you get from grid search or using default number 30.
  - Please note that imputation of metabolomic levels for TCGA KIRC cohort (606 samples, trained with 4 ccRCC datasets containing 341 samples) takes ~1 hour on a HPC cluster.
- You can also run the bash script `/UnitedMet/impute_met/jobs/job_met_imputation` for imputation, if you prefer to run it as a job submitted to the HPC. 
 4) **Output**:
 - UnitedMet outputs the following cross-validation results under `/<results_dir>`:
   - `cv_best_dims_scores.csv`: the best `ndims` found by cross-validation. **This is the main output of cross-validation.**
   - `cv_score.pdf`: under `/<results_dir>/plots`, a plot displaying how fold-average mean absolute error changes with `ndims`. **This is the main visualization output of cross-validation.**
   - `cv_folds_scores.csv`: logs the mean absolute error in each fold for each number of embedding dimensions sampled
   - `scores.txt`: : logs the mean absolute error in each fold for each number of embedding dimensions sampled, txt file
   
- UnitedMet outputs posterior predicted matrices under `/<results_dir>/embeddings`:
  - `normalized_met_rna_data_pyro.csv`: the aggregate data matrix consisting of the merged datasets with normalized RNA-seq and metabolomics data
  - `W_H_loc_scale.npy`: arrays of posterior distribution parameters (loc, scale of the normal distribution) of W and H embedding matrices
  - `W_draws.npy`: posterior draws of W
  - `H_draws.npy`: posterior draws of W
  - `rank_hat_draws_met.npy`: samples of posterior predicted metabolites
    
- UnitedMet outputs final imputed results under `/<results_dir>`:
  - `target_imputed_met_mean.csv`: predicted metabolomics data (posterior mean) for the target dataset. **This is the main imputed output.**
  - `target_imputed_met_std.csv`: standard deviations of posterior draws for the target dataset (as a approach of uncertainty quantification)

## Scaled-down demonstration of UnitedMet imputation performance
The easiest way to view the Jupyter Notebook demonstration in on this GitHub repository. Please navigate to the `impute_met_demo.ipynb` file.

To run the Jupyter Notebook demonstration locally, install Jupyter Notebook and the required dependencies used in the demonstration.
```
conda install -c conda-forge notebook
```


By default, the demonstration reads 4 ccRCC datasets from the `data` folder in the same directory as `impute_met_demo.ipynb`. Our demonstration was now running only with 100 steps in SVI in the interest of runtime (not the same performance as we showed in the paper).

Then, at the command line, navigate to the directory containing `impute_met_demo.ipynb` and run the following:
```
jupyter notebook
```
This should open Jupyter Notebook in a web browser. Click on the `impute_met_demo.ipynb` file to open it, and run each cell of the file in order from the top.

