import csv
import argparse
from UnitedPPC.utils import parsearg_utils
from UnitedPPC.cross_validation import summarize_cv_results

if __name__ == "__main__":
    # ------------------------------------- argparse -------------------------------------
    args = parsearg_utils().parse_args()

    # output options
    dir_results = args.results_dir
    # cross-validation options
    cv_dims = args.crossval_dims
    cv_step = args.crossval_steps
    cv_folds = args.crossval_folds

    # ------------------------------------- main -------------------------------------
    plots_dir = f"{dir_results}/plots"

    n_dims_knee = summarize_cv_results(
        cv_dims, 
        cv_step, 
        cv_folds, 
        dir_results, 
        plots_dir
    )

    with open(f"{dir_results}/cv_best_dims.csv", "w+") as file:
        writer = csv.writer(file)
        writer.writerow(["best_n_dims_elbow"])
        writer.writerow([n_dims_knee])
