#!/usr/bin/env python3
"""
preprocessing.py — Patient-level fold generation and leakage validation.

CLI usage:
    python preprocessing.py --csv /path/to/wilmstumor.csv --output /path/to/splits.csv
    python preprocessing.py --check /path/to/splits.csv
"""
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def generate_patient_folds_binary(wilmstumor_csv, output_csv, n_splits=5, random_state=42):
    """
    Generate patient-level stratified k-fold splits for binary classification
    (Not Anaplasia vs. Focal/Diffuse). Prevents patient leakage across folds.

    Parameters
    ----------
    wilmstumor_csv : str
        Path to original CSV with columns: Patient_id, Diagnose, slide_id.
    output_csv : str
        Path to save the CSV with added 'fold' column (1-based).
    n_splits : int
        Number of CV folds.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Updated dataframe with 'fold' column.
    """
    df = pd.read_csv(wilmstumor_csv)

    patient_df = df.groupby("Patient_id").first().reset_index()[["Patient_id", "Diagnose"]]

    binary_map = {"Not Anaplasia": 0, "Focal": 1, "Diffuse": 1}
    patient_df["label"] = patient_df["Diagnose"].map(binary_map)

    if patient_df["label"].isnull().any():
        unknowns = patient_df[patient_df["label"].isnull()]["Diagnose"].unique()
        raise ValueError(f"Unknown label(s) in data: {unknowns}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    patient_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(patient_df["Patient_id"], patient_df["label"])):
        patient_df.loc[val_idx, "fold"] = fold + 1

    df = df.merge(patient_df[["Patient_id", "fold"]], on="Patient_id", how="left")
    df.to_csv(output_csv, index=False)
    print(f"Saved {n_splits}-fold splits to: {output_csv}")
    return df


def check_patient_leakage(csv_path):
    """
    Verify that each patient appears in exactly one fold.

    Parameters
    ----------
    csv_path : str
        Path to CSV with columns: Patient_id, fold.

    Returns
    -------
    bool
        True if no leakage detected.
    """
    df = pd.read_csv(csv_path)
    fold_map = df.groupby("Patient_id")["fold"].nunique()
    leaking = fold_map[fold_map > 1]

    if leaking.empty:
        print("No patient leakage detected across folds.")
        return True
    else:
        print("Patient leakage found — patients appearing in multiple folds:")
        print(leaking)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or validate patient-level CV splits")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="Input CSV to generate folds from")
    group.add_argument("--check", help="CSV with fold column to validate for leakage")
    parser.add_argument("--output", help="Output path for generated splits CSV (required with --csv)")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.csv:
        if not args.output:
            parser.error("--output is required when using --csv")
        generate_patient_folds_binary(args.csv, args.output, n_splits=args.n_splits, random_state=args.seed)
    else:
        check_patient_leakage(args.check)
