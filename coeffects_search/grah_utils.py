#係数調整実験の結果をcsvでまとめる
import os
import pandas as pd
from collections import defaultdict


#係数設定ごとの結果をまとめて，データごとにCSVに
def coeffects_result2csv(base_path):
    report_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # For final aggregation
    iemodf_list = []
    melddf_list = []

    for dirpath, dirnames, filenames in os.walk(base_path):
        if "classification_report.csv" in filenames and "arguemts.csv" in filenames:
            parts = dirpath.split(os.sep)
            try:
                coeff_name = next(p for p in parts if p.startswith("coeff_"))
                comb_name = next(p for p in parts if p.startswith("comb_"))
                dataset_name = parts[-1]  # IEMO or MELD
            except StopIteration:
                continue

            classification_path = os.path.join(dirpath, "classification_report.csv")
            arguemts_path = os.path.join(dirpath, "arguemts.csv")

            try:
                classification_df = pd.read_csv(classification_path)
                arguemts_df = pd.read_csv(arguemts_path)
            except Exception as e:
                print(f"Error reading CSVs in {dirpath}: {e}")
                continue
            # Attempt to find the accuracy row based on content
            acc_row = classification_df[classification_df.iloc[:, 0].astype(str).str.lower() == "accuracy"]
            if not acc_row.empty:
                acc_value = acc_row.iloc[0, 1]  # usually in the 'precision' column position
                accuracy_row = pd.DataFrame({
                    "accuracy": [acc_value],
                    "coeff": [coeff_name],
                    "comb": [comb_name],
                    "dataset": [dataset_name]
                })
            else:
                print(f"No accuracy row found in {classification_path}")
                continue

            arguemts_df["coeff"] = coeff_name
            arguemts_df["comb"] = comb_name
            arguemts_df["dataset"] = dataset_name


            # Store for grouped collection
            if dataset_name == "IEMO":
                iemodf_list.append(accuracy_row)
            elif dataset_name == "MELD":
                melddf_list.append(accuracy_row)

            report_data[coeff_name][comb_name][dataset_name] = {
                "classification_report": accuracy_row,
                "arguemts": arguemts_df
            }

    # Aggregate all IEMO and MELD DataFrames
    iemodf = pd.concat(iemodf_list, ignore_index=True) if iemodf_list else pd.DataFrame()
    melddf = pd.concat(melddf_list, ignore_index=True) if melddf_list else pd.DataFrame()

    # Example usage: print structure
    # print("IEMO Classification Reports Shape:", iemodf.shape)
    # print("MELD Classification Reports Shape:", melddf.shape)

    # Optional save
    iemodf.to_csv("iemodf_all.csv", index=False)
    melddf.to_csv("melddf_all.csv", index=False)  # optional output

#損失の調査
def history_csv(base_path):
    history_last_rows = []

    for dirpath, dirnames, filenames in os.walk(base_path):
        parts = dirpath.split(os.sep)

        # Ensure parts contain enough depth
        if len(parts) < 4:
            continue

        try:
            coeff_name = next(p for p in parts if p.startswith("coeff_"))
            comb_name = next(p for p in parts if p.startswith("comb_"))
            dataset_name = parts[-2] if parts[-1] == "train_results" else (parts[-1] if parts[-1] in ["IEMO", "MELD"] else None)
        except StopIteration:
            continue

        # Train history.csv
        if "history.csv" in filenames and dataset_name:
            history_path = os.path.join(dirpath, "history.csv")
            try:
                history_df = pd.read_csv(history_path)
                last_row = history_df.tail(1).copy()
                last_row["coeff"] = coeff_name
                last_row["comb"] = comb_name
                last_row["dataset"] = dataset_name
                history_last_rows.append(last_row)
            except Exception as e:
                print(f"Error reading history.csv in {dirpath}: {e}")
                continue

    history_summary = pd.concat(history_last_rows, ignore_index=True) if history_last_rows else pd.DataFrame()
    history_summary.to_csv("final_epoch_losses.csv", index=False)

#係数設定ごとの結果をまとめて，データごとにCSVに
def coeffects_result2csv_0704(base_path):
    report_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # For final aggregation
    iemodf_list = []
    melddf_list = []

    for dirpath, dirnames, filenames in os.walk(base_path):
        if "classification_report.csv" in filenames and "arguemts.csv" in filenames:
            parts = dirpath.split(os.sep)
            try:
                gannma_3_name = next(p for p in parts if p.startswith("gannma_3_"))
                kl_epoch_name = next(p for p in parts if p.startswith("kl_epoch_"))
                dataset_name = parts[-1]  # IEMO or MELD
            except StopIteration:
                continue

            classification_path = os.path.join(dirpath, "classification_report.csv")
            arguemts_path = os.path.join(dirpath, "arguemts.csv")

            try:
                classification_df = pd.read_csv(classification_path)
                arguemts_df = pd.read_csv(arguemts_path)
            except Exception as e:
                print(f"Error reading CSVs in {dirpath}: {e}")
                continue
            # Attempt to find the accuracy row based on content
            acc_row = classification_df[classification_df.iloc[:, 0].astype(str).str.lower() == "accuracy"]
            if not acc_row.empty:
                acc_value = acc_row.iloc[0, 1]  # usually in the 'precision' column position
                accuracy_row = pd.DataFrame({
                    "accuracy": [acc_value],
                    "gannma_3": [gannma_3_name],
                    "kl_epoch": [kl_epoch_name],
                    "dataset": [dataset_name]
                })
            else:
                print(f"No accuracy row found in {classification_path}")
                continue

            arguemts_df["gannma_3"] = gannma_3_name
            arguemts_df["kl_epoch"] = kl_epoch_name
            arguemts_df["dataset"] = dataset_name


            # Store for grouped collection
            if dataset_name == "IEMO":
                iemodf_list.append(accuracy_row)
            elif dataset_name == "MELD":
                melddf_list.append(accuracy_row)

            report_data[gannma_3_name][kl_epoch_name][dataset_name] = {
                "classification_report": accuracy_row,
                "arguemts": arguemts_df
            }

    # Aggregate all IEMO and MELD DataFrames
    iemodf = pd.concat(iemodf_list, ignore_index=True) if iemodf_list else pd.DataFrame()
    melddf = pd.concat(melddf_list, ignore_index=True) if melddf_list else pd.DataFrame()

    # Example usage: print structure
    # print("IEMO Classification Reports Shape:", iemodf.shape)
    # print("MELD Classification Reports Shape:", melddf.shape)

    # Optional save
    iemodf.to_csv("iemodf_all.csv", index=False)
    melddf.to_csv("melddf_all.csv", index=False)  # optional output

#係数設定ごとの結果をまとめて，データごとにCSVに
def coeffects_result2csv_0709(base_path):
    report_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # For final aggregation
    iemodf_list = []
    melddf_list = []

    for dirpath, dirnames, filenames in os.walk(base_path):
        if "classification_report.csv" in filenames and "arguemts.csv" in filenames:
            parts = dirpath.split(os.sep)
            try:
                gannma_3_name = next(p for p in parts if p.startswith("conff_"))
                kl_epoch_name = next(p for p in parts if p.startswith("comb_"))
                dataset_name = parts[-1]  # IEMO or MELD
            except StopIteration:
                continue

            classification_path = os.path.join(dirpath, "classification_report.csv")
            arguemts_path = os.path.join(dirpath, "arguemts.csv")

            try:
                classification_df = pd.read_csv(classification_path)
                arguemts_df = pd.read_csv(arguemts_path)
            except Exception as e:
                print(f"Error reading CSVs in {dirpath}: {e}")
                continue
            # Attempt to find the accuracy row based on content
            acc_row = classification_df[classification_df.iloc[:, 0].astype(str).str.lower() == "accuracy"]
            if not acc_row.empty:
                acc_value = acc_row.iloc[0, 1]  # usually in the 'precision' column position
                accuracy_row = pd.DataFrame({
                    "accuracy": [acc_value],
                    "conff": [gannma_3_name],
                    "comb": [kl_epoch_name],
                    "dataset": [dataset_name]
                })
            else:
                print(f"No accuracy row found in {classification_path}")
                continue

            arguemts_df["gannma_3"] = gannma_3_name
            arguemts_df["kl_epoch"] = kl_epoch_name
            arguemts_df["dataset"] = dataset_name


            # Store for grouped collection
            if dataset_name == "IEMO":
                iemodf_list.append(accuracy_row)
            elif dataset_name == "MELD":
                melddf_list.append(accuracy_row)

            report_data[gannma_3_name][kl_epoch_name][dataset_name] = {
                "classification_report": accuracy_row,
                "arguemts": arguemts_df
            }

    # Aggregate all IEMO and MELD DataFrames
    iemodf = pd.concat(iemodf_list, ignore_index=True) if iemodf_list else pd.DataFrame()
    melddf = pd.concat(melddf_list, ignore_index=True) if melddf_list else pd.DataFrame()

    # Example usage: print structure
    # print("IEMO Classification Reports Shape:", iemodf.shape)
    # print("MELD Classification Reports Shape:", melddf.shape)

    # Optional save
    iemodf.to_csv("iemodf_all.csv", index=False)
    melddf.to_csv("melddf_all.csv", index=False)  # optional output