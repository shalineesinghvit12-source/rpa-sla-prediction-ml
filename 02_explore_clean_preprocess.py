"""
02_explore_clean_preprocess.py
Explores, cleans, and preprocesses the raw RPA operations dataset.
Converted from R (02_explore_clean_preprocess.R) to Python.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
PLOTS_DIR = os.path.join(BASE, "plots")
OUTPUTS_DIR = os.path.join(BASE, "outputs")
for d in (DATA_DIR, PLOTS_DIR, OUTPUTS_DIR):
    os.makedirs(d, exist_ok=True)

log_path = os.path.join(OUTPUTS_DIR, "02_log.txt")
log_file = open(log_path, "w")


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    log_file.write(msg + "\n")


df = pd.read_csv(os.path.join(DATA_DIR, "rpa_process_runs_raw.csv"))
df["run_date"] = pd.to_datetime(df["run_date"])

num_cols = ["transactions_processed", "avg_handling_time_sec", "data_volume_mb",
            "exception_count", "cpu_utilization_pct", "queue_wait_time_sec",
            "manual_time_saved_min"]

log("================ b. DATA EXPLORATION (RAW) ================")
log("Dimensions:", df.shape[0], "rows x", df.shape[1], "cols\n")
log("---- Structure ----")
log(df.dtypes.to_string())
log("\n---- Summary (numeric) ----")
log(df[num_cols].describe().to_string())

log("\n---- Categorical frequency (raw, shows messiness) ----")
log(df["department"].value_counts(dropna=False).to_string())
log(df["process_type"].value_counts(dropna=False).to_string())
log(df["sla_met"].value_counts(dropna=False).to_string())

log("\n---- Missing values per column ----")
log(df.isna().sum().to_string())

log("\n---- Duplicate rows (excluding process_id) ----")
log("Duplicated (all cols):", df.duplicated().sum())
log("Duplicated (excl. process_id):", df.drop(columns=["process_id"]).duplicated().sum())

log("\n---- Correlation matrix (numeric, pairwise complete) ----")
log(df[num_cols].corr().round(2).to_string())

# ---- Exploration plots ----
plt.figure(figsize=(9, 6.5))
plt.hist(df["avg_handling_time_sec"].dropna(), bins=40, color="#4C72B0", edgecolor="white")
plt.title("Distribution: Avg Handling Time per Transaction (raw)")
plt.xlabel("Seconds")
plt.savefig(os.path.join(PLOTS_DIR, "hist_handling_time.png"), dpi=130)
plt.close()

plt.figure(figsize=(9, 6.5))
plt.hist(df["exception_count"], bins=20, color="#DD8452", edgecolor="white")
plt.title("Distribution: Exception Count per Run (raw)")
plt.xlabel("Exceptions")
plt.savefig(os.path.join(PLOTS_DIR, "hist_exceptions.png"), dpi=130)
plt.close()

plt.figure(figsize=(10, 6.5))
df.boxplot(column="transactions_processed", by="department", rot=90)
plt.title("Transactions Processed by Department (raw)")
plt.suptitle("")
plt.ylabel("Transactions")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "box_dept_transactions.png"), dpi=130)
plt.close()

plt.figure(figsize=(9, 6.5))
plt.scatter(df["data_volume_mb"], df["avg_handling_time_sec"], alpha=0.5, color="#8172B2")
plt.title("Handling Time vs Data Volume (raw)")
plt.xlabel("Data Volume (MB)")
plt.ylabel("Avg Handling Time (sec)")
plt.savefig(os.path.join(PLOTS_DIR, "scatter_exceptions_handling.png"), dpi=130)
plt.close()

plt.figure(figsize=(9, 7.5))
cm_raw = df[num_cols].corr()
plt.imshow(cm_raw, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(num_cols)), num_cols, rotation=90)
plt.yticks(range(len(num_cols)), num_cols)
plt.title("Correlation Heatmap (numeric features)")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "corr_heatmap.png"), dpi=130)
plt.close()

log("\nExploration plots saved to", PLOTS_DIR)

################################################################
log("\n\n================ c. DATA CLEANING ================")

df_clean = df.copy()

# 1. Standardized categorical text: trim whitespace + map to canonical category names.
#    (Using an explicit lookup rather than generic title-casing avoids edge cases like
#    "IT Support" -> "It Support" or acronyms getting mangled.)
department_levels = ["Finance", "HR", "Procurement", "Order Processing", "IT Support", "Customer Service"]
process_levels = ["Invoice Processing", "Data Entry", "Report Generation",
                   "Email Triage", "Order Validation", "Reconciliation"]


def canon_map(series, canon_levels):
    lookup = {re.sub(r"[^a-z ]", "", lvl.lower()): lvl for lvl in canon_levels}

    def _map(x):
        if pd.isna(x):
            return x
        key = re.sub(r"[^a-z ]", "", str(x).strip().lower())
        return lookup.get(key, x)

    return series.apply(_map)


df_clean["department"] = canon_map(df_clean["department"], department_levels)
df_clean["process_type"] = canon_map(df_clean["process_type"], process_levels)
log("After standardizing text case/whitespace:")
log(df_clean["department"].value_counts(dropna=False).to_string())

# 2. Remove duplicate rows (based on all columns except process_id, since duplicate
#    log entries share every operational attribute)
before = len(df_clean)
df_clean = df_clean[~df_clean.drop(columns=["process_id"]).duplicated(keep="first")]
log(f"\nRemoved {before - len(df_clean)} duplicate rows. Rows now: {len(df_clean)}")

# 3. Missing values
#    - department (categorical, ~2% missing): impute with mode (safe, low information loss)
#    - numeric columns: impute with median WITHIN a related group when possible,
#      falling back to global median, since RPA runs vary systematically by process
mode_val = df_clean["department"].mode(dropna=True)[0]
df_clean["department"] = df_clean["department"].fillna(mode_val)


def impute_grouped_median(series, group):
    global_median = series.median()
    group_median = series.groupby(group).transform(
        lambda s: s.median() if not pd.isna(s.median()) else global_median
    )
    return series.fillna(group_median).fillna(global_median)


df_clean["avg_handling_time_sec"] = impute_grouped_median(df_clean["avg_handling_time_sec"], df_clean["process_type"])
df_clean["cpu_utilization_pct"] = impute_grouped_median(df_clean["cpu_utilization_pct"], df_clean["department"])
df_clean["data_volume_mb"] = impute_grouped_median(df_clean["data_volume_mb"], df_clean["process_type"])
df_clean["manual_time_saved_min"] = impute_grouped_median(df_clean["manual_time_saved_min"], df_clean["process_type"])

log("\nMissing values after imputation:")
log(df_clean.isna().sum().to_string())

# 4. Outlier treatment: IQR-based capping (winsorizing) on right-skewed operational metrics.
#    We cap rather than delete, since these rows still carry valid info on other columns
#    and RPA logs legitimately contain occasional slow runs / failure bursts.


def cap_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr  # use 3x IQR ("extreme" outlier) to only touch true anomalies
    n_out = int(((series < lower) | (series > upper)).sum())
    capped = series.clip(lower=lower, upper=upper)
    return capped, n_out


for col in ["avg_handling_time_sec", "exception_count", "queue_wait_time_sec", "data_volume_mb"]:
    capped, n_out = cap_outliers(df_clean[col])
    log(f"Column '{col}': capped {n_out} extreme outlier(s)")
    df_clean[col] = capped

# Sanity re-check
log("\n---- Summary after cleaning ----")
log(df_clean[num_cols].describe().to_string())
log("\nAny remaining NAs?", bool(df_clean.isna().any().any()))

plt.figure(figsize=(9, 6.5))
plt.hist(df_clean["avg_handling_time_sec"], bins=40, color="#4C72B0", edgecolor="white")
plt.title("Avg Handling Time per Transaction (cleaned)")
plt.xlabel("Seconds")
plt.savefig(os.path.join(PLOTS_DIR, "hist_handling_time_clean.png"), dpi=130)
plt.close()

clean_path = os.path.join(DATA_DIR, "rpa_process_runs_clean.csv")
df_clean.to_csv(clean_path, index=False)
log(f"\nCleaned dataset saved. Rows: {df_clean.shape[0]}  Cols: {df_clean.shape[1]}")

################################################################
log("\n\n================ d. DATA PREPROCESSING ================")

df_prep = df_clean.copy()

# Ordinal encode complexity (has natural order Low<Medium<High)
complexity_order = ["Low", "Medium", "High"]
complexity_map = {lvl: i + 1 for i, lvl in enumerate(complexity_order)}
df_prep["complexity_num"] = df_prep["complexity"].map(complexity_map)

# Bin avg_handling_time_sec into a categorical speed tier (useful for reporting / exploration)
df_prep["handling_speed_tier"] = pd.qcut(
    df_prep["avg_handling_time_sec"], q=[0, .33, .66, 1], labels=["Fast", "Moderate", "Slow"]
)

# Z-score normalize numeric predictors (needed for distance-based clustering / kNN-style methods)
z_cols = ["transactions_processed", "avg_handling_time_sec", "data_volume_mb",
          "exception_count", "cpu_utilization_pct", "queue_wait_time_sec", "manual_time_saved_min"]
for c in z_cols:
    df_prep[f"z_{c}"] = (df_prep[c] - df_prep[c].mean()) / df_prep[c].std()

# One-hot / dummy encode categorical predictors used for modeling (department, region, process_type, day_of_week)
dummy_cols = ["department", "region", "process_type", "day_of_week"]
dummies = pd.get_dummies(df_prep[dummy_cols], prefix=dummy_cols)
dummies.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in dummies.columns]
df_prep = pd.concat([df_prep, dummies], axis=1)

log("Added ordinal complexity encoding, binned handling-speed tier,")
log("z-score standardized numeric features (prefixed z_), and one-hot dummy variables for:")
log("\n".join(f" - {c}" for c in dummy_cols))
log(f"\nNew feature count: {df_prep.shape[1]} (was {df_clean.shape[1]})")
log("\nSpeed tier distribution:")
log(df_prep["handling_speed_tier"].value_counts().to_string())

prep_path = os.path.join(DATA_DIR, "rpa_process_runs_preprocessed.csv")
df_prep.to_csv(prep_path, index=False)
log("\nPreprocessed dataset saved.")

log_file.close()
print(f"Done. See {log_path}")
