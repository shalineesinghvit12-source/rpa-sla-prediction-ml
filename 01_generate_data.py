"""
01_generate_data.py
Generates a realistic RPA (Robotic Process Automation) operations dataset.

Rationale: no sufficiently granular public dataset on RPA bot run-level
performance exists on Kaggle/UCI, so this dataset is synthesized to reflect
realistic distributions/relationships found in enterprise RPA deployments
(Blue Prism / UiPath style process metrics), based on domain knowledge of
RPA operations (bot run times, exception rates, SLA outcomes, etc).
It intentionally includes messiness (missing values, outliers, inconsistent
text casing, duplicates) so the cleaning step of the pipeline is meaningful.

Converted from R (01_generate_data.R) to Python.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(441)
n = 620  # will drop some to duplicates/cleaning, ending comfortably above 100

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

departments = ["Finance", "HR", "Procurement", "Order Processing", "IT Support", "Customer Service"]
regions = ["US", "UK", "APAC"]
process_types = ["Invoice Processing", "Data Entry", "Report Generation",
                  "Email Triage", "Order Validation", "Reconciliation"]
complexity_levels = ["Low", "Medium", "High"]
bots = [f"Bot_{i}" for i in range(1, 13)]
days = ["Mon", "Tue", "Wed", "Thu", "Fri"]

dept = np.random.choice(departments, n, p=[.22, .12, .14, .20, .18, .14])
region = np.random.choice(regions, n, p=[.45, .30, .25])
ptype = np.random.choice(process_types, n)
complexity = np.random.choice(complexity_levels, n, p=[.4, .4, .2])
bot_id = np.random.choice(bots, n)
day_of_week = np.random.choice(days, n)

complexity_map = {"Low": 1, "Medium": 2, "High": 3}
complexity_num = np.array([complexity_map[c] for c in complexity])

# Transactions processed per run - depends on complexity (simpler processes batch more)
transactions_processed = np.round(np.random.normal(250 - complexity_num * 40, 40)).astype(int)
low_mask = transactions_processed < 5
transactions_processed[low_mask] = np.random.randint(5, 21, size=low_mask.sum())

# Average handling time per transaction (seconds) - higher complexity = slower
avg_handling_time_sec = np.round(np.random.normal(8 + complexity_num * 6, 3), 2)
avg_handling_time_sec[avg_handling_time_sec < 1] = 1.5

# Data volume processed (MB) -- R: rgamma(n, shape=2, rate=0.15); numpy uses scale = 1/rate
data_volume_mb = np.round(np.random.gamma(shape=2, scale=1 / 0.15, size=n) + complexity_num * 3, 2)

# Exceptions - Poisson, rate increases with complexity and data volume
exception_lambda = 0.6 + complexity_num * 0.9 + data_volume_mb / 60
exception_count = np.random.poisson(exception_lambda)

# CPU utilization during run (%)
cpu_utilization_pct = np.round(
    np.clip(np.random.normal(35 + complexity_num * 8, 12), 5, 99), 1
)

# Queue wait time before bot picks up job (seconds) -- R: rexp(n, rate=1/45); numpy uses scale=45
queue_wait_time_sec = np.round(np.random.exponential(scale=45, size=n), 1)

# Manual time this run would have taken a human (minutes) -> basis for time saved
manual_equiv_min = np.round(transactions_processed * (0.8 + complexity_num * 0.4) / 60 * 60, 1)
manual_time_saved_min = np.round(
    np.maximum(
        0,
        manual_equiv_min - (transactions_processed * avg_handling_time_sec) / 60
        + np.random.normal(0, 5, n),
    ),
    1,
)

# SLA outcome (classification target): more exceptions, high queue wait, high complexity -> more likely to miss SLA
logit = (
    -2.2
    + 0.55 * exception_count
    + 0.015 * queue_wait_time_sec
    + 0.5 * complexity_num
    + 0.01 * (cpu_utilization_pct - 40)
    - 0.004 * manual_time_saved_min
)
prob_breach = 1 / (1 + np.exp(-logit))
sla_breach = np.random.binomial(1, prob_breach)
sla_met = np.where(sla_breach == 1, "No", "Yes")

run_date = pd.Timestamp("2026-01-05") + pd.to_timedelta(
    np.random.randint(0, 151, size=n), unit="D"
)

df = pd.DataFrame({
    "process_id": [f"RPA-{i:05d}" for i in range(1, n + 1)],
    "run_date": run_date,
    "department": dept,
    "region": region,
    "process_type": ptype,
    "complexity": complexity,
    "bot_id": bot_id,
    "day_of_week": day_of_week,
    "transactions_processed": transactions_processed,
    "avg_handling_time_sec": avg_handling_time_sec,
    "data_volume_mb": data_volume_mb,
    "exception_count": exception_count,
    "cpu_utilization_pct": cpu_utilization_pct,
    "queue_wait_time_sec": queue_wait_time_sec,
    "manual_time_saved_min": manual_time_saved_min,
    "sla_met": sla_met,
})

# ---- Inject realistic messiness for the cleaning step ----

# 1. Inconsistent text casing / whitespace in categorical columns
messy_idx = np.random.choice(n, 60, replace=False)
df.loc[df.index[messy_idx[0:20]], "department"] = df.loc[df.index[messy_idx[0:20]], "department"].str.upper()
df.loc[df.index[messy_idx[20:40]], "department"] = " " + df.loc[df.index[messy_idx[20:40]], "department"].str.lower() + " "
df.loc[df.index[messy_idx[40:60]], "process_type"] = df.loc[df.index[messy_idx[40:60]], "process_type"].str.upper()


def set_na(series, pct):
    idx = np.random.choice(series.index, size=int(np.floor(len(series) * pct)), replace=False)
    series = series.copy()
    series.loc[idx] = np.nan
    return series


# 2. Missing values (~5%) scattered across several columns
df["avg_handling_time_sec"] = set_na(df["avg_handling_time_sec"], 0.05)
df["cpu_utilization_pct"] = set_na(df["cpu_utilization_pct"], 0.04)
df["data_volume_mb"] = set_na(df["data_volume_mb"], 0.03)
df["manual_time_saved_min"] = set_na(df["manual_time_saved_min"], 0.04)
df["department"] = set_na(df["department"], 0.02)

# 3. Outliers - a handful of extreme/erroneous values
out_idx = np.random.choice(n, 8, replace=False)
df.loc[df.index[out_idx[0:4]], "avg_handling_time_sec"] *= 12  # runaway bot / stuck process
df.loc[df.index[out_idx[4:8]], "exception_count"] += np.random.randint(15, 26, size=4)  # bulk failure event

# 4. Duplicate rows (common in log exports)
dup_rows = df.sample(15, random_state=441)
df = pd.concat([df, dup_rows], ignore_index=True)

# shuffle
df = df.sample(frac=1).reset_index(drop=True)

out_path = os.path.join(DATA_DIR, "rpa_process_runs_raw.csv")
df.to_csv(out_path, index=False)
print("Rows:", df.shape[0], " Cols:", df.shape[1])
print("Saved raw dataset to", out_path)
print(df.info())
