
import io
import json
import math
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

# ----------------------------
# Utilities
# ----------------------------

def parse_datetime_safe(s):
    if pd.isna(s):
        return pd.NaT
    # Try multiple common formats, fallback to pandas
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(str(s), fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

def ensure_sorted_events(df, case_col, act_col, ts_col):
    d = df.copy()
    d[ts_col] = d[ts_col].apply(parse_datetime_safe)
    d = d.dropna(subset=[case_col, act_col, ts_col])
    d = d.sort_values([case_col, ts_col, act_col]).reset_index(drop=True)
    return d

def build_variants(df, case_col, act_col, ts_col):
    d = ensure_sorted_events(df, case_col, act_col, ts_col)
    seqs = d.groupby(case_col)[act_col].apply(list)
    variant_counts = seqs.value_counts().reset_index()
    variant_counts.columns = ["variant", "count"]
    variant_counts["variant_str"] = variant_counts["variant"].apply(lambda x: " → ".join(x))
    total_cases = variant_counts["count"].sum()
    variant_counts["percent"] = 100 * variant_counts["count"] / total_cases if total_cases else 0
    return variant_counts, seqs

def directly_follows(df, case_col, act_col, ts_col):
    d = ensure_sorted_events(df, case_col, act_col, ts_col)
    edges = {}
    for case, group in d.groupby(case_col):
        acts = group[act_col].tolist()
        for i in range(len(acts)-1):
            edge = (acts[i], acts[i+1])
            edges[edge] = edges.get(edge, 0) + 1
    return edges

def activity_durations(df, case_col, act_col, ts_col):
    """Compute average time spent in each activity as time until next event within each case."""
    d = ensure_sorted_events(df, case_col, act_col, ts_col)
    records = []
    for case, g in d.groupby(case_col):
        ts = g[ts_col].tolist()
        acts = g[act_col].tolist()
        for i in range(len(acts)-1):
            delta = (ts[i+1] - ts[i]).total_seconds()
            if delta >= 0:
                records.append({"activity": acts[i], "duration_s": delta})
    if not records:
        return pd.DataFrame(columns=["activity", "count", "avg_hours", "median_hours"])
    out = pd.DataFrame(records)
    agg = out.groupby("activity")["duration_s"].agg(["count", "mean", "median"]).reset_index()
    agg["avg_hours"] = agg["mean"] / 3600.0
    agg["median_hours"] = agg["median"] / 3600.0
    return agg[["activity", "count", "avg_hours", "median_hours"]].sort_values("avg_hours", ascending=False)

def case_cycle_times(df, case_col, ts_col):
    d = ensure_sorted_events(df, case_col, ts_col, ts_col)  # reuse sorter even if act_col unused
    agg = d.groupby(case_col)[ts_col].agg(["min", "max"]).reset_index()
    agg["cycle_time_hours"] = (agg["max"] - agg["min"]).dt.total_seconds() / 3600.0
    return agg

def edit_distance(a: List[str], b: List[str]) -> int:
    """Levenshtein distance between two activity sequences."""
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m]

def conformance_scores(seqs: pd.Series, ideal: List[str]) -> pd.DataFrame:
    rows = []
    for case_id, seq in seqs.items():
        dist = edit_distance(seq, ideal)
        rows.append({"case_id": case_id, "distance": dist, "length": len(seq)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["normalized"] = out.apply(lambda r: r["distance"] / max(1, r["length"]), axis=1)
    return out

def estimate_simulation_effect(df, case_col, act_col, ts_col, target_act, reduction_pct):
    """Rudimentary what-if: reduce durations for a chosen activity by X% and estimate new average cycle time."""
    cycles = case_cycle_times(df, case_col, ts_col)
    base_avg = cycles["cycle_time_hours"].mean() if not cycles.empty else 0.0

    # Estimate time spent per case in target_act as share of activity durations *within that case* (approximate).
    d = ensure_sorted_events(df, case_col, act_col, ts_col)
    # Compute durations between events and tag activity
    rows = []
    for case, g in d.groupby(case_col):
        ts = g[ts_col].tolist()
        acts = g[act_col].tolist()
        for i in range(len(acts)-1):
            delta_h = max(0.0, (ts[i+1] - ts[i]).total_seconds()/3600.0)
            rows.append({"case": case, "activity": acts[i], "hours": delta_h})
    if not rows:
        return base_avg, base_avg
    per = pd.DataFrame(rows)
    per_case = per.groupby("case")["hours"].sum().rename("total").reset_index()
    target = per[per["activity"] == target_act].groupby("case")["hours"].sum().rename("target").reset_index()
    merged = per_case.merge(target, on="case", how="left").fillna({"target": 0.0})
    merged["reduced"] = merged["total"] - merged["target"] * (reduction_pct/100.0)
    new_avg = merged["reduced"].mean()
    return base_avg, new_avg

def to_csv_download(df, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download " + filename, data=csv, file_name=filename, mime="text/csv")


# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(page_title="Process Mining Explorer", layout="wide")
st.title("🧭 Process Mining Explorer")

st.markdown("""
Upload an **event log** (CSV or Excel). Then map columns and explore:
- **Process discovery** (directly-follows graph)
- **Variant analysis**
- **Performance metrics** (cycle time, activity sojourn times)
- **Conformance checking** (compare to an ideal path)
- **Root-cause analysis** (duration by attribute)
- **What-if simulation** (reduce time in an activity)
""")

with st.sidebar:
    st.header("1) Upload & Map Columns")
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    default_case = "Id"
    default_act = "NewValue"
    default_ts = "CreatedOn"

    case_col = st.text_input("Case ID column", value=default_case)
    act_col = st.text_input("Activity/Status column", value=default_act)
    ts_col = st.text_input("Timestamp column", value=default_ts)

    st.caption("Tip: For your CRM export, try Id / NewValue / CreatedOn.")

if not file:
    st.info("Upload an event log to begin.")
    st.stop()

# Load data
try:
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

missing = [c for c in [case_col, act_col, ts_col] if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.write("Columns present:", list(df.columns))
    st.stop()

# Preview
st.subheader("Data Preview")
st.dataframe(df.head(50))

# Clean/sort once
df_sorted = ensure_sorted_events(df, case_col, act_col, ts_col)

# ----------------------------
# Key Metrics
# ----------------------------
st.subheader("📊 Key Metrics")
cycles = case_cycle_times(df_sorted, case_col, ts_col)
avg_cycle = cycles["cycle_time_hours"].mean() if not cycles.empty else 0.0
median_cycle = cycles["cycle_time_hours"].median() if not cycles.empty else 0.0
num_cases = cycles.shape[0]

st.metric("Cases", num_cases)
st.metric("Average Cycle Time (hours)", round(avg_cycle, 2))
st.metric("Median Cycle Time (hours)", round(median_cycle, 2))

# ----------------------------
# Process Discovery (Directly-Follows Graph)
# ----------------------------
st.subheader("🗺️ Process Discovery (Directly-Follows Graph)")
edges = directly_follows(df_sorted, case_col, act_col, ts_col)

if not edges:
    st.info("Not enough data to build a graph.")
else:
    # Build graph
    G = nx.DiGraph()
    for (src, dst), w in edges.items():
        G.add_edge(src, dst, weight=w)

    pos = nx.spring_layout(G, seed=42, k=1.0) if len(G.nodes) > 1 else None

    fig = plt.figure()
    if pos:
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos, font_size=8)
        nx.draw_networkx_edges(G, pos, arrows=True, width=[1 + (w / max(weights)) * 3 for w in weights])
        edge_labels = {(u, v): w for (u, v, w) in [(u, v, data["weight"]) for u, v, data in G.edges(data=True)]}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    else:
        nx.draw(G, with_labels=True)

    st.pyplot(fig)

# ----------------------------
# Variant Analysis
# ----------------------------
st.subheader("🧬 Variant Analysis")
variants_df, seqs = build_variants(df_sorted, case_col, act_col, ts_col)
st.write(f"Unique variants: **{variants_df.shape[0]}**")
st.dataframe(variants_df.head(25))

fig2 = plt.figure()
plt.bar(range(min(20, variants_df.shape[0])), variants_df["count"].head(20))
plt.title("Top Variants (count)")
plt.xlabel("Variant rank")
plt.ylabel("Count")
st.pyplot(fig2)

to_csv_download(variants_df, "variants.csv")

# ----------------------------
# Performance: Activity Sojourns
# ----------------------------
st.subheader("⏱️ Activity Sojourn Times")
acts_df = activity_durations(df_sorted, case_col, act_col, ts_col)
st.dataframe(acts_df)

fig3 = plt.figure()
plt.barh(acts_df["activity"], acts_df["avg_hours"])
plt.xlabel("Average hours in activity")
plt.ylabel("Activity")
plt.title("Average Time Spent per Activity")
st.pyplot(fig3)

to_csv_download(acts_df, "activity_sojourns.csv")

# Cycle time distribution
st.subheader("📦 Case Cycle Time Distribution")
st.dataframe(cycles.head(25))

fig4 = plt.figure()
plt.hist(cycles["cycle_time_hours"], bins=30)
plt.xlabel("Cycle time (hours)")
plt.ylabel("Cases")
plt.title("Distribution of Case Cycle Times")
st.pyplot(fig4)

to_csv_download(cycles, "case_cycle_times.csv")

# ----------------------------
# Conformance Checking
# ----------------------------
st.subheader("✅ Conformance Checking (vs. Ideal Path)")
default_ideal = []
if len(variants_df) > 0:
    default_ideal = variants_df.iloc[0]["variant"]  # most common path
ideal_str = st.text_input("Ideal path (comma-separated activities)", value=", ".join(default_ideal) if default_ideal else "")
ideal_path = [s.strip() for s in ideal_str.split(",")] if ideal_str.strip() else []

if ideal_path:
    conf = conformance_scores(seqs, ideal_path)
    st.dataframe(conf.sort_values("normalized").head(50))
    fig5 = plt.figure()
    plt.hist(conf["normalized"], bins=30)
    plt.xlabel("Normalized edit distance")
    plt.ylabel("Cases")
    plt.title("Conformance Distribution")
    st.pyplot(fig5)
    to_csv_download(conf, "conformance_scores.csv")
else:
    st.info("Enter an ideal path above to compute conformance.")

# ----------------------------
# Root Cause Analysis (by attribute)
# ----------------------------
st.subheader("🪤 Root Cause (Duration by Attribute)")
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in [case_col, act_col]]
if cat_cols:
    chosen = st.selectbox("Choose a categorical column", cat_cols)
    # Merge cycle times back to a case-level attribute (take first value per case for the chosen column)
    case_attr = df_sorted.groupby(case_col)[chosen].agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else None).reset_index()
    merged = cycles.merge(case_attr, left_on=case_col, right_on=case_col, how="left")
    rc = merged.groupby(chosen)["cycle_time_hours"].mean().sort_values(ascending=False).reset_index().dropna()
    st.dataframe(rc.head(50))

    fig6 = plt.figure()
    plt.bar(rc[chosen].astype(str).head(20), rc["cycle_time_hours"].head(20))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Avg cycle time (hours)")
    plt.title(f"Average Cycle Time by {chosen}")
    st.pyplot(fig6)

    to_csv_download(rc, f"root_cause_by_{chosen}.csv")
else:
    st.info("No additional string/categorical columns found for root-cause grouping.")

# ----------------------------
# What-if Simulation
# ----------------------------
st.subheader("🧪 What-if Simulation")
unique_acts = list(df_sorted[act_col].dropna().unique())
if unique_acts:
    target_act = st.selectbox("Activity to optimize", unique_acts)
    reduction = st.slider("Reduce time in this activity by (%)", min_value=0, max_value=90, value=20, step=5)
    base_avg, new_avg = estimate_simulation_effect(df_sorted, case_col, act_col, ts_col, target_act, reduction)
    st.metric("Baseline Avg Cycle Time (hours)", round(base_avg, 2))
    st.metric("Projected Avg Cycle Time (hours)", round(new_avg, 2))
    st.caption("Estimation assumes proportional reduction in time spent in the selected activity across cases.")
else:
    st.info("No activities found.")

st.success("Analysis complete. Explore tabs above and download CSVs where needed.")
