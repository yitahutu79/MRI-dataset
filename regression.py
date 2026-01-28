#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import nibabel as nib
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


# -----------------------------
# Utilities
# -----------------------------
def log(msg: str):
    print(f"[{datetime.now().strftime('%F %T')}] {msg}")

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_date_any(x):
    """Parse many date formats to pandas Timestamp; return NaT if fail."""
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x)
    s = str(x).strip()
    if not s:
        return pd.NaT
    # common ADNI formats: 2017-11-30, 11/30/2017, 2017-11-30_14_44_30.0
    s2 = s.replace("_", " ").replace(".0", "")
    try:
        return pd.to_datetime(s2, errors="coerce", infer_datetime_format=True)
    except Exception:
        return pd.NaT

def extract_date_from_tractout_name(name: str):
    """
    tract_out_<SUBJ>__<DATE>__<Ixxxx>
    where DATE like 2017-11-30_14_44_30.0
    We return date (YYYY-MM-DD) Timestamp and full datetime Timestamp if possible.
    """
    m = re.match(r"^tract_out_(.+?)__(\d{4}-\d{2}-\d{2}[^_]*)__I\d+", name)
    if not m:
        # try more permissive: tract_out_SUBJ__2017-11-30_...__Ixxxx
        m = re.match(r"^tract_out_(.+?)__(\d{4}-\d{2}-\d{2}.*)__I\d+", name)
    if not m:
        return None, pd.NaT, pd.NaT

    subj = m.group(1)
    date_str = m.group(2)
    dt = parse_date_any(date_str)
    d = pd.NaT if pd.isna(dt) else pd.Timestamp(dt.date())
    return subj, d, dt

def count_streamlines_trk(trk_path: str) -> float:
    try:
        if not trk_path or (not os.path.exists(trk_path)):
            return np.nan
        obj = nib.streamlines.load(trk_path)
        return float(len(obj.tractogram.streamlines))
    except Exception:
        return np.nan

def run_univariate_ols(df: pd.DataFrame, y_col: str, x_col: str):
    """OLS: y ~ x. Return beta, p, n"""
    y = pd.to_numeric(df[y_col], errors="coerce")
    x = pd.to_numeric(df[x_col], errors="coerce")
    tmp = pd.DataFrame({"y": y, "x": x}).dropna()
    n = len(tmp)
    if n < 20:
        return np.nan, np.nan, n
    X = sm.add_constant(tmp["x"].values)
    model = sm.OLS(tmp["y"].values, X).fit()
    return float(model.params[1]), float(model.pvalues[1]), int(n)


# -----------------------------
# MoCA auto-detect
# -----------------------------
def guess_subject_column(df: pd.DataFrame):
    """
    Try find a column that looks like ADNI PTID (###_S_####) or similar.
    Fallback to RID if exists.
    """
    cols = list(df.columns)

    # direct name hits
    for c in cols:
        cu = c.upper()
        if cu in {"PTID", "SUBJECT", "SUBJ", "SUBJID", "SUBJECTID"}:
            return c

    # value pattern hits: ###_S_####
    pat = re.compile(r"^\d{3}_S_\d{4,5}$")
    best = None
    best_hits = 0
    for c in cols:
        vals = df[c].dropna().astype(str).str.strip()
        if len(vals) == 0:
            continue
        hits = vals.apply(lambda x: bool(pat.match(x))).sum()
        if hits > best_hits:
            best_hits = hits
            best = c
    if best and best_hits >= 5:
        return best

    # RID fallback
    for c in cols:
        if c.upper() == "RID":
            return c

    return None

def guess_date_column(df: pd.DataFrame):
    cols = list(df.columns)
    # direct name hits
    name_candidates = []
    for c in cols:
        cu = c.upper()
        if "EXAM" in cu and "DATE" in cu:
            name_candidates.append(c)
        elif cu in {"EXAMDATE", "VISDATE", "SCANDATE", "DATE"}:
            name_candidates.append(c)
        elif "DATE" in cu:
            name_candidates.append(c)
    # pick the one with most parseable dates
    best = None
    best_ok = 0
    for c in name_candidates:
        dt = df[c].apply(parse_date_any)
        ok = dt.notna().sum()
        if ok > best_ok:
            best_ok = ok
            best = c
    return best

def guess_moca_columns(df: pd.DataFrame):
    """
    Find MoCA columns in ADNI tables.
    Common: 'MOCA', 'MOCATOT', sometimes 'MCA...' (PPMI-style, but ADNI usually MOCA).
    We keep numeric columns with enough non-NaNs.
    """
    cols = list(df.columns)
    moca_like = []
    for c in cols:
        cu = c.upper()
        if cu == "MOCA" or "MOCA" in cu or cu.startswith("MCA"):
            moca_like.append(c)

    # exclude date/location-ish columns if any
    bad = {"MCADATE", "MCAMONTH", "MCAYR", "MCADAY", "MCAPLACE", "MCACITY"}
    moca_like = [c for c in moca_like if c.upper() not in bad]

    cleaned = []
    for c in moca_like:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() >= 10:
            cleaned.append(c)

    # prefer total first if present
    for totname in ["MOCA", "MOCATOT", "MCATOT"]:
        if totname in [c.upper() for c in cleaned]:
            # bring the exact-matched column to front
            exact = [c for c in cleaned if c.upper() == totname]
            rest = [c for c in cleaned if c.upper() != totname]
            cleaned = exact + rest
            break
    return cleaned


# -----------------------------
# Bundle tables
# -----------------------------
def build_long_from_tract_out(tract_root: Path, cache_csv: Path = None):
    """
    Scan TRACT_OUT/tract_out_* and compute streamline counts for bundles_recob/*.trk
    Return long dataframe:
    SUBJ, SCAN_DATE, SCAN_DATETIME, I, TRACT_OUT, BUNDLE, TRK_PATH, STREAMLINE_COUNT
    """
    tract_root = Path(tract_root)
    out_dirs = sorted([p for p in tract_root.iterdir() if p.is_dir() and p.name.startswith("tract_out_")])

    # optional cache (speeds up re-run)
    cache = None
    if cache_csv:
        if cache_csv.exists():
            try:
                cache = pd.read_csv(cache_csv)
            except Exception:
                cache = None
        if cache is None:
            cache = pd.DataFrame(columns=["TRK_PATH", "MTIME", "STREAMLINE_COUNT"])
        cache_map = {r["TRK_PATH"]: (r["MTIME"], r["STREAMLINE_COUNT"]) for _, r in cache.iterrows()}
    else:
        cache_map = {}

    rows = []
    for od in out_dirs:
        subj, scan_date, scan_dt = extract_date_from_tractout_name(od.name)
        if subj is None:
            continue
        # extract Ixxxx
        mi = re.search(r"__(I\d+)$", od.name.replace("tract_out_", ""))
        I = mi.group(1) if mi else ""

        bdir = od / "RecobundlesX" / "bundles_recob"
        if not bdir.exists():
            continue
        trks = sorted(bdir.glob("*.trk"))
        if len(trks) == 0:
            continue

        for t in trks:
            bundle = t.stem
            trk_path = str(t)

            # cache check
            mtime = os.path.getmtime(trk_path)
            if trk_path in cache_map:
                old_mtime, old_cnt = cache_map[trk_path]
                if (not pd.isna(old_mtime)) and float(old_mtime) == float(mtime):
                    cnt = float(old_cnt)
                else:
                    cnt = count_streamlines_trk(trk_path)
                    cache_map[trk_path] = (mtime, cnt)
            else:
                cnt = count_streamlines_trk(trk_path)
                cache_map[trk_path] = (mtime, cnt)

            rows.append({
                "SUBJ": subj,
                "SCAN_DATE": scan_date,
                "SCAN_DATETIME": scan_dt,
                "I": I,
                "TRACT_OUT": str(od),
                "BUNDLE": bundle,
                "TRK_PATH": trk_path,
                "STREAMLINE_COUNT": cnt
            })

    long_df = pd.DataFrame(rows)

    # write back cache
    if cache_csv:
        cache_out = pd.DataFrame([
            {"TRK_PATH": k, "MTIME": v[0], "STREAMLINE_COUNT": v[1]}
            for k, v in cache_map.items()
        ])
        cache_out.to_csv(cache_csv, index=False)

    return long_df

def build_wide_from_long(long_df: pd.DataFrame):
    # total
    total = long_df.groupby(["SUBJ", "SCAN_DATE"], as_index=False)["STREAMLINE_COUNT"].sum()
    total = total.rename(columns={"STREAMLINE_COUNT": "BUNDLE_TOTAL_STREAMLINES"})

    wide = long_df.pivot_table(
        index=["SUBJ", "SCAN_DATE"],
        columns="BUNDLE",
        values="STREAMLINE_COUNT",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    wide = wide.merge(total, on=["SUBJ", "SCAN_DATE"], how="left")

    bundle_cols = [c for c in wide.columns if c not in ["SUBJ", "SCAN_DATE", "BUNDLE_TOTAL_STREAMLINES"]]
    for c in bundle_cols:
        wide[f"{c}__PCT"] = np.where(
            wide["BUNDLE_TOTAL_STREAMLINES"] > 0,
            wide[c] / wide["BUNDLE_TOTAL_STREAMLINES"],
            np.nan
        )
    return wide


# -----------------------------
# Merge MoCA by nearest date
# -----------------------------
def merge_by_nearest_date(bundle_wide: pd.DataFrame, moca_df: pd.DataFrame, subj_col: str, date_col: str, tol_days: int = 30):
    """
    For each (SUBJ, SCAN_DATE) in bundle_wide, attach nearest MoCA row within +-tol_days.
    """
    bw = bundle_wide.copy()
    bw["SCAN_DATE"] = pd.to_datetime(bw["SCAN_DATE"], errors="coerce")
    bw = bw.dropna(subset=["SCAN_DATE"])

    md = moca_df.copy()
    md[subj_col] = md[subj_col].astype(str).str.strip()
    md[date_col] = md[date_col].apply(parse_date_any)
    md = md.dropna(subset=[subj_col, date_col])

    # normalize subject format if it's RID (numeric)
    # If subj_col is RID, we cannot match to 003_S_XXXX unless you have mapping; we warn later.
    # Here we only match string-equal.
    tol = pd.Timedelta(days=int(tol_days))

    merged_rows = []
    md_group = {k: g.sort_values(date_col) for k, g in md.groupby(subj_col)}

    for _, r in bw.iterrows():
        subj = str(r["SUBJ"]).strip()
        d = pd.Timestamp(r["SCAN_DATE"])

        if subj not in md_group:
            # no match
            out = r.to_dict()
            out["_MOCA_MATCHED"] = 0
            out["_MOCA_DAYS_DIFF"] = np.nan
            merged_rows.append(out)
            continue

        g = md_group[subj]
        # find nearest
        deltas = (g[date_col] - d).abs()
        i = deltas.idxmin()
        best_delta = deltas.loc[i]
        if pd.isna(best_delta) or best_delta > tol:
            out = r.to_dict()
            out["_MOCA_MATCHED"] = 0
            out["_MOCA_DAYS_DIFF"] = float(best_delta / pd.Timedelta(days=1)) if not pd.isna(best_delta) else np.nan
            merged_rows.append(out)
            continue

        out = r.to_dict()
        mrow = g.loc[i].to_dict()
        # attach all MoCA columns (except subject/date)
        for k, v in mrow.items():
            if k in {subj_col, date_col}:
                continue
            out[k] = v
        out["_MOCA_MATCHED"] = 1
        out["_MOCA_DAYS_DIFF"] = float(best_delta / pd.Timedelta(days=1))
        merged_rows.append(out)

    return pd.DataFrame(merged_rows)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tract_out", required=True, help="Path to ADNI-DTI/TRACT_OUT")
    ap.add_argument("--moca", required=True, nargs="+", help="MoCA table(s): MOCA.csv / MoCA_ADNI.csv etc.")
    ap.add_argument("--outdir", default="adni_bundle_moca_out", help="Output directory")
    ap.add_argument("--tol_days", type=int, default=30, help="Nearest-date merge tolerance in days")
    args = ap.parse_args()

    tract_root = Path(args.tract_out)
    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    # 1) build bundle long/wide
    log(f"Scanning bundles under: {tract_root}")
    cache_csv = outdir / "trk_count_cache.csv"
    long_df = build_long_from_tract_out(tract_root, cache_csv=cache_csv)
    if len(long_df) == 0:
        raise SystemExit("No bundles found. Check TRACT_OUT structure and bundles_recob outputs.")

    long_path = outdir / "adni_bundle_long.csv"
    long_df.to_csv(long_path, index=False)
    log(f"Saved: {long_path} (rows={len(long_df)})")

    wide_df = build_wide_from_long(long_df)
    wide_path = outdir / "adni_bundle_wide.csv"
    wide_df.to_csv(wide_path, index=False)
    log(f"Saved: {wide_path} (rows={len(wide_df)})")

    # 2) read & concat MoCA tables
    moca_all = []
    for p in args.moca:
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        df["_SRC_FILE"] = os.path.basename(p)
        moca_all.append(df)
        log(f"Loaded MoCA table: {p} (rows={len(df)})")
    moca_df = pd.concat(moca_all, axis=0, ignore_index=True)

    subj_col = guess_subject_column(moca_df)
    date_col = guess_date_column(moca_df)
    if subj_col is None or date_col is None:
        log("ERROR: Cannot auto-detect subject/date columns in MoCA tables.")
        log(f"Detected subj_col={subj_col}, date_col={date_col}")
        log(f"Columns: {list(moca_df.columns)[:80]}")
        raise SystemExit(1)

    log(f"MoCA subject col = {subj_col}")
    log(f"MoCA date col    = {date_col}")

    moca_cols = guess_moca_columns(moca_df)
    if len(moca_cols) == 0:
        log("WARN: No numeric MoCA-like columns detected (MOCA/MOCATOT/...). You can still merge but regression will skip.")
    else:
        log(f"Detected MoCA columns (first 10): {moca_cols[:10]} (total={len(moca_cols)})")

    # 3) merge by nearest date
    merged = merge_by_nearest_date(wide_df, moca_df, subj_col=subj_col, date_col=date_col, tol_days=args.tol_days)
    merged_path = outdir / "adni_bundle_moca_wide_merged.csv"
    merged.to_csv(merged_path, index=False)
    log(f"Saved: {merged_path} (rows={len(merged)})")
    log(f"MoCA matched rows: {int((merged['_MOCA_MATCHED']==1).sum())} / {len(merged)}")

    # 4) regressions
    if len(moca_cols) == 0:
        log("No MoCA columns -> skip regression.")
        return

    # choose X columns
    raw_bundle_cols = [c for c in wide_df.columns if c not in ["SUBJ", "SCAN_DATE", "BUNDLE_TOTAL_STREAMLINES"] and not c.endswith("__PCT")]
    pct_bundle_cols = [f"{c}__PCT" for c in raw_bundle_cols if f"{c}__PCT" in merged.columns]

    results = []
    for y in moca_cols:
        for b in raw_bundle_cols:
            beta, p, n = run_univariate_ols(merged, y, b)
            results.append(["ADNI", "COUNT", y, b, beta, p, n])
        for b in pct_bundle_cols:
            beta, p, n = run_univariate_ols(merged, y, b)
            results.append(["ADNI", "PCT", y, b, beta, p, n])

    res = pd.DataFrame(results, columns=["GROUP", "X_TYPE", "MOCA_Y", "BUNDLE_X", "BETA", "P", "N"])
    res["P"] = pd.to_numeric(res["P"], errors="coerce")
    res["Q_FDR"] = np.nan

    for (xt, y), idx in res.groupby(["X_TYPE", "MOCA_Y"]).groups.items():
        pvals = res.loc[idx, "P"].values
        ok = np.isfinite(pvals)
        if ok.sum() >= 5:
            q = np.full_like(pvals, np.nan, dtype=float)
            q[ok] = multipletests(pvals[ok], method="fdr_bh")[1]
            res.loc[idx, "Q_FDR"] = q

    res_path = outdir / "adni_bundle_moca_univariate_ols_with_fdr.csv"
    res.to_csv(res_path, index=False)
    log(f"Saved: {res_path} (rows={len(res)})")

    # top10 per MoCA_y per X_TYPE
    top_rows = []
    sub = res.dropna(subset=["Q_FDR"])
    if len(sub) > 0:
        for (xt, y), gdf in sub.groupby(["X_TYPE", "MOCA_Y"]):
            top_rows.append(gdf.sort_values("Q_FDR").head(10))
        top = pd.concat(top_rows, axis=0)
        top_path = outdir / "adni_top10_per_moca.csv"
        top.to_csv(top_path, index=False)
        log(f"Saved: {top_path} (rows={len(top)})")
    else:
        log("No enough p-values for FDR/top10 (or too few valid samples).")

    log("DONE ✅")


if __name__ == "__main__":
    main()
