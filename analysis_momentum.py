# analysis_momentum.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import glob, os
import sys

RATING_PATH = 'rating_history.parquet'  # output from engine.py
RESULTS_DIR = 'results_csvs'            # folder with per-strategy csv: {strategy}.csv with date,ret,alpha_ret (optional)

W = 20  # momentum window days
T = 20  # forward window days

def load_rating_history(path=RATING_PATH):
    if not os.path.exists(path):
        print(f"rating file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    df = pd.read_parquet(path)
    # keep only alpha metric snapshots (we snapshot both alpha and tail_survival)
    df = df[df['metric']=='alpha'].copy()
    # ensure datetime
    df['date'] = pd.to_datetime(df['date'])
    # normalize to midnight and drop tz info (safe for daily alignment)
    df['date'] = df['date'].dt.tz_convert(None).dt.normalize() if df['date'].dt.tz is not None else df['date'].dt.normalize()
    return df

def _robust_read_csv(path):
    """
    Read CSV and return a DataFrame with a datetime index called 'date'.
    Handles cases where the CSV has:
      - a 'date' column,
      - an 'index' column,
      - the first column is a date,
      - or no header fallback.
    Also strips tz info and normalizes to midnight.
    """
    # first try expected case
    try:
        df = pd.read_csv(path, parse_dates=['date'])
        df = df.set_index('date').sort_index()
    except Exception:
        df = pd.read_csv(path, header=0)
        cols = list(df.columns)
        cand_names = ['date', 'Date', 'index', 'Index']
        found = False
        for name in cand_names:
            if name in df.columns:
                try:
                    df[name] = pd.to_datetime(df[name])
                    df = df.set_index(name).sort_index()
                    found = True
                    break
                except Exception:
                    continue
        if not found:
            # try first column
            if cols:
                first_col = cols[0]
                try:
                    df[first_col] = pd.to_datetime(df[first_col])
                    df = df.set_index(first_col).sort_index()
                    found = True
                except Exception:
                    pass
        if not found:
            # fallback: try no-header parse:
            try:
                df2 = pd.read_csv(path, header=None)
                df2[0] = pd.to_datetime(df2[0])
                df2 = df2.set_index(0).sort_index()
                df2.columns = [f'col{i}' for i in range(1, len(df2.columns)+1)]
                df = df2
            except Exception:
                raise ValueError(f"Could not parse date index from CSV: {path}. Columns found: {cols}")

    # now ensure index is DatetimeIndex, remove tz info and normalize to date
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError(f"Index could not be converted to datetime for file {path}")
    # strip tz info if present, normalize to midnight (daily)
    try:
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
    except Exception:
        # some pandas versions raise if tz is None — ignore
        pass
    df.index = df.index.normalize()
    return df

def load_results(results_dir=RESULTS_DIR):
    results = {}
    if not os.path.isdir(results_dir):
        print(f"results directory not found: {results_dir}", file=sys.stderr)
        return results
    for f in sorted(glob.glob(os.path.join(results_dir, '*.csv'))):
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            df = _robust_read_csv(f)
            preview = df.head(3)
            print(f"[LOAD] {name} -> rows={len(df)} columns={list(df.columns)} preview:\n{preview}\n")
            results[name] = df
        except Exception as e:
            print(f"[WARN] Skipping {f} due to parse error: {e}", file=sys.stderr)
            continue
    return results

def pivot_mu(df_rating):
    # df_rating: columns date, strategy, mu
    pivot = df_rating.pivot_table(index='date', columns='strategy', values='mu')
    pivot = pivot.sort_index()
    # pivot index is already normalized in load_rating_history()
    return pivot

def build_long_table(pivot_mu_df, results_dict):
    rows = []
    missing_strats = []
    for strat in pivot_mu_df.columns:
        mu_ser = pivot_mu_df[strat].dropna()
        if mu_ser.empty:
            continue
        # mu_ser index are normalized dates (tz-stripped)
        if strat not in results_dict:
            missing_strats.append(strat)
            continue
        res = results_dict[strat]
        # res index normalized as well by loader
        # pick alpha_ret first, else ret, else first numeric column
        if 'alpha_ret' in res.columns:
            ret_ser = res['alpha_ret'].reindex(mu_ser.index)
        elif 'ret' in res.columns:
            ret_ser = res['ret'].reindex(mu_ser.index)
        else:
            numeric_cols = [c for c in res.columns if pd.api.types.is_numeric_dtype(res[c])]
            if numeric_cols:
                ret_ser = res[numeric_cols[0]].reindex(mu_ser.index)
            else:
                missing_strats.append(strat)
                continue

        # diagnostic: how many matched dates?
        matched = mu_ser.index.intersection(ret_ser.dropna().index)
        print(f"[ALIGN] {strat}: mu_dates={len(mu_ser)} ret_nonnull_dates={len(ret_ser.dropna())} matched={len(matched)}")

        for d, mu in mu_ser.items():
            r_val = ret_ser.loc[d] if d in ret_ser.index else np.nan
            try:
                r_float = float(r_val) if not pd.isna(r_val) else np.nan
            except Exception:
                r_float = np.nan
            rows.append({'date': d, 'strategy': strat, 'mu': float(mu), 'ret': r_float})

    if missing_strats:
        print(f"Warning: {len(missing_strats)} strategies missing from results CSVs or had no numeric column: {missing_strats}", file=sys.stderr)

    if not rows:
        return pd.DataFrame(columns=['date','strategy','mu','ret'])

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.sort_values(['strategy','date'])
    return df

def compute_momentum_and_forward(df_long, W=W, T=T):
    if df_long.empty:
        return pd.DataFrame(columns=['date','strategy','mu','elo_mom','fwd_ret'])
    df_long['date'] = pd.to_datetime(df_long['date']).dt.normalize()
    out_rows = []
    for strat, g in df_long.groupby('strategy'):
        g = g.set_index('date').sort_index()
        if len(g) < max(W,T)+5:
            print(f"[SKIP] {strat}: length {len(g)} too short for W={W},T={T}")
            continue
        g = g.copy()
        g['elo_mom'] = g['mu'].diff(W)
        g['fwd_ret'] = g['ret'].rolling(window=T).sum().shift(-T+1)
        g2 = g.dropna(subset=['elo_mom','fwd_ret']).reset_index()
        if not g2.empty:
            out_rows.append(g2[['date','strategy','mu','elo_mom','fwd_ret']])
    if not out_rows:
        return pd.DataFrame(columns=['date','strategy','mu','elo_mom','fwd_ret'])
    return pd.concat(out_rows, ignore_index=True)

def run_regression(df_final, T=T):
    X = sm.add_constant(df_final['elo_mom'])
    y = df_final['fwd_ret']
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': T})
    return model

def main():
    rating = load_rating_history()
    results = load_results()
    if not results:
        print("No results CSVs found in results_csvs/ — create per-strategy files with columns date,ret and optional alpha_ret", file=sys.stderr)
    pivot = pivot_mu(rating)
    pivot.to_csv('debug_pivot_mu_snapshot.csv')
    df_long = build_long_table(pivot, results)
    df_long.to_csv('debug_long_table.csv', index=False)
    df_final = compute_momentum_and_forward(df_long)
    if df_final.empty:
        print("No momentum-forward pairs computed. Check debug CSVs (debug_pivot_mu_snapshot.csv, debug_long_table.csv) for what's missing.", file=sys.stderr)
        return
    model = run_regression(df_final)
    print(model.summary())
    df_final.to_csv('elo_momentum_table.csv', index=False)
    print("Saved elo_momentum_table.csv")

if __name__ == '__main__':
    main()
