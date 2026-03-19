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
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_convert(None)
    df['date'] = df['date'].dt.normalize()
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
    # Strategy 1: Try reading with 'date' column directly
    try:
        df = pd.read_csv(path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
            df = df.dropna(subset=['date'])
            df['date'] = df['date'].dt.tz_convert(None).dt.normalize()
            df = df.set_index('date').sort_index()
            return df
    except Exception as e:
        print(f"  [DEBUG] Strategy 1 failed for {path}: {e}")
    
    # Strategy 2: Try reading with various date column names
    try:
        df = pd.read_csv(path)
        cand_names = ['date', 'Date', 'index', 'Index', 'timestamp', 'Timestamp']
        for name in cand_names:
            if name in df.columns:
                try:
                    df[name] = pd.to_datetime(df[name], utc=True, errors='coerce')
                    df = df.dropna(subset=[name])
                    df[name] = df[name].dt.tz_convert(None).dt.normalize()
                    df = df.set_index(name).sort_index()
                    df.index.name = 'date'
                    return df
                except Exception:
                    continue
    except Exception as e:
        print(f"  [DEBUG] Strategy 2 failed for {path}: {e}")
    
    # Strategy 3: Try first column as date
    try:
        df = pd.read_csv(path)
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], utc=True, errors='coerce')
        df = df.dropna(subset=[first_col])
        df[first_col] = df[first_col].dt.tz_convert(None).dt.normalize()
        df = df.set_index(first_col).sort_index()
        df.index.name = 'date'
        return df
    except Exception as e:
        print(f"  [DEBUG] Strategy 3 failed for {path}: {e}")
    
    # Strategy 4: Try reading without header
    try:
        df = pd.read_csv(path, header=None)
        df[0] = pd.to_datetime(df[0], utc=True, errors='coerce')
        df = df.dropna(subset=[0])
        df[0] = df[0].dt.tz_convert(None).dt.normalize()
        df = df.set_index(0).sort_index()
        df.index.name = 'date'
        df.columns = [f'col{i}' for i in range(1, len(df.columns)+1)]
        return df
    except Exception as e:
        print(f"  [DEBUG] Strategy 4 failed for {path}: {e}")
    
    raise ValueError(f"Could not parse date index from CSV: {path}")

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
            print(f"[LOAD] {name} -> rows={len(df)} columns={list(df.columns)}")
            print(f"       Index range: {df.index[0]} to {df.index[-1]}")
            print(f"       Preview:\n{preview}\n")
            results[name] = df
        except Exception as e:
            print(f"[WARN] Skipping {f} due to parse error: {e}", file=sys.stderr)
            continue
    return results

def pivot_mu(df_rating):
    # df_rating: columns date, strategy, mu
    pivot = df_rating.pivot_table(index='date', columns='strategy', values='mu')
    pivot = pivot.sort_index()
    return pivot

def build_long_table(pivot_mu_df, results_dict):
    rows = []
    missing_strats = []
    for strat in pivot_mu_df.columns:
        mu_ser = pivot_mu_df[strat].dropna()
        if mu_ser.empty:
            continue
        if strat not in results_dict:
            missing_strats.append(strat)
            continue
        res = results_dict[strat]
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
    print("=" * 80)
    print("MOMENTUM ANALYSIS")
    print("=" * 80)
    
    print("\n1. Loading rating history...")
    rating = load_rating_history()
    print(f"   Loaded {len(rating)} rating records")
    print(f"   Date range: {rating['date'].min()} to {rating['date'].max()}")
    print(f"   Strategies: {sorted(rating['strategy'].unique())}")
    
    print("\n2. Loading results CSVs...")
    results = load_results()
    if not results:
        print("No results CSVs found in results_csvs/", file=sys.stderr)
        print("Please run export_results_csvs.py first!", file=sys.stderr)
        return
    
    print("\n3. Pivoting rating data...")
    pivot = pivot_mu(rating)
    pivot.to_csv('debug_pivot_mu_snapshot.csv')
    print(f"   Pivot shape: {pivot.shape}")
    print(f"   Columns: {list(pivot.columns)}")
    
    print("\n4. Building long table (aligning ratings with returns)...")
    df_long = build_long_table(pivot, results)
    df_long.to_csv('debug_long_table.csv', index=False)
    print(f"   Long table rows: {len(df_long)}")
    
    if df_long.empty:
        print("\nERROR: Long table is empty!", file=sys.stderr)
        print("Check debug_long_table.csv and debug_pivot_mu_snapshot.csv", file=sys.stderr)
        return
    
    print("\n5. Computing momentum and forward returns...")
    df_final = compute_momentum_and_forward(df_long, W=W, T=T)
    
    if df_final.empty:
        print("\nERROR: No momentum-forward pairs computed!", file=sys.stderr)
        print("Check if date ranges align and series are long enough.", file=sys.stderr)
        print("Debug files saved: debug_pivot_mu_snapshot.csv, debug_long_table.csv", file=sys.stderr)
        return
    
    print(f"   Final analysis rows: {len(df_final)}")
    
    print("\n6. Running regression...")
    model = run_regression(df_final, T=T)
    print("\n" + "=" * 80)
    print("REGRESSION RESULTS")
    print("=" * 80)
    print(model.summary())
    
    print("\n7. Saving output...")
    df_final.to_csv('elo_momentum_table.csv', index=False)
    print("   Saved: elo_momentum_table.csv")
    print("   Saved: debug_pivot_mu_snapshot.csv")
    print("   Saved: debug_long_table.csv")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()