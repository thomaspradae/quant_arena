# datamanager_v2.py
from __future__ import annotations
import os, json, sqlite3, time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from contextlib import contextmanager
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# --------- provider adapters ----------
class Provider:
    name: str
    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str, asset_type: str) -> pd.DataFrame:
        raise NotImplementedError

class YahooProvider(Provider):
    name = "yahoo"
    def fetch(self, symbol, start, end, timeframe, asset_type):
        import yfinance as yf
        interval = {"1min":"1m","2min":"2m","5min":"5m","15min":"15m","30min":"30m","1h":"60m","1d":"1d","1wk":"1wk","1mo":"1mo"}[timeframe]
        df = yf.download(symbol, start=start, end=end + pd.Timedelta(days=1), interval=interval, auto_adjust=False, progress=False)
        if df.empty: raise ValueError(f"No data for {symbol}")
        df = df.rename(columns=str.lower).reset_index().rename(columns={"index":"timestamp"})
        # enforce schema
        df["timestamp_utc"] = pd.to_datetime(df["timestamp"], utc=True)
        cols = ["open","high","low","close","adj close","volume"]
        for c in cols:
            if c not in df.columns: raise ValueError(f"Missing {c}")
        out = pd.DataFrame({
            "timestamp_utc": df["timestamp_utc"],
            "symbol": symbol,
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "adj_close": df["adj close"].astype(float),
            "volume": df["volume"].astype("int64", errors="ignore"),
            "dividends": df.get("dividends", pd.Series([0.0]*len(df))).astype(float) if "dividends" in df else 0.0,
            "splits": df.get("stock splits", pd.Series([0.0]*len(df))).astype(float) if "stock splits" in df else 0.0,
            "source": self.name,
        }).sort_values("timestamp_utc")
        return out

# --------- file lock ----------
@contextmanager
def file_lock(path: Path, timeout_sec=30):
    lock = path.with_suffix(".lock")
    t0 = time.time()
    while lock.exists():
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"Lock timeout {path}")
        time.sleep(0.1)
    try:
        lock.touch(exist_ok=False)
        yield
    finally:
        try: lock.unlink()
        except FileNotFoundError: pass

# --------- DataManager ----------
class DataManager:
    def __init__(self, root="data_store", provider: Optional[Provider]=None):
        self.root = Path(root); (self.root / "data").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "manifest.sqlite")
        self._init_db()
        self.provider = provider or YahooProvider()

    def _init_db(self):
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS series (
            source TEXT, asset_type TEXT, timeframe TEXT, symbol TEXT,
            min_ts TEXT, max_ts TEXT, rows INTEGER, last_asof TEXT,
            PRIMARY KEY (source, asset_type, timeframe, symbol)
        )""")
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            tag TEXT PRIMARY KEY, created TEXT
        )""")
        self.db.commit()

    # partitioned path: data/source/asset/timeframe/symbol/year=YYYY/chunk.parquet
    def _path(self, source, asset_type, timeframe, symbol, year):
        return self.root / "data" / source / asset_type / timeframe / symbol / f"year={year}.parquet"

    def ensure(self, symbol: str, timeframe="1d", asset_type="equity",
               start="2010-01-01", end=None, asof=None, overlap_days=5) -> pd.DataFrame:
        """Incrementally fetch & persist up to `asof` (default: now), then return full range [start,end]."""
        source = self.provider.name
        end = end or pd.Timestamp.utcnow().date().isoformat()
        asof = asof or datetime.now(timezone.utc).isoformat()

        # find last cached max_ts
        row = self.db.execute("""SELECT max_ts FROM series WHERE source=? AND asset_type=? AND timeframe=? AND symbol=?""",
                              (source, asset_type, timeframe, symbol)).fetchone()
        last_max = pd.Timestamp.min.tz_localize("UTC") if not row or not row[0] else pd.Timestamp(row[0])

        fetch_start = max(pd.to_datetime(start, utc=True), last_max - pd.Timedelta(days=overlap_days))
        fetch_end = pd.to_datetime(end, utc=True)

        if fetch_start < fetch_end:
            new_df = self.provider.fetch(symbol, fetch_start, fetch_end, timeframe, asset_type)
            self._validate(new_df)
            # write partitioned
            with file_lock(self.root / "data" / "LOCK"):
                self._append_partitioned(new_df, source, asset_type, timeframe, symbol, asof)
                self._refresh_manifest(source, asset_type, timeframe, symbol, asof)

        return self.load_panel([symbol], timeframe, asset_type, start, end)

    def _append_partitioned(self, df, source, asset_type, timeframe, symbol, asof):
        df = df.assign(asof_utc=asof)
        for year, chunk in df.groupby(df["timestamp_utc"].dt.year):
            path = self._path(source, asset_type, timeframe, symbol, year)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                old = pq.read_table(path).to_pandas()
                merged = (pd.concat([old, chunk], ignore_index=True)
                          .drop_duplicates(subset=["timestamp_utc"], keep="last")
                          .sort_values("timestamp_utc"))
            else:
                merged = chunk.sort_values("timestamp_utc")
            table = pa.Table.from_pandas(merged)
            pq.write_table(table, path, compression="snappy")

    def _refresh_manifest(self, source, asset_type, timeframe, symbol, asof):
        # scan partitions
        sym_dir = self.root / "data" / source / asset_type / timeframe / symbol
        if not sym_dir.exists(): return
        parts = list(sym_dir.glob("year=*.parquet"))
        if not parts:
            return
        min_ts, max_ts, rows = None, None, 0
        for p in parts:
            t = pq.read_table(p, columns=["timestamp_utc"])
            s = pd.to_datetime(t.column(0).to_pandas(), utc=True)
            if len(s) == 0: continue
            rows += len(s)
            mn, mx = s.min(), s.max()
            min_ts = mn if (min_ts is None or mn < min_ts) else min_ts
            max_ts = mx if (max_ts is None or mx > max_ts) else max_ts
        self.db.execute("""
        INSERT INTO series (source,asset_type,timeframe,symbol,min_ts,max_ts,rows,last_asof)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(source,asset_type,timeframe,symbol) DO UPDATE SET
            min_ts=excluded.min_ts, max_ts=excluded.max_ts, rows=excluded.rows, last_asof=excluded.last_asof
        """, (source, asset_type, timeframe, symbol, min_ts.isoformat(), max_ts.isoformat(), rows, asof))
        self.db.commit()

    def load_panel(self, symbols: List[str], timeframe="1d", asset_type="equity",
                   start="2010-01-01", end=None) -> pd.DataFrame:
        """Return a tidy panel dataframe: [timestamp_utc, symbol, ...]."""
        end = end or pd.Timestamp.utcnow().date().isoformat()
        start_ts = pd.to_datetime(start, utc=True); end_ts = pd.to_datetime(end, utc=True)
        frames = []
        for sym in symbols:
            sym_dir = self.root / "data" / self.provider.name / asset_type / timeframe / sym
            if not sym_dir.exists(): continue
            for p in sorted(sym_dir.glob("year=*.parquet")):
                year = int(p.stem.split("=")[1])
                if year < start_ts.year - 1 or year > end_ts.year + 1:  # prune reads
                    continue
                tbl = pq.read_table(p)
                df = tbl.to_pandas()
                m = (df["timestamp_utc"] >= start_ts) & (df["timestamp_utc"] <= end_ts)
                frames.append(df.loc[m])
        if not frames: return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True).sort_values(["symbol","timestamp_utc"])
        # drop dupes post-filter
        out = out.drop_duplicates(subset=["symbol","timestamp_utc"], keep="last")
        return out

    def snapshot_asof(self, tag: str):
        """Create a time-travel tag; future reads can pin to this as-of."""
        ts = datetime.now(timezone.utc).isoformat()
        self.db.execute("INSERT OR REPLACE INTO snapshots(tag, created) VALUES (?,?)", (tag, ts))
        self.db.commit()

    # --- validations ---
    def _validate(self, df: pd.DataFrame):
        if df.empty: raise ValueError("empty frame")
        req = ["timestamp_utc","open","high","low","close","adj_close","volume"]
        if not all(c in df.columns for c in req): raise ValueError("missing required columns")
        if df["timestamp_utc"].isna().any(): raise ValueError("NaT timestamps")
        if (df[["open","high","low","close"]] < 0).any().any(): raise ValueError("negative prices")
        if (df["high"] < df["low"]).any(): raise ValueError("high < low")
        if not df["timestamp_utc"].is_monotonic_increasing:
            df.sort_values("timestamp_utc", inplace=True)
        if df.duplicated(subset=["timestamp_utc"]).any():
            # allow dedup in append; error if persists
            pass

    def coverage(self, source=None, asset_type=None, timeframe=None) -> pd.DataFrame:
        q = "SELECT * FROM series"
        df = pd.read_sql_query(q, self.db)
        if source: df = df[df.source==source]
        if asset_type: df = df[df.asset_type==asset_type]
        if timeframe: df = df[df.timeframe==timeframe]
        return df
