"""
Data Manager v1: Handles data fetching, caching, and retrieval
Supports Yahoo Finance and Alpaca with Parquet caching
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Data source imports
import yfinance as yf

# Optional: Alpaca (install with: pip install alpaca-trade-api)
try:
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    """
    Centralized data fetching, caching, and retrieval system.
    
    Features:
    - Multi-source support (Yahoo Finance, Alpaca)
    - Automatic local caching (Parquet format)
    - Metadata tracking (JSON)
    - Incremental updates
    - Data quality validation
    """
    
    def __init__(self, cache_dir: str = './data_cache', alpaca_api_key: Optional[str] = None, 
                 alpaca_secret_key: Optional[str] = None):
        """
        Initialize DataManager.
        
        Args:
            cache_dir: Directory for storing cached data
            alpaca_api_key: Alpaca API key (optional)
            alpaca_secret_key: Alpaca secret key (optional)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.cache_dir / 'data'
        self.metadata_dir = self.cache_dir / 'metadata'
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Load metadata database
        self.metadata_file = self.metadata_dir / 'metadata.json'
        self.metadata_db = self._load_metadata()
        
        # Initialize API clients
        self.alpaca_stock_client = None
        self.alpaca_crypto_client = None
        if ALPACA_AVAILABLE and alpaca_api_key and alpaca_secret_key:
            self.alpaca_stock_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)
            self.alpaca_crypto_client = CryptoHistoricalDataClient(alpaca_api_key, alpaca_secret_key)
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def fetch_data(
        self,
        symbol: str,
        source: str = 'yahoo',
        timeframe: str = '1d',
        start_date: str = '2020-01-01',
        end_date: Optional[str] = None,
        asset_type: str = 'equity',
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch data from source or cache.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'BTC-USD')
            source: Data source ('yahoo', 'alpaca')
            timeframe: Bar size ('1min', '5min', '15min', '1h', '1d')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            asset_type: Asset class ('equity', 'crypto', 'futures')
            force_refresh: Force re-download even if cached
        
        Returns:
            DataFrame with OHLCV data (datetime index)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Generate cache key
        cache_key = self._generate_cache_key(symbol, source, timeframe, start_date, end_date)
        
        # Check cache
        if not force_refresh and self._is_cached(cache_key):
            print(f"Loading {symbol} from cache...")
            data = self._load_from_cache(cache_key)
            
            # Validate cached data
            if self._validate_data(data):
                return data
            else:
                print(f"Cached data invalid, re-fetching...")
        
        # Fetch from source
        print(f"Fetching {symbol} from {source}...")
        data = self._fetch_from_source(symbol, source, timeframe, start_date, end_date, asset_type)
        
        # Validate fetched data
        if not self._validate_data(data):
            raise ValueError(f"Invalid data received for {symbol}")
        
        # Cache it
        self._save_to_cache(cache_key, data, metadata={
            'symbol': symbol,
            'source': source,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'asset_type': asset_type,
            'fetch_timestamp': datetime.now().isoformat(),
            'num_rows': len(data)
        })
        
        return data
    
    def get_available_symbols(self, source: Optional[str] = None, 
                            asset_type: Optional[str] = None) -> List[str]:
        """Get list of cached symbols, optionally filtered."""
        symbols = set()
        for key, meta in self.metadata_db.items():
            if source and meta['source'] != source:
                continue
            if asset_type and meta['asset_type'] != asset_type:
                continue
            symbols.add(meta['symbol'])
        return sorted(list(symbols))
    
    def get_date_range(self, symbol: str, source: str = 'yahoo', 
                      timeframe: str = '1d') -> Optional[Tuple[str, str]]:
        """Get available date range for a symbol from cache."""
        for key, meta in self.metadata_db.items():
            if (meta['symbol'] == symbol and 
                meta['source'] == source and 
                meta['timeframe'] == timeframe):
                return (meta['start_date'], meta['end_date'])
        return None
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cache (all or specific symbol)."""
        if symbol is None:
            # Clear all
            for f in self.data_dir.glob('*.parquet'):
                f.unlink()
            self.metadata_db = {}
            self._save_metadata()
            print("All cache cleared")
        else:
            # Clear specific symbol
            keys_to_remove = [k for k, v in self.metadata_db.items() if v['symbol'] == symbol]
            for key in keys_to_remove:
                cache_file = self.data_dir / f"{key}.parquet"
                if cache_file.exists():
                    cache_file.unlink()
                del self.metadata_db[key]
            self._save_metadata()
            print(f"Cache cleared for {symbol}")
    
    # ========================================================================
    # CACHING
    # ========================================================================
    
    def _generate_cache_key(self, symbol: str, source: str, timeframe: str, 
                           start_date: str, end_date: str) -> str:
        """Generate unique cache key (hash to avoid filesystem issues)."""
        key_string = f"{symbol}_{source}_{timeframe}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached."""
        cache_file = self.data_dir / f"{cache_key}.parquet"
        return cache_file.exists() and cache_key in self.metadata_db
    
    def _load_from_cache(self, cache_key: str) -> pd.DataFrame:
        """Load data from Parquet cache."""
        cache_file = self.data_dir / f"{cache_key}.parquet"
        df = pd.read_parquet(cache_file)
        df.index = pd.to_datetime(df.index)
        return df
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame, metadata: Dict):
        """Save data to Parquet cache with metadata."""
        # Save data
        cache_file = self.data_dir / f"{cache_key}.parquet"
        data.to_parquet(cache_file, compression='snappy')
        
        # Update metadata
        self.metadata_db[cache_key] = metadata
        self._save_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load metadata database from JSON."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata database to JSON."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata_db, f, indent=2)
    
    # ========================================================================
    # DATA FETCHING
    # ========================================================================
    
    def _fetch_from_source(self, symbol: str, source: str, timeframe: str,
                          start_date: str, end_date: str, asset_type: str) -> pd.DataFrame:
        """Dispatch to appropriate data source."""
        if source == 'yahoo':
            return self._fetch_yahoo(symbol, timeframe, start_date, end_date)
        elif source == 'alpaca':
            return self._fetch_alpaca(symbol, timeframe, start_date, end_date, asset_type)
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def _fetch_yahoo(self, symbol: str, timeframe: str, start_date: str, 
                    end_date: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        # Map timeframe to yfinance interval
        interval_map = {
            '1min': '1m',
            '2min': '2m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m',
            '1h': '1h',
            '1d': '1d',
            '1wk': '1wk',
            '1mo': '1mo'
        }
        interval = interval_map.get(timeframe, '1d')
        
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Standardize column names (lowercase)
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing required columns for {symbol}")
        
        return df
    
    def _fetch_alpaca(self, symbol: str, timeframe: str, start_date: str,
                     end_date: str, asset_type: str) -> pd.DataFrame:
        """Fetch data from Alpaca."""
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca not installed. Install with: pip install alpaca-trade-api")
        
        if not self.alpaca_stock_client:
            raise ValueError("Alpaca credentials not provided")
        
        # Map timeframe to Alpaca TimeFrame
        tf_map = {
            '1min': TimeFrame(1, TimeFrameUnit.Minute),
            '5min': TimeFrame(5, TimeFrameUnit.Minute),
            '15min': TimeFrame(15, TimeFrameUnit.Minute),
            '1h': TimeFrame(1, TimeFrameUnit.Hour),
            '1d': TimeFrame(1, TimeFrameUnit.Day),
        }
        alpaca_tf = tf_map.get(timeframe)
        if not alpaca_tf:
            raise ValueError(f"Unsupported timeframe for Alpaca: {timeframe}")
        
        # Fetch based on asset type
        if asset_type == 'equity':
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_tf,
                start=start_date,
                end=end_date
            )
            bars = self.alpaca_stock_client.get_stock_bars(request)
        elif asset_type == 'crypto':
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_tf,
                start=start_date,
                end=end_date
            )
            bars = self.alpaca_crypto_client.get_crypto_bars(request)
        else:
            raise ValueError(f"Unsupported asset_type for Alpaca: {asset_type}")
        
        # Convert to DataFrame
        df = bars.df
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Reset index if multi-index
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        return df
    
    # ========================================================================
    # DATA VALIDATION
    # ========================================================================
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality."""
        if data.empty:
            return False
        
        # Check for required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required):
            return False
        
        # Check for NaN in critical columns
        critical_cols = ['close']
        if data[critical_cols].isnull().any().any():
            print("Warning: NaN values in critical columns")
            return False
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        if (data[price_cols] < 0).any().any():
            print("Warning: Negative prices detected")
            return False
        
        # Check high >= low
        if (data['high'] < data['low']).any():
            print("Warning: High < Low detected")
            return False
        
        # Check for duplicated timestamps
        if data.index.duplicated().any():
            print("Warning: Duplicated timestamps detected")
            return False
        
        return True
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        total_files = len(list(self.data_dir.glob('*.parquet')))
        total_size = sum(f.stat().st_size for f in self.data_dir.glob('*.parquet'))
        
        return {
            'total_symbols': len(set(m['symbol'] for m in self.metadata_db.values())),
            'total_cached_series': total_files,
            'cache_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir)
        }
    
    def __repr__(self):
        stats = self.get_cache_stats()
        return (f"DataManager(symbols={stats['total_symbols']}, "
                f"cached={stats['total_cached_series']}, "
                f"size={stats['cache_size_mb']:.2f}MB)")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize DataManager
    dm = DataManager(cache_dir='./data_cache')
    
    print("="*60)
    print("Data Manager v1 - Example Usage")
    print("="*60)
    
    # Fetch some data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in symbols:
        try:
            df = dm.fetch_data(
                symbol=symbol,
                source='yahoo',
                timeframe='1d',
                start_date='2023-01-01',
                end_date='2024-01-01',
                asset_type='equity'
            )
            print(f"\n{symbol}: {len(df)} bars")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            print(f"Columns: {list(df.columns)}")
        except Exception as e:
            print(f"\nError fetching {symbol}: {e}")
    
    # Show cache stats
    print("\n" + "="*60)
    print("Cache Statistics:")
    print("="*60)
    print(dm)
    stats = dm.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show available symbols
    print("\n" + "="*60)
    print("Available Symbols:")
    print("="*60)
    available = dm.get_available_symbols()
    print(f"  {available}")
    
    # Test cache hit
    print("\n" + "="*60)
    print("Testing Cache Hit:")
    print("="*60)
    df = dm.fetch_data(
        symbol='AAPL',
        source='yahoo',
        timeframe='1d',
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    print(f"AAPL loaded from cache: {len(df)} bars")