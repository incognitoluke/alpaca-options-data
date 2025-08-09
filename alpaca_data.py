"""
Alpaca Options Data Module

This module handles all interactions with the Alpaca API for options data,
including data fetching, parsing, and preprocessing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, Dict, Tuple, List
import os
from dotenv import load_dotenv

# Alpaca imports
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionLatestQuoteRequest
from time import sleep
from alpaca.common.exceptions import APIError

class OptionsDataError(Exception):
    """Custom exception for options data errors"""
    pass

class AlpacaOptionsClient:
    """
    Client for fetching and processing options data from Alpaca
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialize the Alpaca options client
        
        Args:
            api_key: Alpaca API key (will try to load from env if not provided)
            api_secret: Alpaca API secret (will try to load from env if not provided)
        """
        # Load environment variables
        load_dotenv()
        
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise OptionsDataError("API key and secret are required")
        
        self.client = OptionHistoricalDataClient(self.api_key, self.api_secret)
        
    def parse_option_symbol(self, symbol: str) -> Tuple[Optional[date], Optional[float], Optional[str]]:
        """
        Parse option symbol to extract expiration, strike, and type.
        
        Standard format: ROOT{YYMMDD}{C/P}{STRIKE*1000}
        Example: AAPL240119C00150000 = AAPL, Jan 19 2024, Call, $150 strike
        
        Args:
            symbol: Option symbol string
            
        Returns:
            Tuple of (expiration_date, strike_price, option_type)
        """
        try:
            if len(symbol) < 15:
                return None, None, None
                
            # Find where the date part starts (6 digits after letters)
            root_end = 0
            for i, char in enumerate(symbol):
                if char.isdigit():
                    root_end = i
                    break
            
            if root_end == 0:
                return None, None, None
                
            remainder = symbol[root_end:]
            
            if len(remainder) < 15:
                return None, None, None
            
            # Extract date (YYMMDD)
            date_str = remainder[:6]
            year = 2000 + int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            expiration = datetime(year, month, day).date()
            
            # Extract call/put
            option_type = remainder[6].lower()
            if option_type not in ['c', 'p']:
                return None, None, None
            option_type = 'call' if option_type == 'c' else 'put'
            
            # Extract strike (8 digits, divide by 1000)
            strike_str = remainder[7:15]
            if not strike_str.isdigit():
                return None, None, None
            strike = int(strike_str) / 1000
            
            return expiration, strike, option_type
            
        except Exception as e:
            print(f"Error parsing symbol {symbol}: {e}")
            return None, None, None
    
    def extract_from_snapshot(self, symbol: str, snap) -> Dict:
        """
        Extract data from a single options snapshot, trying multiple attribute names
        
        Args:
            symbol: Option contract symbol
            snap: Options snapshot object from Alpaca
            
        Returns:
            Dictionary with extracted data
        """
        # Try to get quote, trade, and greeks data
        q = getattr(snap, "latest_quote", None)
        t = getattr(snap, "latest_trade", None) 
        g = getattr(snap, "greeks", None)

        # Try various attribute names for the core option data
        root = None
        exp = None
        strike = None
        ctype = None
        
        # Try different possible attribute names for root symbol
        for attr_name in ["root_symbol", "underlying_symbol", "underlying", "symbol"]:
            if hasattr(snap, attr_name):
                root = getattr(snap, attr_name)
                if root:
                    break
                    
        # Try different possible attribute names for expiration
        for attr_name in ["expiration_date", "expiration", "exp_date", "maturity"]:
            if hasattr(snap, attr_name):
                exp = getattr(snap, attr_name)
                if exp:
                    break
                    
        # Try different possible attribute names for strike
        for attr_name in ["strike_price", "strike", "strike_px"]:
            if hasattr(snap, attr_name):
                strike = getattr(snap, attr_name)
                if strike:
                    break
                    
        # Try different possible attribute names for option type
        for attr_name in ["contract_type", "option_type", "type", "side"]:
            if hasattr(snap, attr_name):
                ctype_obj = getattr(snap, attr_name)
                if ctype_obj:
                    if hasattr(ctype_obj, "value"):
                        ctype = ctype_obj.value
                    elif isinstance(ctype_obj, str):
                        ctype = ctype_obj
                    break
        
        # If we still don't have the core data, try parsing from symbol
        if not exp or not strike or not ctype:
            parsed_exp, parsed_strike, parsed_type = self.parse_option_symbol(symbol)
            if not exp:
                exp = parsed_exp
            if not strike:
                strike = parsed_strike  
            if not ctype:
                ctype = parsed_type
                
        # If root is still None, try to extract from symbol
        if not root:
            for i, char in enumerate(symbol):
                if char.isdigit():
                    root = symbol[:i]
                    break

        return {
            "contract": symbol,
            "root": root,
            "expiration": exp,
            "strike": strike,
            "type": ctype,
            "bid": getattr(q, "bid_price", None) if q else None,
            "bid_size": getattr(q, "bid_size", None) if q else None,
            "ask": getattr(q, "ask_price", None) if q else None,
            "ask_size": getattr(q, "ask_size", None) if q else None,
            "quote_ts": getattr(q, "timestamp", None) if q else None,
            "last_price": getattr(t, "price", None) if t else None,
            "last_size": getattr(t, "size", None) if t else None,
            "trade_ts": getattr(t, "timestamp", None) if t else None,
            "iv": getattr(g, "implied_volatility", None) if g else None,
            "delta": getattr(g, "delta", None) if g else None,
            "gamma": getattr(g, "gamma", None) if g else None,
            "theta": getattr(g, "theta", None) if g else None,
            "vega": getattr(g, "vega", None) if g else None,
        }
    
    def snapshots_to_dataframe(self, chain_snapshots: Dict) -> pd.DataFrame:
        """
        Convert chain snapshots dictionary to pandas DataFrame
        
        Args:
            chain_snapshots: Dictionary of {symbol: OptionsSnapshot}
            
        Returns:
            Processed pandas DataFrame
        """
        rows = []
        
        for symbol, snap in chain_snapshots.items():
            row_data = self.extract_from_snapshot(symbol, snap)
            rows.append(row_data)

        df = pd.DataFrame(rows)
        
        # Convert data types
        datetime_cols = ["expiration", "quote_ts", "trade_ts"]
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                
        if "strike" in df.columns:
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

        # Calculate additional metrics
        self._calculate_derived_metrics(df)

        # Sort the dataframe
        sort_cols = [c for c in ["expiration", "strike", "type"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="mergesort")
            
        return df
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> None:
        """
        Calculate derived metrics like mid price, spread, etc.
        
        Args:
            df: DataFrame to add metrics to (modified in place)
        """
        # Calculate mid price and spread
        if "bid" in df.columns and "ask" in df.columns:
            df["mid_price"] = (df["bid"] + df["ask"]) / 2
            df["spread"] = df["ask"] - df["bid"]
            df["spread_pct"] = (df["spread"] / df["mid_price"]) * 100
            
        # Calculate total volume (bid + ask size)
        if "bid_size" in df.columns and "ask_size" in df.columns:
            df["total_volume"] = df["bid_size"].fillna(0) + df["ask_size"].fillna(0)
            
        # Calculate moneyness if we have underlying price (would need to be passed in)
        # This could be enhanced to fetch current stock price
        
    def build_chain_request(
        self,
        underlying: str,
        call_put: Optional[str] = None,
        strike_min: Optional[float] = None,
        strike_max: Optional[float] = None,
        exp_gte: Optional[str] = None,
        exp_lte: Optional[str] = None,
        feed: Optional[str] = None,
    ) -> OptionChainRequest:
        """
        Build an OptionChainRequest object
        
        Args:
            underlying: Underlying symbol
            call_put: 'call', 'put', or None for both
            strike_min: Minimum strike price
            strike_max: Maximum strike price
            exp_gte: Minimum expiration date (YYYY-MM-DD)
            exp_lte: Maximum expiration date (YYYY-MM-DD)
            feed: Data feed ('opra', 'indicative', or None)
            
        Returns:
            OptionChainRequest object
        """
        type_value = None
        if call_put and call_put.lower() in ("call", "put"):
            type_value = call_put.lower()

        kwargs = dict(
            underlying_symbol=underlying.upper(),
            type=type_value,
            strike_price_gte=strike_min,
            strike_price_lte=strike_max,
            expiration_date_gte=exp_gte,
            expiration_date_lte=exp_lte,
        )
        
        if feed and feed.lower() in ("opra", "indicative"):
            kwargs["feed"] = feed.lower()

        return OptionChainRequest(**kwargs)
    
    def fetch_latest_quotes(self, symbols: List[str]) -> Dict:
        """
        Fetch latest quotes for a list of option symbols
        
        Args:
            symbols: List of option symbols
            
        Returns:
            Dictionary of latest quotes
        """
        if not symbols:
            return {}
            
        def chunked(seq, n):
            """Split sequence into chunks of size n"""
            for i in range(0, len(seq), n):
                yield seq[i:i+n]

        MAX_PER_REQ = 100  # Alpaca limit for /options/quotes/latest
        all_quotes = {}

        for chunk in chunked(symbols, MAX_PER_REQ):
            for attempt in range(3):  # Retry logic
                try:
                    quotes = self.client.get_option_latest_quote(
                        OptionLatestQuoteRequest(symbol_or_symbols=chunk)
                    )
                    all_quotes.update(quotes)
                    break
                except APIError as e:
                    # Handle rate limiting
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        sleep(0.8 * (attempt + 1))
                        continue
                    raise
                    
        return all_quotes
    
    def fetch_options_chain(
        self,
        underlying: str,
        call_put: Optional[str] = None,
        strike_min: Optional[float] = None,
        strike_max: Optional[float] = None,
        exp_gte: Optional[str] = None,
        exp_lte: Optional[str] = None,
        feed: Optional[str] = None,
        refresh_quotes: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch complete options chain data
        
        Args:
            underlying: Underlying symbol (e.g., 'AAPL')
            call_put: Filter by option type ('call', 'put', or None for both)
            strike_min: Minimum strike price filter
            strike_max: Maximum strike price filter
            exp_gte: Minimum expiration date (YYYY-MM-DD format)
            exp_lte: Maximum expiration date (YYYY-MM-DD format)
            feed: Data feed preference ('opra', 'indicative', or None)
            refresh_quotes: Whether to fetch latest quotes for better NBBO
            
        Returns:
            DataFrame with options chain data
            
        Raises:
            OptionsDataError: If API call fails or data is invalid
        """
        try:
            # Build the request
            chain_req = self.build_chain_request(
                underlying=underlying,
                call_put=call_put,
                strike_min=strike_min,
                strike_max=strike_max,
                exp_gte=exp_gte,
                exp_lte=exp_lte,
                feed=feed,
            )

            # Get chain snapshots
            chain = self.client.get_option_chain(chain_req)
            
            if not chain:
                raise OptionsDataError("No options data returned from API")
            
            # Convert to DataFrame
            df = self.snapshots_to_dataframe(chain)
            
            # Optionally refresh with latest quotes
            if refresh_quotes and not df.empty:
                symbols = df['contract'].tolist()
                try:
                    latest_quotes = self.fetch_latest_quotes(symbols)
                    df = self._merge_latest_quotes(df, latest_quotes)
                except Exception as e:
                    print(f"Warning: Could not fetch latest quotes: {e}")
            
            return df
            
        except APIError as e:
            raise OptionsDataError(f"Alpaca API error: {e}")
        except Exception as e:
            raise OptionsDataError(f"Error fetching options data: {e}")
    
    def _merge_latest_quotes(self, df: pd.DataFrame, latest_quotes: Dict) -> pd.DataFrame:
        """
        Merge latest quotes into the main DataFrame
        
        Args:
            df: Main options DataFrame
            latest_quotes: Dictionary of latest quotes
            
        Returns:
            DataFrame with updated quote data
        """
        if not latest_quotes:
            return df
            
        # Create DataFrame from latest quotes
        latest_rows = []
        for symbol, quote in latest_quotes.items():
            latest_rows.append({
                "contract": symbol,
                "bid": getattr(quote, "bid_price", None),
                "bid_size": getattr(quote, "bid_size", None),
                "ask": getattr(quote, "ask_price", None),
                "ask_size": getattr(quote, "ask_size", None),
                "quote_ts": getattr(quote, "timestamp", None),
            })

        if not latest_rows:
            return df
            
        latest_df = pd.DataFrame(latest_rows)
        latest_df["quote_ts"] = pd.to_datetime(latest_df["quote_ts"], errors="coerce")
        
        # Drop old quote columns and merge new ones
        quote_cols = ["bid", "bid_size", "ask", "ask_size", "quote_ts"]
        df = df.drop(columns=[c for c in quote_cols if c in df.columns])
        df = df.merge(latest_df, on="contract", how="left")
        
        # Recalculate derived metrics with new quote data
        self._calculate_derived_metrics(df)
        
        return df
    
    def get_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate data quality metrics for the options DataFrame
        
        Args:
            df: Options DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        if df.empty:
            return {}
            
        total_contracts = len(df)
        
        metrics = {
            "total_contracts": total_contracts,
            "parsing_success": {},
            "data_availability": {},
            "summary_stats": {}
        }
        
        # Parsing success rates
        key_fields = ["strike", "expiration", "type"]
        for field in key_fields:
            if field in df.columns:
                success_count = df[field].notna().sum()
                metrics["parsing_success"][field] = {
                    "count": success_count,
                    "percentage": (success_count / total_contracts) * 100
                }
        
        # Data availability
        data_fields = ["bid", "ask", "iv", "delta", "gamma", "theta", "vega"]
        for field in data_fields:
            if field in df.columns:
                available_count = df[field].notna().sum()
                metrics["data_availability"][field] = {
                    "count": available_count,
                    "percentage": (available_count / total_contracts) * 100
                }
        
        # Summary statistics
        if "type" in df.columns:
            type_counts = df["type"].value_counts().to_dict()
            metrics["summary_stats"]["type_distribution"] = type_counts
            
        if "expiration" in df.columns:
            unique_expirations = df["expiration"].nunique()
            metrics["summary_stats"]["unique_expirations"] = unique_expirations
            
        if "strike" in df.columns:
            strike_stats = df["strike"].agg(["min", "max", "count"]).to_dict()
            metrics["summary_stats"]["strike_range"] = strike_stats
        
        return metrics

# Convenience functions for easy import
def create_client(api_key: str = None, api_secret: str = None) -> AlpacaOptionsClient:
    """Create an AlpacaOptionsClient instance"""
    return AlpacaOptionsClient(api_key, api_secret)

def fetch_options_data(
    underlying: str,
    api_key: str = None,
    api_secret: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to fetch options data
    
    Args:
        underlying: Underlying symbol
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        **kwargs: Additional arguments passed to fetch_options_chain
        
    Returns:
        DataFrame with options data
    """
    client = create_client(api_key, api_secret)
    return client.fetch_options_chain(underlying, **kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    try:
        client = create_client()
        df = client.fetch_options_chain("AAPL")
        
        print(f"Fetched {len(df)} options contracts")
        print("\nData quality metrics:")
        metrics = client.get_data_quality_metrics(df)
        
        for category, data in metrics.items():
            if isinstance(data, dict):
                print(f"\n{category.upper()}:")
                for key, value in data.items():
                    print(f"  {key}: {value}")
            else:
                print(f"{category}: {data}")
                
        print("\nSample data:")
        cols = ["contract", "type", "strike", "expiration", "bid", "ask", "iv"]
        available_cols = [c for c in cols if c in df.columns]
        print(df[available_cols].head())
        
    except Exception as e:
        print(f"Error: {e}")