"""
Streamlit Options Chain Visualization App

This module provides the web interface for visualizing options data
fetched through the alpaca_data module.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import io

# Import our data module
try:
    from alpaca_data import AlpacaOptionsClient, OptionsDataError
except ImportError:
    st.error("Could not import alpaca_data module. Make sure alpaca_data.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Options Chain Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class OptionsVisualizer:
    """Class to handle all visualization functions"""
    
    @staticmethod
    def create_price_chart(df: pd.DataFrame) -> go.Figure:
        """Create option prices scatter plot"""
        fig = go.Figure()
        
        if df.empty or 'strike' not in df.columns or 'mid_price' not in df.columns:
            fig.add_annotation(
                text="No price data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        calls = df[df['type'] == 'call'] if 'call' in df['type'].values else pd.DataFrame()
        puts = df[df['type'] == 'put'] if 'put' in df['type'].values else pd.DataFrame()
        
        if not calls.empty:
            fig.add_trace(go.Scatter(
                x=calls['strike'],
                y=calls['mid_price'],
                mode='markers',
                name='Calls',
                marker=dict(color='green', size=8, opacity=0.7),
                hovertemplate='<b>Call</b><br>Strike: $%{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
        
        if not puts.empty:
            fig.add_trace(go.Scatter(
                x=puts['strike'],
                y=puts['mid_price'],
                mode='markers',
                name='Puts',
                marker=dict(color='red', size=8, opacity=0.7),
                hovertemplate='<b>Put</b><br>Strike: $%{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Option Prices by Strike',
            xaxis_title='Strike Price ($)',
            yaxis_title='Mid Price ($)',
            hovermode='closest',
            height=400,
            showlegend=True
        )
        
        return fig

    @staticmethod
    def create_iv_chart(df: pd.DataFrame) -> go.Figure:
        """Create implied volatility smile chart"""
        fig = go.Figure()
        
        if df.empty or 'strike' not in df.columns or 'iv' not in df.columns:
            fig.add_annotation(
                text="No implied volatility data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        calls = df[(df['type'] == 'call') & df['iv'].notna()] if 'call' in df['type'].values else pd.DataFrame()
        puts = df[(df['type'] == 'put') & df['iv'].notna()] if 'put' in df['type'].values else pd.DataFrame()
        
        if not calls.empty:
            fig.add_trace(go.Scatter(
                x=calls['strike'],
                y=calls['iv'] * 100,
                mode='lines+markers',
                name='Calls',
                line=dict(color='green', width=2),
                marker=dict(size=6),
                hovertemplate='<b>Call</b><br>Strike: $%{x}<br>IV: %{y:.1f}%<extra></extra>'
            ))
        
        if not puts.empty:
            fig.add_trace(go.Scatter(
                x=puts['strike'],
                y=puts['iv'] * 100,
                mode='lines+markers',
                name='Puts',
                line=dict(color='red', width=2),
                marker=dict(size=6),
                hovertemplate='<b>Put</b><br>Strike: $%{x}<br>IV: %{y:.1f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title='Implied Volatility Smile',
            xaxis_title='Strike Price ($)',
            yaxis_title='Implied Volatility (%)',
            hovermode='closest',
            height=400,
            showlegend=True
        )
        
        return fig

    @staticmethod
    def create_volume_chart(df: pd.DataFrame) -> go.Figure:
        """Create volume bar chart"""
        fig = go.Figure()
        
        if df.empty or 'total_volume' not in df.columns:
            fig.add_annotation(
                text="No volume data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        calls = df[df['type'] == 'call'] if 'call' in df['type'].values else pd.DataFrame()
        puts = df[df['type'] == 'put'] if 'put' in df['type'].values else pd.DataFrame()
        
        if not calls.empty:
            fig.add_trace(go.Bar(
                x=calls['strike'],
                y=calls['total_volume'],
                name='Calls',
                marker_color='green',
                opacity=0.7,
                hovertemplate='<b>Call</b><br>Strike: $%{x}<br>Volume: %{y}<extra></extra>'
            ))
        
        if not puts.empty:
            fig.add_trace(go.Bar(
                x=puts['strike'],
                y=puts['total_volume'],
                name='Puts',
                marker_color='red',
                opacity=0.7,
                hovertemplate='<b>Put</b><br>Strike: $%{x}<br>Volume: %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Total Volume by Strike',
            xaxis_title='Strike Price ($)',
            yaxis_title='Total Volume (Bid + Ask Size)',
            barmode='group',
            height=400,
            showlegend=True
        )
        
        return fig

    @staticmethod
    def create_greeks_chart(df: pd.DataFrame) -> go.Figure:
        """Create Greeks subplots chart"""
        greeks = ['delta', 'gamma', 'theta', 'vega']
        available_greeks = [g for g in greeks if g in df.columns and not df[g].isna().all()]
        
        if not available_greeks:
            fig = go.Figure()
            fig.add_annotation(
                text="No Greeks data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create subplots - adjust based on available greeks
        n_greeks = len(available_greeks)
        if n_greeks == 1:
            rows, cols = 1, 1
        elif n_greeks == 2:
            rows, cols = 1, 2
        elif n_greeks <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
            
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[g.capitalize() for g in available_greeks[:rows*cols]],
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
        )
        
        colors = {'call': 'green', 'put': 'red'}
        
        for i, greek in enumerate(available_greeks[:rows*cols]):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            for option_type in ['call', 'put']:
                data = df[(df['type'] == option_type) & df[greek].notna()]
                if not data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=data['strike'],
                            y=data[greek],
                            mode='lines+markers',
                            name=f'{option_type.title()}s',
                            line=dict(color=colors[option_type]),
                            showlegend=(i == 0),
                            hovertemplate=f'<b>{option_type.title()}</b><br>Strike: $%{{x}}<br>{greek.title()}: %{{y:.3f}}<extra></extra>'
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(height=600, title='Greeks by Strike Price')
        
        return fig

    @staticmethod
    def create_spread_chart(df: pd.DataFrame) -> go.Figure:
        """Create bid-ask spread chart"""
        fig = go.Figure()
        
        if df.empty or 'spread_pct' not in df.columns:
            fig.add_annotation(
                text="No spread data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        spread_data = df.dropna(subset=['spread_pct'])
        if spread_data.empty:
            fig.add_annotation(
                text="No valid spread data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        calls_spread = spread_data[spread_data['type'] == 'call'] if 'call' in spread_data['type'].values else pd.DataFrame()
        puts_spread = spread_data[spread_data['type'] == 'put'] if 'put' in spread_data['type'].values else pd.DataFrame()
        
        if not calls_spread.empty:
            fig.add_trace(go.Scatter(
                x=calls_spread['strike'],
                y=calls_spread['spread_pct'],
                mode='markers',
                name='Calls',
                marker=dict(color='green', size=6, opacity=0.7),
                hovertemplate='<b>Call</b><br>Strike: $%{x}<br>Spread: %{y:.2f}%<extra></extra>'
            ))
        
        if not puts_spread.empty:
            fig.add_trace(go.Scatter(
                x=puts_spread['strike'],
                y=puts_spread['spread_pct'],
                mode='markers',
                name='Puts',
                marker=dict(color='red', size=6, opacity=0.7),
                hovertemplate='<b>Put</b><br>Strike: $%{x}<br>Spread: %{y:.2f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title='Bid-Ask Spread by Strike',
            xaxis_title='Strike Price ($)',
            yaxis_title='Bid-Ask Spread (%)',
            hovermode='closest',
            height=400,
            showlegend=True
        )
        
        return fig

def safe_numeric_conversion(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Safely convert specified columns to numeric, handling errors gracefully
    
    Args:
        df: DataFrame to process
        columns: List of column names to convert
        
    Returns:
        DataFrame with converted columns
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean

def display_data_quality_metrics(metrics: dict):
    """Display data quality metrics in a nice format"""
    if not metrics:
        st.warning("No data quality metrics available")
        return
    
    st.subheader("📊 Data Quality Metrics")
    
    # Total contracts
    total = metrics.get('total_contracts', 0)
    st.metric("Total Contracts", total)
    
    # Parsing success rates
    if 'parsing_success' in metrics:
        st.markdown("**Parsing Success Rates:**")
        cols = st.columns(3)
        
        parsing_data = metrics['parsing_success']
        for i, (field, data) in enumerate(parsing_data.items()):
            with cols[i % 3]:
                count = data.get('count', 0)
                percentage = data.get('percentage', 0)
                st.metric(
                    f"{field.title()} Parsed",
                    f"{count}/{total}",
                    f"{percentage:.1f}%"
                )
    
    # Data availability
    if 'data_availability' in metrics:
        st.markdown("**Data Availability:**")
        availability_data = metrics['data_availability']
        
        # Create a DataFrame for better display
        avail_df = pd.DataFrame([
            {
                'Field': field.title(),
                'Available': data.get('count', 0),
                'Total': total,
                'Percentage': f"{data.get('percentage', 0):.1f}%"
            }
            for field, data in availability_data.items()
        ])
        
        if not avail_df.empty:
            st.dataframe(avail_df, use_container_width=True, hide_index=True)

def display_summary_statistics(df: pd.DataFrame):
    """Display summary statistics"""
    if df.empty:
        return
    
    st.subheader("📈 Summary Statistics")
    
    cols = st.columns(4)
    
    # Basic counts
    with cols[0]:
        if 'type' in df.columns:
            calls_count = (df['type'] == 'call').sum()
            st.metric("Calls", calls_count)
        else:
            st.metric("Calls", "N/A")
    
    with cols[1]:
        if 'type' in df.columns:
            puts_count = (df['type'] == 'put').sum()
            st.metric("Puts", puts_count)
        else:
            st.metric("Puts", "N/A")
    
    with cols[2]:
        if 'expiration' in df.columns:
            unique_exps = df['expiration'].nunique()
            st.metric("Expiration Dates", unique_exps)
        else:
            st.metric("Expiration Dates", "N/A")
    
    with cols[3]:
        if 'strike' in df.columns:
            strike_range = df['strike'].max() - df['strike'].min()
            st.metric("Strike Range", f"${strike_range:.2f}")
        else:
            st.metric("Strike Range", "N/A")
    
    # Additional statistics
    if any(col in df.columns for col in ['iv', 'delta', 'mid_price']):
        st.markdown("**Market Data Summary:**")
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            if 'iv' in df.columns and not df['iv'].isna().all():
                avg_iv = df['iv'].mean() * 100
                st.metric("Avg Implied Vol", f"{avg_iv:.1f}%")
            else:
                st.metric("Avg Implied Vol", "N/A")
        
        with summary_cols[1]:
            if 'mid_price' in df.columns and not df['mid_price'].isna().all():
                avg_price = df['mid_price'].mean()
                st.metric("Avg Option Price", f"${avg_price:.2f}")
            else:
                st.metric("Avg Option Price", "N/A")
        
        with summary_cols[2]:
            if 'total_volume' in df.columns and not df['total_volume'].isna().all():
                total_vol = df['total_volume'].sum()
                st.metric("Total Volume", f"{total_vol:,.0f}")
            else:
                st.metric("Total Volume", "N/A")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_options_data_cached(underlying, call_put, strike_min, strike_max, exp_gte, exp_lte, feed, api_key, api_secret):
    """Cached data fetching function"""
    try:
        client = AlpacaOptionsClient(api_key, api_secret)
        
        df = client.fetch_options_chain(
            underlying=underlying,
            call_put=call_put if call_put != "both" else None,
            strike_min=strike_min,
            strike_max=strike_max,
            exp_gte=exp_gte,
            exp_lte=exp_lte,
            feed=feed if feed != "default" else None,
            refresh_quotes=True
        )
        
        metrics = client.get_data_quality_metrics(df)
        
        return df, metrics, None
        
    except OptionsDataError as e:
        return pd.DataFrame(), {}, f"Options Data Error: {str(e)}"
    except Exception as e:
        return pd.DataFrame(), {}, f"Unexpected Error: {str(e)}"

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">📈 Options Chain Analyzer</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Configuration section
    with st.sidebar.expander("🔑 API Configuration", expanded=True):
        st.markdown("Enter your Alpaca API credentials:")
        api_key = st.text_input("API Key", type="password", help="Your Alpaca API key")
        api_secret = st.text_input("API Secret", type="password", help="Your Alpaca API secret")
        
        if not api_key or not api_secret:
            st.warning("⚠️ API credentials required")
    
    # Options Parameters section
    with st.sidebar.expander("📊 Options Parameters", expanded=True):
        underlying = st.text_input("Underlying Symbol", value="AAPL", help="Stock symbol (e.g., AAPL, TSLA, SPY)").upper()
        
        call_put = st.selectbox(
            "Option Type",
            ["both", "call", "put"],
            index=0,
            help="Filter by option type"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            strike_min = st.number_input("Min Strike", value=None, min_value=0.0, step=1.0, help="Minimum strike price")
        with col2:
            strike_max = st.number_input("Max Strike", value=None, min_value=0.0, step=1.0, help="Maximum strike price")
        
        # Date filters
        today = date.today()
        col1, col2 = st.columns(2)
        with col1:
            exp_gte = st.date_input("Min Expiration", value=today, help="Earliest expiration date")
        with col2:
            exp_lte = st.date_input("Max Expiration", value=today + timedelta(days=90), help="Latest expiration date")
        
        feed = st.selectbox(
            "Data Feed",
            ["default", "opra", "indicative"],
            index=0,
            help="Data feed preference (OPRA for real-time, Indicative for delayed)"
        )
    
    # Advanced Options section
    with st.sidebar.expander("🔧 Advanced Options", expanded=False):
        auto_refresh = st.checkbox("Auto-refresh data", value=False, help="Automatically refresh data every 5 minutes")
        show_debug = st.checkbox("Show debug info", value=False, help="Display technical debugging information")
    
    # Fetch data button
    fetch_button = st.sidebar.button("🔄 Fetch Options Data", type="primary", use_container_width=True)
    
    # Auto-fetch on startup if credentials are available
    if "data_fetched" not in st.session_state and api_key and api_secret:
        st.session_state.data_fetched = True
        fetch_button = True
    
    # Main content area
    if not api_key or not api_secret:
        st.markdown('<div class="info-box">👆 Please enter your Alpaca API credentials in the sidebar to get started.</div>', unsafe_allow_html=True)
        
        # Show some example content
        st.subheader("📖 About This App")
        st.markdown("""
        This Options Chain Analyzer helps you visualize and analyze options data from Alpaca Markets. 
        
        **Features:**
        - 📊 Interactive price and volatility charts
        - 📈 Greeks analysis (Delta, Gamma, Theta, Vega)
        - 📋 Comprehensive data tables with filtering
        - 🔍 Data quality metrics and parsing success rates
        - 💾 CSV export functionality
        
        **To get started:**
        1. Sign up for an Alpaca account at [alpaca.markets](https://alpaca.markets)
        2. Generate API keys from your dashboard
        3. Enter the credentials in the sidebar
        4. Configure your options parameters
        5. Click "Fetch Options Data"
        """)
        return
    
    # Data fetching logic
    if fetch_button or st.session_state.get("data_fetched", False):
        with st.spinner("🔄 Fetching options data from Alpaca..."):
            df, metrics, error = fetch_options_data_cached(
                underlying=underlying,
                call_put=call_put,
                strike_min=strike_min if strike_min else None,
                strike_max=strike_max if strike_max else None,
                exp_gte=exp_gte.strftime("%Y-%m-%d") if exp_gte else None,
                exp_lte=exp_lte.strftime("%Y-%m-%d") if exp_lte else None,
                feed=feed,
                api_key=api_key,
                api_secret=api_secret
            )
        
        if error:
            st.error(f"❌ {error}")
            if show_debug:
                st.exception(error)
            return
        
        if df.empty:
            st.warning("⚠️ No options data found for the specified criteria. Try adjusting your filters.")
            return
        
        # Success message
        st.markdown('<div class="success-box">✅ Data fetched successfully!</div>', unsafe_allow_html=True)
        
        # Display summary statistics
        display_summary_statistics(df)
        
        # Display data quality metrics
        if metrics:
            display_data_quality_metrics(metrics)
        
        # Visualization tabs
        st.subheader("📊 Interactive Visualizations")
        
        viz_tabs = st.tabs([
            "💰 Prices", 
            "📊 Volatility", 
            "📈 Volume", 
            "📉 Spreads",
            "🔢 Greeks", 
            "📋 Data Table"
        ])
        
        visualizer = OptionsVisualizer()
        
        with viz_tabs[0]:  # Prices
            with st.container():
                if 'strike' in df.columns and 'mid_price' in df.columns:
                    price_chart = visualizer.create_price_chart(df)
                    st.plotly_chart(price_chart, use_container_width=True)
                    
                    # Additional price analysis
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'type' in df.columns and 'mid_price' in df.columns:
                            price_summary = df.groupby('type')['mid_price'].agg(['mean', 'min', 'max']).round(2)
                            st.subheader("Price Summary by Type")
                            st.dataframe(price_summary, use_container_width=True)
                    
                    with col2:
                        if 'expiration' in df.columns and 'mid_price' in df.columns:
                            exp_summary = df.groupby('expiration')['mid_price'].agg(['count', 'mean']).round(2)
                            exp_summary.columns = ['Contract Count', 'Avg Price']
                            st.subheader("Price Summary by Expiration")
                            st.dataframe(exp_summary.head(), use_container_width=True)
                else:
                    st.warning("Price data not available for visualization.")
        
        with viz_tabs[1]:  # Volatility
            if 'strike' in df.columns and 'iv' in df.columns:
                iv_chart = visualizer.create_iv_chart(df)
                st.plotly_chart(iv_chart, use_container_width=True)
                
                # IV statistics
                col1, col2 = st.columns(2)
                with col1:
                    if 'iv' in df.columns:
                        # Ensure iv column is numeric
                        df_clean = df.copy()
                        df_clean['iv'] = pd.to_numeric(df_clean['iv'], errors='coerce')
                        
                        if not df_clean['iv'].isna().all():
                            iv_stats = df_clean['iv'].describe() * 100
                            iv_stats = iv_stats.round(2)
                            st.subheader("IV Statistics (%)")
                            st.dataframe(pd.DataFrame(iv_stats, columns=['Value']), use_container_width=True)
                        else:
                            st.subheader("IV Statistics (%)")
                            st.write("No valid IV data available")
                
                with col2:
                    if 'type' in df.columns and 'iv' in df.columns:
                        # Ensure iv column is numeric
                        df_clean = df.copy()
                        df_clean['iv'] = pd.to_numeric(df_clean['iv'], errors='coerce')
                        
                        if not df_clean['iv'].isna().all():
                            iv_by_type = df_clean.groupby('type')['iv'].mean() * 100
                            iv_by_type = iv_by_type.round(2)
                            st.subheader("Average IV by Type (%)")
                            st.dataframe(pd.DataFrame(iv_by_type, columns=['Avg IV']), use_container_width=True)
                        else:
                            st.subheader("Average IV by Type (%)")
                            st.write("No valid IV data available")
            else:
                st.warning("Implied volatility data not available for visualization.")
        
        with viz_tabs[2]:  # Volume
            if 'total_volume' in df.columns:
                volume_chart = visualizer.create_volume_chart(df)
                st.plotly_chart(volume_chart, use_container_width=True)
                
                # Top contracts by volume
                if not df['total_volume'].isna().all():
                    top_volume = df.nlargest(10, 'total_volume')[
                        ['contract', 'type', 'strike', 'expiration', 'total_volume']
                    ]
                    st.subheader("Top 10 Contracts by Volume")
                    st.dataframe(top_volume, use_container_width=True, hide_index=True)
            else:
                st.warning("Volume data not available for visualization.")
        
        with viz_tabs[3]:  # Spreads
            if 'spread_pct' in df.columns:
                spread_chart = visualizer.create_spread_chart(df)
                st.plotly_chart(spread_chart, use_container_width=True)
                
                # Spread analysis
                if 'spread_pct' in df.columns:
                    # Ensure spread_pct is numeric
                    df_clean = safe_numeric_conversion(df, ['spread_pct', 'total_volume'])
                    
                    if not df_clean['spread_pct'].isna().all():
                        col1, col2 = st.columns(2)
                        with col1:
                            spread_stats = df_clean['spread_pct'].describe().round(2)
                            st.subheader("Spread Statistics (%)")
                            st.dataframe(pd.DataFrame(spread_stats, columns=['Value']), use_container_width=True)
                        
                        with col2:
                            # Filter for tight spreads with valid data
                            valid_spread_data = df_clean.dropna(subset=['spread_pct', 'total_volume'])
                            if not valid_spread_data.empty:
                                tight_spreads = valid_spread_data[valid_spread_data['spread_pct'] < 5].nlargest(5, 'total_volume')[
                                    ['contract', 'strike', 'spread_pct', 'total_volume']
                                ]
                                st.subheader("Tightest Spreads (< 5%)")
                                if not tight_spreads.empty:
                                    st.dataframe(tight_spreads, use_container_width=True, hide_index=True)
                                else:
                                    st.write("No contracts with spreads < 5%")
                    else:
                        st.write("No valid spread data available")
            else:
                st.warning("Spread data not available for visualization.")
        
        with viz_tabs[4]:  # Greeks
            greeks_chart = visualizer.create_greeks_chart(df)
            if greeks_chart.data:
                st.plotly_chart(greeks_chart, use_container_width=True)
                
                # Greeks summary
                greeks_cols = ['delta', 'gamma', 'theta', 'vega']
                available_greeks = [col for col in greeks_cols if col in df.columns]
                
                if available_greeks:
                    # Safely convert Greeks columns to numeric
                    df_clean = safe_numeric_conversion(df, available_greeks)
                    
                    # Filter to only include columns with valid data
                    valid_greeks = [col for col in available_greeks if not df_clean[col].isna().all()]
                    
                    if valid_greeks:
                        greeks_summary = df_clean[valid_greeks].describe().round(4)
                        st.subheader("Greeks Summary Statistics")
                        st.dataframe(greeks_summary, use_container_width=True)
                    else:
                        st.subheader("Greeks Summary Statistics")
                        st.write("No valid Greeks data available")
            else:
                st.warning("Greeks data not available for visualization.")
        
        with viz_tabs[5]:  # Data Table
            st.subheader("📋 Complete Options Data")
            
            # Filter controls
            filter_cols = st.columns(4)
            
            with filter_cols[0]:
                if 'type' in df.columns:
                    type_options = ["All"] + sorted(df['type'].dropna().unique())
                    type_filter = st.selectbox("Filter by Type", type_options)
                else:
                    type_filter = "All"
            
            with filter_cols[1]:
                if 'expiration' in df.columns:
                    exp_options = ["All"] + sorted(df['expiration'].dropna().dt.strftime('%Y-%m-%d').unique())
                    exp_filter = st.selectbox("Filter by Expiration", exp_options)
                else:
                    exp_filter = "All"
            
            with filter_cols[2]:
                sort_options = ['strike', 'mid_price', 'iv', 'total_volume', 'spread_pct']
                available_sort = [opt for opt in sort_options if opt in df.columns]
                if available_sort:
                    sort_by = st.selectbox("Sort by", available_sort)
                else:
                    sort_by = None
            
            with filter_cols[3]:
                ascending = st.checkbox("Ascending", value=False)
            
            # Apply filters
            display_df = df.copy()
            
            if type_filter != "All" and 'type' in df.columns:
                display_df = display_df[display_df['type'] == type_filter]
            
            if exp_filter != "All" and 'expiration' in df.columns:
                display_df = display_df[display_df['expiration'].dt.strftime('%Y-%m-%d') == exp_filter]
            
            if sort_by and sort_by in display_df.columns:
                display_df = display_df.sort_values(sort_by, ascending=ascending)
            
            # Column selection
            all_cols = [
                'contract', 'type', 'strike', 'expiration', 'bid', 'ask', 'mid_price',
                'spread', 'spread_pct', 'bid_size', 'ask_size', 'total_volume', 
                'iv', 'delta', 'gamma', 'theta', 'vega', 'last_price', 'quote_ts'
            ]
            display_cols = [col for col in all_cols if col in display_df.columns]
            
            # Display the filtered data
            st.dataframe(
                display_df[display_cols],
                use_container_width=True,
                height=400
            )
            
            # Download functionality
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                # CSV download
                csv_buffer = io.StringIO()
                display_df[display_cols].to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"{underlying}_options_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Show row count
                st.metric("Displayed Rows", len(display_df))
        
        # Debug information
        if show_debug:
            with st.expander("🔧 Debug Information", expanded=False):
                st.subheader("DataFrame Info")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
                
                st.subheader("Sample Raw Data")
                st.json(df.head(3).to_dict())
                
                if metrics:
                    st.subheader("Metrics Object")
                    st.json(metrics)

if __name__ == "__main__":
    main()