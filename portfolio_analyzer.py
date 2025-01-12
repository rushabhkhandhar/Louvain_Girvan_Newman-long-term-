import pandas as pd
import numpy as np
import yfinance as yf
import networkx as nx
from community import community_louvain
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import requests
import os
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
import arch  # for GARCH volatility modeling
import logging
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# from dotenv import load_dotenv
# load_dotenv()


class EnhancedIndianMarketAnalyzer:
    def __init__(self, start_date: str, end_date: str, sector_mapping: dict):
        """
        Initialize the enhanced market analyzer with date range and sector mappings.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            sector_mapping (dict): Dictionary mapping stock symbols to their sectors
        """
        self.start_date = start_date
        self.end_date = end_date
        self.sector_mapping = sector_mapping
        self.stock_data = None
        self.returns_data = None
        self.correlation_matrix = None
        self.communities_louvain = None
        self.G = None
        self.alert_thresholds = {
            'drawdown': -0.10,  # Alert if 10% drawdown
            'volatility': 0.25,  # Alert if annualized volatility exceeds 25%
            'return': -0.05     # Alert if monthly return below -5%
        }
        self.forecasting_models = {
            'lstm': None,
            'prophet': None,
            'arima': None
        }
        self.stress_scenarios = {
            'bull_market': 1.5,
            'bear_market': 0.5,
            'high_volatility': 2.0
        }
        self.reoptimization_frequency = 'quarterly'  # Can be monthly, quarterly, annually


    def fetch_data(self, symbols):
        """
        Fetch historical data for given symbols using Yahoo Finance with retries and error handling.
        """
        data = pd.DataFrame()
        max_retries = 3
        retry_delay = 5  # seconds
        
        for symbol in symbols:
            for attempt in range(max_retries):
                try:
                    # Add a small delay between requests to avoid rate limiting
                    time.sleep(1)
                    
                    # Fetch data with explicit interval and premarket data
                    df = yf.download(
                        symbol, 
                        start=self.start_date, 
                        end=self.end_date, 
                        progress=False,
                        interval='1d',
                        prepost=True,
                        timeout=30
                    )
                    
                    if df.empty:
                        print(f"Warning: No data received for {symbol}")
                        continue
                        
                    if 'Adj Close' not in df.columns:
                        print(f"Warning: No Adj Close column for {symbol}")
                        if 'Close' in df.columns:
                            data[symbol] = df['Close']
                        continue
                        
                    data[symbol] = df['Adj Close']
                    break  # Success - exit retry loop
                    
                except Exception as e:
                    print(f"Attempt {attempt + 1}/{max_retries} failed for {symbol}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        print(f"Failed to fetch data for {symbol} after {max_retries} attempts")
        
        # Check if we got any data
        if data.empty:
            raise ValueError("No data could be fetched for any symbols")
            
        # Remove any columns (symbols) with all NaN values
        data = data.dropna(axis=1, how='all')
        
        # Forward fill missing values up to 5 days
        data = data.fillna(method='ffill', limit=5)
        
        # Remove any remaining rows with NaN values
        data = data.dropna(how='any')
        
        if data.empty:
            raise ValueError("No valid data remains after cleaning")
            
        self.stock_data = data
        return data
    
    def calculate_returns(self):
        """
        Calculate daily returns from the fetched price data.
        """
        if self.stock_data is None or self.stock_data.empty:
            raise ValueError("No stock data found. Did you call fetch_data()?")

        returns = self.stock_data.pct_change().dropna(how='all')
        returns.dropna(axis=0, how='any', inplace=True)
        self.returns_data = returns
        return self.returns_data

    def build_correlation_matrix(self, threshold: float = 0.3):
        """
        Build a thresholded correlation matrix from the returns data.
        """
        if self.returns_data is None:
            raise ValueError("No returns data available. Call calculate_returns() first.")

        corr = self.returns_data.corr()
        self.correlation_matrix = np.where(np.abs(corr) < threshold, 0, corr)
        return self.correlation_matrix
    
    def analyze_communities(self, partition):
        """
        Analyze each community's average annual return, volatility, and Sharpe ratio.
        Now includes sector diversity analysis.
        """
        if self.returns_data is None:
            raise ValueError("No returns data to analyze.")
        if not partition:
            raise ValueError("Partition is empty or invalid.")

        community_analysis = {}
        for community_id in set(partition.values()):
            # Get stocks in this community
            community_stocks = [
                self.returns_data.columns[node]
                for node, comm in partition.items()
                if comm == community_id
            ]
            
            # Skip if community is empty
            if not community_stocks:
                continue
                
            data_subset = self.returns_data[community_stocks]
            
            # Calculate key metrics
            avg_return = data_subset.mean().mean() * 252
            avg_vol = data_subset.std().mean() * np.sqrt(252)
            sharpe = (avg_return - 0.05) / avg_vol if avg_vol != 0 else np.nan
            
            # Analyze sector diversity
            sectors = [self.sector_mapping.get(stock, 'Unknown') for stock in community_stocks]
            sector_counts = pd.Series(sectors).value_counts()
            sector_concentration = sector_counts.max() / len(sectors) if sectors else 1.0
            
            community_analysis[community_id] = {
                'stocks': community_stocks,
                'size': len(community_stocks),
                'avg_return': avg_return,
                'volatility': avg_vol,
                'sharpe': sharpe,
                'sectors': dict(sector_counts),
                'sector_concentration': sector_concentration,
                'unique_sectors': len(sector_counts)
            }
            
        return community_analysis

    def select_best_community(self, community_analysis, criterion='combined'):
        """
        Select the best community based on multiple criteria.
        
        Args:
            community_analysis (dict): Output from analyze_communities()
            criterion (str): 'sharpe', 'return', 'diversity', or 'combined'
        """
        if criterion == 'combined':
            # Score communities on multiple factors
            scores = {}
            for comm_id, metrics in community_analysis.items():
                # Normalize metrics
                sharpe_score = metrics['sharpe'] if not np.isnan(metrics['sharpe']) else -np.inf
                return_score = metrics['avg_return']
                diversity_score = 1 - metrics['sector_concentration']  # Higher is better
                
                # Combined score with weights
                scores[comm_id] = (
                    0.4 * sharpe_score + 
                    0.3 * return_score + 
                    0.3 * diversity_score
                )
            
            return max(scores.items(), key=lambda x: x[1])[0]
        
        else:
            # Single criterion selection
            best_comm = None
            best_val = float('-inf')
            
            for comm_id, metrics in community_analysis.items():
                if criterion == 'sharpe':
                    metric_val = metrics['sharpe']
                elif criterion == 'return':
                    metric_val = metrics['avg_return']
                elif criterion == 'diversity':
                    metric_val = 1 - metrics['sector_concentration']
                else:
                    raise ValueError(f"Unknown criterion: {criterion}")
                
                if metric_val is not None and metric_val > best_val:
                    best_val = metric_val
                    best_comm = comm_id
                    
            return best_comm

    def create_network_graph(self):
        """
        Create a NetworkX graph from the thresholded correlation matrix.
        """
        if self.correlation_matrix is None:
            raise ValueError("No correlation matrix found. Build it first.")

        abs_matrix = np.abs(self.correlation_matrix)
        self.G = nx.from_numpy_array(abs_matrix)
        
        if self.returns_data is not None:
            symbols = self.returns_data.columns.tolist()
            nx.set_node_attributes(self.G, dict(enumerate(symbols)), 'symbol')

    def detect_communities_louvain(self):
        """
        Detect communities using the Louvain algorithm on the network graph.
        """
        if self.G is None:
            raise ValueError("Network graph not built. Call create_network_graph() first.")

        self.communities_louvain = community_louvain.best_partition(self.G)
        return self.communities_louvain

    def get_sector_exposure(self, weights):
        """
        Calculate sector exposure for given portfolio weights.
        """
        sector_exposure = {}
        for symbol, weight in weights.items():
            sector = self.sector_mapping.get(symbol, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        return sector_exposure

    def optimize_portfolio(self, target_return=0.15, risk_free_rate=0.05, max_sector_exposure=0.30, 
                         max_stock_weight=0.15):
        """
        Optimize portfolio with sector constraints and individual stock limits.
        """
        if self.returns_data is None or self.returns_data.empty:
            raise ValueError("No returns data for optimization.")

        def portfolio_stats(weights):
            ret_per_year = np.sum(self.returns_data.mean() * weights) * 252
            cov_matrix = self.returns_data.cov() * 252
            vol_per_year = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (ret_per_year - risk_free_rate) / (vol_per_year if vol_per_year != 0 else 1e-9)
            return ret_per_year, vol_per_year, sharpe

        def objective(w):
            return -portfolio_stats(w)[2]

        n_assets = len(self.returns_data.columns)
        bounds = tuple((0, max_stock_weight) for _ in range(n_assets))

        # Basic constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda x: portfolio_stats(x)[0] - target_return}  # Minimum return
        ]

        # Add sector constraints
        sectors = set(self.sector_mapping.values())
        for sector in sectors:
            sector_indices = [i for i, symbol in enumerate(self.returns_data.columns) 
                            if self.sector_mapping.get(symbol, 'Unknown') == sector]
            
            if sector_indices:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, indices=sector_indices: max_sector_exposure - sum(x[i] for i in indices)
                })

        init_guess = np.array([1.0 / n_assets] * n_assets)
        result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            print("Warning: Optimization did not converge. Using fallback allocation strategy.")
            # Implement a fallback strategy here if needed
            return None

        weights_opt = pd.Series(result.x, index=self.returns_data.columns)
        ret_opt, vol_opt, sharpe_opt = portfolio_stats(result.x)
        
        # Calculate sector exposure
        sector_exposure = self.get_sector_exposure(weights_opt)

        return {
            'weights': weights_opt,
            'expected_return': ret_opt,
            'volatility': vol_opt,
            'sharpe_ratio': sharpe_opt,
            'sector_exposure': sector_exposure
        }

    def check_alerts(self, portfolio_data):
        """
        Check for alert conditions in portfolio performance.
        Returns list of alert messages if any thresholds are breached.
        """
        if portfolio_data.empty:
            return []

        alerts = []
        
        # Check drawdown
        if len(portfolio_data['Drawdown']) > 0:
            if portfolio_data['Drawdown'].iloc[-1] <= self.alert_thresholds['drawdown']:
                alerts.append(f"ALERT: Portfolio drawdown ({portfolio_data['Drawdown'].iloc[-1]:.1%}) "
                            f"has breached threshold of {self.alert_thresholds['drawdown']:.1%}")

        # Check volatility
        if len(portfolio_data['Rolling_Volatility']) > 0:
            recent_vol = portfolio_data['Rolling_Volatility'].iloc[-1]
            if not np.isnan(recent_vol) and recent_vol >= self.alert_thresholds['volatility']:
                alerts.append(f"ALERT: Portfolio volatility ({recent_vol:.1%}) "
                            f"has breached threshold of {self.alert_thresholds['volatility']:.1%}")

        # Check monthly return
        if len(portfolio_data['Returns']) >= 21:  # Only check if we have at least a month of data
            monthly_return = portfolio_data['Returns'].iloc[-21:].sum()
            if monthly_return <= self.alert_thresholds['return']:
                alerts.append(f"ALERT: Monthly return ({monthly_return:.1%}) "
                            f"has breached threshold of {self.alert_thresholds['return']:.1%}")

        return alerts

    def advanced_time_series_forecasting(self, method='lstm'):
        """
        Advanced time series forecasting with multiple methods:
        - LSTM
        - Prophet
        - GARCH
        - ARIMA (with auto_arima fallback for optimization)
        """
        if self.returns_data is None:
            raise ValueError("No returns data available")

        # Prepare data for forecasting
        data = self.returns_data.copy()
        
        if method == 'lstm':

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Nested function for creating LSTM sequences
            def create_sequences(data_array, seq_length=10):
                X, y = [], []
                for i in range(len(data_array) - seq_length):
                    X.append(data_array[i:i+seq_length])
                    y.append(data_array[i+seq_length])
                return np.array(X), np.array(y)
            
            X, y = create_sequences(scaled_data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                Dense(len(data.columns))
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=50, verbose=0)
            
            self.forecasting_models['lstm'] = {
                'model': model,
                'scaler': scaler
            }
            
            # Use the last seq_length steps for our forecast
            last_sequence = scaled_data[-10:]
            forecasted_scaled = model.predict(last_sequence.reshape(1, 10, len(data.columns)))
            forecasted_returns = scaler.inverse_transform(forecasted_scaled)[0]
            
            return pd.Series(forecasted_returns, index=data.columns)
        
        elif method == 'prophet':
    
            prophet_forecasts = {}
            for column in data.columns:
                prophet_df = pd.DataFrame({
                    'ds': data.index,
                    'y': data[column]
                })
                
                model = Prophet()
                model.fit(prophet_df)
                
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                
                # For demonstration, just take the last forecasted value
                prophet_forecasts[column] = forecast['yhat'].iloc[-1]
                
            self.forecasting_models['prophet'] = prophet_forecasts
            return pd.Series(prophet_forecasts)
        
        elif method == 'garch':
    
            garch_forecasts = {}
            for column in data.columns:
                model = arch.arch_model(data[column], vol='Garch', p=1, q=1)
                try:
                    results = model.fit(disp='off')
                    forecast = results.forecast(horizon=30)
                    # For demonstration, we retrieve the final row's variance
                    garch_forecasts[column] = forecast.variance.iloc[-1].values[0]
                except Exception as e:
                    print(f"GARCH failed for {column}: {e}")
                    garch_forecasts[column] = np.nan
            
            return pd.Series(garch_forecasts)
        
        elif method == 'arima':

            arima_forecasts = {}
            
            for column in data.columns:
                series = data[column].dropna()

                if len(series) < 10:
                    # Not enough data to reliably fit an ARIMA model
                    arima_forecasts[column] = np.nan
                    continue
                
                try:
                    # Use auto_arima to find the best (p,d,q), but keep it fairly quick:
                    auto_model = auto_arima(
                        series,
                        start_p=1,
                        start_q=1,
                        max_p=3,        # Increase if you need more complex models
                        max_q=3,
                        seasonal=False, # or True if your data is seasonal
                        stepwise=True,  # stepwise search is usually faster
                        suppress_warnings=True,
                        error_action='ignore'
                    )
                    
                    # Now we have a trained auto_arima model, we can forecast
                    forecast_steps = 1
                    prediction = auto_model.predict(n_periods=forecast_steps)
                    arima_forecasts[column] = prediction[0] if len(prediction) > 0 else np.nan

                except Exception as e:
                    print(f"auto_arima failed for {column}: {e}")
                    # Fallback to a fixed ARIMA(1,1,1) if auto_arima fails
                    try:
                        fallback_model = ARIMA(series, order=(1, 1, 1))
                        results = fallback_model.fit()
                        forecast_val = results.forecast(steps=1)
                        arima_forecasts[column] = forecast_val.iloc[-1]
                    except Exception as e2:
                        print(f"Fallback ARIMA(1,1,1) also failed for {column}: {e2}")
                        arima_forecasts[column] = np.nan
            
            self.forecasting_models['arima'] = arima_forecasts
            return pd.Series(arima_forecasts)
        
        else:
            raise ValueError(f"Method '{method}' not recognized.")

    def stress_test_portfolio(self, weights):
            """
            Comprehensive stress testing of portfolio under different scenarios
            """
            stress_results = {}
            
            for scenario, multiplier in self.stress_scenarios.items():
                # Simulate scenario by scaling returns
                stressed_returns = self.returns_data * multiplier
                daily_returns = (stressed_returns * weights).sum(axis=1)
                
                cum_value = (1 + daily_returns).cumprod()
                max_drawdown = (cum_value / cum_value.cummax() - 1).min()
                
                stress_results[scenario] = {
                    'max_drawdown': max_drawdown,
                    'final_value': cum_value.iloc[-1],
                    'volatility': daily_returns.std() * np.sqrt(252)
                }
            
            return stress_results

    def periodic_portfolio_reoptimization(self):
            """
            Dynamically re-optimize portfolio based on predefined frequency
            """
            # Implement sliding window approach
            window_map = {
                'monthly': 21,   # Trading days in a month
                'quarterly': 63,  # Trading days in a quarter
                'annually': 252   # Trading days in a year
            }
            
            window = window_map.get(self.reoptimization_frequency, 63)
            
            # Use rolling window for optimization
            rolling_returns = self.returns_data.rolling(window=window).apply(
                lambda x: x.mean() * 252  # Annualized return
            )
            
            # Use forecasting to adjust expected returns
            forecasted_returns = self.advanced_time_series_forecasting(method='lstm')
            
            # Blend historical and forecasted returns
            blended_returns = (rolling_returns.iloc[-1] + forecasted_returns) / 2
            
            # Re-optimize with updated return expectations
            updated_portfolio = self.optimize_portfolio(
                target_return=blended_returns.mean(),
                risk_free_rate=0.05
            )
            
            return updated_portfolio
    


    def get_sleep_duration(frequency):

        duration_map = {
            'daily':     1   * 24 * 3600,
            'weekly':    7   * 24 * 3600,
            'monthly':   30  * 24 * 3600,
            'quarterly': 90  * 24 * 3600,
            'annually':  365 * 24 * 3600
        }
        
        # Convert to lowercase for case-insensitive matching, default to 'monthly'
        return duration_map.get(frequency.lower(), 30 * 24 * 3600)

    def continuous_portfolio_monitoring(analyzer):
  
        while True:
            try:
                # Forecast and re-optimize
                new_portfolio = analyzer.periodic_portfolio_reoptimization()
                
                # Stress test the newly re-optimized portfolio
                stress_results = analyzer.stress_test_portfolio(new_portfolio['weights'])
                
                # Log or notify about portfolio changes
                print("Portfolio Rebalanced:")
                print(f"New Weights: {new_portfolio['weights']}")
                print("Stress Test Results:")
                for scenario, results in stress_results.items():
                    print(f"{scenario}: {results}")
  
                
            except Exception as e:
                print(f"Error in continuous monitoring: {e}")
            
            # Wait for the next rebalancing period
            time.sleep(get_sleep_duration(analyzer.reoptimization_frequency))

    def backtest_portfolio(self, weights, window=252):
    
        if self.returns_data is None:
            raise ValueError("No returns_data for backtesting.")

        daily_returns = (self.returns_data * weights).sum(axis=1)
        cum_value = (1 + daily_returns).cumprod()

        rolling_vol = daily_returns.rolling(window).std() * np.sqrt(252)
        rf_daily = 0.05 / 252
        rolling_sharpe = ((daily_returns.rolling(window).mean() - rf_daily) * 252)
        rolling_sharpe = rolling_sharpe / rolling_vol.replace(0, np.nan)

        rolling_max = cum_value.expanding().max()
        drawdown = cum_value / rolling_max - 1

        # Calculate additional risk metrics
        max_drawdown = drawdown.min()
        annual_return = (cum_value.iloc[-1] ** (252/len(cum_value)) - 1)
        
        results = pd.DataFrame({
            'Portfolio_Value': cum_value,
            'Returns': daily_returns,
            'Rolling_Volatility': rolling_vol,
            'Rolling_Sharpe': rolling_sharpe,
            'Drawdown': drawdown
        })

        # Check for alerts
        alerts = self.check_alerts(results)
        
        metrics = {
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'final_value': cum_value.iloc[-1],
            'sharpe_ratio': rolling_sharpe.mean(),
            'volatility': rolling_vol.mean(),
            'alerts': alerts
        }

        return results, metrics

    def forward_test_portfolio(self, weights, start_forward, end_forward):
        """
        Paper trade / forward test with monitoring and alerts.
        """
        forward_data = pd.DataFrame()
        for symbol in weights.index:
            try:
                df = yf.download(symbol, start=start_forward, end=end_forward, progress=False)
                forward_data[symbol] = df['Adj Close']
            except Exception as e:
                print(f"Error fetching forward data for {symbol}: {e}")

        # Check if we have any data
        if forward_data.empty:
            print("Warning: No forward data available for testing")
            return None, []

        forward_returns = forward_data.pct_change().dropna(how='all')
        forward_returns.dropna(axis=0, how='any', inplace=True)

        # Check if we have any returns data
        if forward_returns.empty:
            print("Warning: No forward returns data available for testing")
            return None, []

        daily_returns_fw = (forward_returns * weights).sum(axis=1)
        cumulative_value_fw = (1 + daily_returns_fw).cumprod()

        window = min(252, len(daily_returns_fw))  # Adjust window size based on available data
        rolling_vol_fw = daily_returns_fw.rolling(window).std() * np.sqrt(252)
        rf_daily = 0.05 / 252
        rolling_sharpe_fw = ((daily_returns_fw.rolling(window).mean() - rf_daily) * 252)
        rolling_sharpe_fw = rolling_sharpe_fw / rolling_vol_fw.replace(0, np.nan)

        rolling_max_fw = cumulative_value_fw.expanding().max()
        drawdown_fw = cumulative_value_fw / rolling_max_fw - 1

        fw_results = pd.DataFrame({
            'Portfolio_Value': cumulative_value_fw,
            'Returns': daily_returns_fw,
            'Rolling_Volatility': rolling_vol_fw,
            'Rolling_Sharpe': rolling_sharpe_fw,
            'Drawdown': drawdown_fw
        })

        # Check for alerts only if we have data
        alerts = self.check_alerts(fw_results) if not fw_results.empty else []
        
        return fw_results, alerts

    def save_portfolio_performance(self, df_results, title="Portfolio Performance", output_file="portfolio_metrics.txt"):
        """
        Save portfolio performance metrics to a file instead of plotting
        """
        metrics_summary = f"""
    {title}
    {'-' * 50}
    Latest Portfolio Value: {df_results['Portfolio_Value'].iloc[-1]:.2f}
    Latest Rolling Volatility: {df_results['Rolling_Volatility'].iloc[-1]:.2%}
    Latest Rolling Sharpe: {df_results['Rolling_Sharpe'].iloc[-1]:.2f}
    Latest Drawdown: {df_results['Drawdown'].iloc[-1]:.2%}
    Average Daily Return: {df_results['Returns'].mean():.2%}
        """
        
        with open(output_file, 'w') as f:
            f.write(metrics_summary)
            
    def format_notification(self, portfolio, fw_alerts=None, max_items=10):
        """
        Format notification message with dynamic portfolio allocation
        
        Args:
            portfolio (dict): Optimized portfolio dictionary
            fw_alerts (list): List of forward testing alerts
            max_items (int): Maximum number of holdings to display
        
        Returns:
            str: Formatted notification message
        """
        # Ensure portfolio and weights exist
        if not portfolio or 'weights' not in portfolio:
            return "❌ Portfolio data unavailable"
    
        # Current timestamp for context
        current_time = datetime.now().strftime('%Y-%m-%d')
    
        # Prepare notification base
        notification = f"📊 Portfolio Update ({current_time})\n\n"
    
        # Portfolio Performance Metrics
        notification += (
            f"📈 Return: {portfolio.get('expected_return', 0):.1%}\n"
            f"📊 Vol: {portfolio.get('volatility', 0):.1%}\n"
            f"⚖️ Sharpe: {portfolio.get('sharpe_ratio', 0):.2f}\n\n"
        )
    
        # Sort weights and filter significant holdings
        weights = portfolio['weights']
        sorted_holdings = weights[weights >= 0.01].sort_values(ascending=False)
        
        # Key Holdings Section
        notification += "💼 Key Holdings (>1%):\n"
        cumulative_sum = 0
        for stock, weight in sorted_holdings.items():
            # Remove .NS suffix if present
            stock_display = stock.replace('.NS', '')
            notification += f"• {stock_display}: {weight:.1%}\n"
            cumulative_sum += weight
            
            # Limit to max_items
            if len(notification.split('\n')) >= max_items + 5:
                break
        
        # Add remaining if not fully allocated
        if cumulative_sum < 1.0:
            notification += f"• Others: {(1 - cumulative_sum):.1%}\n"
    
        # Sector Exposure Section
        sector_exposure = self.get_sector_exposure(weights)
        sorted_sectors = dict(sorted(sector_exposure.items(), key=lambda x: x[1], reverse=True))
        
        notification += "\n🏢 Sector Allocation (to 100%):\n"
        cumulative_sum = 0
        for sector, exposure in sorted_sectors.items():
            if exposure > 0.01:  # Only show sectors with >1% exposure
                cumulative_sum += exposure
                notification += f"• {sector}: {exposure:.1%} (Cum: {cumulative_sum:.1%})\n"
                
                # Stop if we've reached 100% or very close
                if cumulative_sum >= 0.999:
                    break
    
        # Add Alerts if present
        if fw_alerts:
            notification += "\n⚠️ Alerts:\n"
            for alert in fw_alerts[:3]:  # Limit to 3 most important alerts
                notification += f"• {alert}\n"
            
            if len(fw_alerts) > 3:
                notification += f"• +{len(fw_alerts)-3} more alerts\n"
    
        # Optional: Add a performance summary or recommendation
        if portfolio.get('sharpe_ratio', 0) > 2:
            notification += "\n🌟 Strong Performance: Portfolio showing robust returns\n"
        elif portfolio.get('sharpe_ratio', 0) < 1:
            notification += "\n⚠️ Performance Note: Consider portfolio review\n"
    
        return notification

    def notify_via_telegram(self, message):
        """
        Enhanced Telegram notification with message splitting and formatting
        """
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not bot_token or not chat_id:
            print("Skipping Telegram notification (missing token or chat_id).")
            return

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Split message if too long (Telegram limit is 4096 characters)
        max_length = 4000  # Leave some buffer
        messages = []
        
        if len(message) > max_length:
            # Split into parts
            parts = [message[i:i+max_length] for i in range(0, len(message), max_length)]
            for i, part in enumerate(parts):
                if i == 0:
                    messages.append(part)
                else:
                    messages.append(f"(Continued {i+1}/{len(parts)})\n{part}")
        else:
            messages = [message]

        # Send each part
        for msg in messages:
            try:
                payload = {
                    "chat_id": chat_id,
                    "text": msg,
                    "parse_mode": "Markdown"
                }
                
                resp = requests.post(url, json=payload, timeout=10)
                
                if resp.status_code == 200:
                    print(f"Notification part sent successfully")
                else:
                    # If markdown fails, try without formatting
                    print(f"Trying without markdown formatting...")
                    payload["parse_mode"] = ""
                    resp = requests.post(url, json=payload, timeout=10)
                    
                    if resp.status_code != 200:
                        print(f"Telegram notification failed: {resp.status_code} {resp.text}")
                        
                # Add delay between messages to avoid rate limiting
                if len(messages) > 1:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Error sending Telegram notification: {e}")

def main():
    # Define sector mapping for stocks
    current_time = datetime.now()
    print(f"Analysis running at UTC: {current_time}")

    sector_mapping = {
       "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "INFY.NS": "IT",
    "ICICIBANK.NS": "Banking",
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "SBIN.NS": "Banking",
    "BHARTIARTL.NS": "Telecom",
    "KOTAKBANK.NS": "Banking",
    "LT.NS": "Infrastructure",
    "AXISBANK.NS": "Banking",
    "ASIANPAINT.NS": "FMCG",
    "HCLTECH.NS": "IT",
    "BAJFINANCE.NS": "Financial Services",
    "WIPRO.NS": "IT",
    "MARUTI.NS": "Automobile",
    "ULTRACEMCO.NS": "Cement",
    "NESTLEIND.NS": "FMCG",
    "TITAN.NS": "Consumer Durables",
    "TECHM.NS": "IT",
    "SUNPHARMA.NS": "Pharma",
    "M&M.NS": "Automobile",
    "ADANIGREEN.NS": "Renewable Energy",
    "POWERGRID.NS": "Energy",
    "NTPC.NS": "Energy",
    "ONGC.NS": "Energy",
    "BPCL.NS": "Energy",
    "INDUSINDBK.NS": "Banking",
    "GRASIM.NS": "Cement",
    "ADANIPORTS.NS": "Logistics",
    "JSWSTEEL.NS": "Steel",
    "COALINDIA.NS": "Energy",
    "DRREDDY.NS": "Pharma",
    "APOLLOHOSP.NS": "Healthcare",
    "EICHERMOT.NS": "Automobile",
    "BAJAJFINSV.NS": "Financial Services",
    "TATAMOTORS.NS": "Automobile",
    "DIVISLAB.NS": "Pharma",
    "HDFCLIFE.NS": "Insurance",
    "CIPLA.NS": "Pharma",
    "HEROMOTOCO.NS": "Automobile",
    "SBICARD.NS": "Financial Services",
    "ADANIENT.NS": "Conglomerate",
    "UPL.NS": "Chemicals",
    "BRITANNIA.NS": "FMCG",
    "ICICIPRULI.NS": "Insurance",
    "SHREECEM.NS": "Cement",
    "PIDILITIND.NS": "Chemicals",
    "DMART.NS": "Retail",
    "ABB.NS": "Industrial Equipment",
    "AIAENG.NS": "Engineering",
    "ALKEM.NS": "Pharma",
    "AMBUJACEM.NS": "Cement",
    "AUROPHARMA.NS": "Pharma",
    "BANDHANBNK.NS": "Banking",
    "BERGEPAINT.NS": "FMCG",
    "BOSCHLTD.NS": "Automobile",
    "CANBK.NS": "Banking",
    "CHOLAFIN.NS": "Financial Services",
    "CUMMINSIND.NS": "Industrial Equipment",
    "DABUR.NS": "FMCG",
    "DLF.NS": "Real Estate",
    "ESCORTS.NS": "Automobile",
    "FEDERALBNK.NS": "Banking",
    "GLAND.NS": "Pharma",
    "GLAXO.NS": "Pharma",
    "GODREJCP.NS": "FMCG",
    "GODREJPROP.NS": "Real Estate",
    "HAL.NS": "Aerospace",
    "HAVELLS.NS": "Consumer Durables",
    "IGL.NS": "Energy",
    "IRCTC.NS": "Transportation",
    "LICI.NS": "Insurance",
    "LUPIN.NS": "Pharma",
    "NAUKRI.NS": "IT Services",
    "PEL.NS": "Financial Services",
    "PFC.NS": "Energy",
    "PNB.NS": "Banking",
    "RECLTD.NS": "Energy",
    "SIEMENS.NS": "Industrial Equipment",
    "SRF.NS": "Chemicals",
    "TATACHEM.NS": "Chemicals",
    "TATAELXSI.NS": "IT",
    "TRENT.NS": "Retail",
    "TVSMOTOR.NS": "Automobile",
    "VBL.NS": "FMCG",
    "VEDL.NS": "Metals",
    "WHIRLPOOL.NS": "Consumer Durables",
    "ZOMATO.NS": "Food Services",
    "INOXWIND.NS": "Renewable Energy",
    "SOLARA.NS": "Pharma",
    "INOXGREEN.NS": "Renewable Energy",
    "MOTHERSON.NS": "Automobile",
    "LLOYDSENGG.NS": "Steel",
    "HCC.NS": "Infrastructure",
    "CAMLINFINE.NS": "Chemicals",
    "AURUM.NS": "Real Estate",
    "AXISCADES.NS": "Engineering"
    }

    backtest_start = (current_time - timedelta(days=3*365)).strftime('%Y-%m-%d')
    backtest_end = current_time.strftime('%Y-%m-%d')
    
    # Additional configuration options
    config = {
        'reoptimization_frequency': 'quarterly',
        'forecast_methods': ['lstm', 'prophet', 'garch'],
        'stress_test_scenarios': {
            'bull_market': 1.5,
            'bear_market': 0.5,
            'high_volatility': 2.0
        },
        'risk_parameters': {
            'target_return': 0.15,
            'risk_free_rate': 0.05,
            'max_sector_exposure': 0.30,
            'max_stock_weight': 0.15
        }
    }

    analyzer = EnhancedIndianMarketAnalyzer(
        start_date=backtest_start,
        end_date=backtest_end,
        sector_mapping=sector_mapping
    )

    # Update analyzer with configuration
    analyzer.reoptimization_frequency = config['reoptimization_frequency']
    analyzer.stress_scenarios = config['stress_test_scenarios']

    try:
        # Create output directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        logging.info("1. Fetching and preparing data...")
        analyzer.fetch_data(list(sector_mapping.keys()))
        analyzer.calculate_returns()
        analyzer.build_correlation_matrix(threshold=0.3)
        analyzer.create_network_graph()

        logging.info("2. Advanced Forecasting...")
        # Run multiple forecasting methods
        forecasts = {}
        for method in config['forecast_methods']:
            try:
                forecast = analyzer.advanced_time_series_forecasting(method)
                forecasts[method] = forecast
                logging.info(f"Forecast using {method} method completed.")
            except Exception as e:
                logging.exception(f"Forecasting with {method} failed.")

        logging.info("3. Detecting communities and analyzing performance...")
        partition = analyzer.detect_communities_louvain()
        comm_analysis = analyzer.analyze_communities(partition)

        logging.info("4. Optimizing portfolio...")
        portfolio = analyzer.optimize_portfolio(
            target_return=config['risk_parameters']['target_return'],
            risk_free_rate=config['risk_parameters']['risk_free_rate'],
            max_sector_exposure=config['risk_parameters']['max_sector_exposure'],
            max_stock_weight=config['risk_parameters']['max_stock_weight']
        )

        if portfolio is None:
            raise ValueError("Portfolio optimization failed")

        # Comprehensive Reporting
        logging.info("5. Generating Comprehensive Portfolio Analysis...")
        
        # Stress Testing
        logging.info("Running Stress Tests...")
        stress_results = analyzer.stress_test_portfolio(portfolio['weights'])
        
        # Save Stress Test Results
        with open('results/stress_test_results.txt', 'w') as f:
            f.write("Portfolio Stress Test Results:\n")
            for scenario, results in stress_results.items():
                f.write(f"\n{scenario.upper()} Scenario:\n")
                if all(k in results for k in ["max_drawdown", "final_value", "volatility"]):
                    f.write(f"Max Drawdown: {results['max_drawdown']:.2%}\n")
                    f.write(f"Final Portfolio Value: {results['final_value']:.2f}\n")
                    f.write(f"Volatility: {results['volatility']:.2%}\n")
                else:
                    f.write("One or more keys are missing from this scenario's results.\n")

        # Periodic Re-optimization Simulation
        logging.info("Simulating Periodic Re-optimization...")
        periodic_portfolio = analyzer.periodic_portfolio_reoptimization()

        # Detailed Reporting
        logging.info("6. Generating Detailed Reports...")
        
        # Portfolio Allocation Report
        with open('results/comprehensive_portfolio_report.txt', 'w') as f:
            f.write("Comprehensive Portfolio Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            # Original Portfolio Details
            f.write("Original Portfolio Optimization:\n")
            f.write(f"Expected Return: {portfolio['expected_return']:.2%}\n")
            f.write(f"Volatility: {portfolio['volatility']:.2%}\n")
            f.write(f"Sharpe Ratio: {portfolio['sharpe_ratio']:.2f}\n\n")
            
            # Periodic Re-optimization Comparison
            f.write("Periodic Re-optimization Results:\n")
            f.write(f"Expected Return: {periodic_portfolio['expected_return']:.2%}\n")
            f.write(f"Volatility: {periodic_portfolio['volatility']:.2%}\n")
            f.write(f"Sharpe Ratio: {periodic_portfolio['sharpe_ratio']:.2f}\n\n")
            
            # Forecasts Comparison
            f.write("Forecasting Results:\n")
            for method, forecast in forecasts.items():
                f.write(f"{method.upper()} Forecast:\n")
                f.write(str(forecast) + "\n\n")

        logging.info("7. Running Backtest and Forward Test...")
        # Existing backtest and forward test code remains the same
        backtest_results, backtest_metrics = analyzer.backtest_portfolio(portfolio['weights'])
        
        test_period = 30
        forward_start = (current_time - timedelta(days=test_period)).strftime('%Y-%m-%d')
        forward_end = current_time.strftime('%Y-%m-%d')

        fw_results, fw_alerts = analyzer.forward_test_portfolio(
            portfolio['weights'], 
            forward_start, 
            forward_end
        )

        # Notification with enhanced information
        notification = analyzer.format_notification(portfolio, fw_alerts)

        # Send Telegram Notification
        analyzer.notify_via_telegram(notification)

    except Exception as e:
        logging.exception("Comprehensive Analysis Failed.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()