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

    def backtest_portfolio(self, weights, window=252):
        """
        Backtest the portfolio with enhanced monitoring and alerts.
        """
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

    def format_notification(self, portfolio, fw_alerts, max_items=5):
        """
        Format notification message while keeping it within Telegram's limits
        """
        # Start with basic portfolio metrics
        notification = (
            f"📊 Portfolio Update ({datetime.now().strftime('%Y-%m-%d')})\n\n"
            f"📈 Return: {portfolio['expected_return']:.1%}\n"
            f"📊 Vol: {portfolio['volatility']:.1%}\n"
            f"⚖️ Sharpe: {portfolio['sharpe_ratio']:.2f}\n"
        )

        # Calculate cumulative weights and show only significant holdings
        sorted_holdings = portfolio['weights'].sort_values(ascending=False)
        cumulative_sum = 0
        notification += "\n💼 Key Holdings (>1%):\n"
        for stock, weight in sorted_holdings.items():
            if weight >= 0.01:  # Only show holdings >= 1%
                cumulative_sum += weight
                notification += f"• {stock.replace('.NS', '')}: {weight:.1%}\n"
            if cumulative_sum >= 1.0:
                break
        if cumulative_sum < 1.0:
            notification += f"• Others: {(1 - cumulative_sum):.1%}\n"

        # Show only major sector exposures
        sorted_sectors = dict(sorted(portfolio['sector_exposure'].items(), 
                                key=lambda x: x[1], reverse=True))
        cumulative_sum = 0
        sectors_added = 0
        
# Calculate cumulative sector exposure with running total
        sorted_sectors = dict(sorted(portfolio['sector_exposure'].items(), 
                                key=lambda x: x[1], reverse=True))
        cumulative_sum = 0
        sector_exposure = "\n🏢 Sector Allocation (to 100%):\n"

        for sector, exposure in sorted_sectors.items():
            if exposure > 0.001:  # Filter out extremely small exposures (< 0.1%)
                cumulative_sum += exposure
                sector_exposure += f"• {sector}: {exposure:.1%} (Cum: {cumulative_sum:.1%})\n"
                
                # Break if we've reached 100% or very close to it
                if cumulative_sum >= 0.9999:  # Account for floating point precision
                    break

        # Add any remaining exposure if significant
        remaining = 1.0 - cumulative_sum
        if remaining > 0.001:  # Only show if remaining is > 0.1%
            sector_exposure += f"• Others: {remaining:.1%} (Final: 100%)\n"

        notification += sector_exposure

        # Add alerts if any (limited to most important ones)
        if fw_alerts:
            notification += "\n⚠️ Alerts:\n"
            for alert in fw_alerts[:2]:  # Show maximum 2 most important alerts
                # Remove "ALERT:" prefix to save space
                alert_text = alert.split('ALERT: ')[-1]
                notification += f"• {alert_text}\n"
            if len(fw_alerts) > 2:
                notification += f"• +{len(fw_alerts)-2} more alerts\n"

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
    current_time = datetime.utcnow()
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

    # Initialize analyzer with 3-year backtest period
    backtest_start = (current_time - timedelta(days=3*365)).strftime('%Y-%m-%d')
    backtest_end = current_time.strftime('%Y-%m-%d')
    
    analyzer = EnhancedIndianMarketAnalyzer(
        start_date=backtest_start,
        end_date=backtest_end,
        sector_mapping=sector_mapping
    )

    try:
        # Create output directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        print("1. Fetching historical data...")
        analyzer.fetch_data(list(sector_mapping.keys()))
        analyzer.calculate_returns()
        analyzer.build_correlation_matrix(threshold=0.3)
        analyzer.create_network_graph()

        print("2. Detecting communities and analyzing performance...")
        partition = analyzer.detect_communities_louvain()
        comm_analysis = analyzer.analyze_communities(partition)

        print("3. Optimizing portfolio...")
        portfolio = analyzer.optimize_portfolio(
            target_return=0.15,
            risk_free_rate=0.05,
            max_sector_exposure=0.30,
            max_stock_weight=0.15
        )

        if portfolio is None:
            raise ValueError("Portfolio optimization failed")

        # Save portfolio allocation to file
        with open('results/portfolio_allocation.txt', 'w') as f:
            f.write("Portfolio Allocation:\n\n")
            f.write("Optimal Weights:\n")
            for stock, weight in portfolio['weights'].items():
                if weight > 0.01:
                    f.write(f"{stock}: {weight:.1%}\n")
            
            f.write("\nSector Exposure:\n")
            for sector, exposure in portfolio['sector_exposure'].items():
                f.write(f"{sector}: {exposure:.1%}\n")

        print("4. Running backtest...")
        backtest_results, backtest_metrics = analyzer.backtest_portfolio(portfolio['weights'])
        analyzer.save_portfolio_performance(backtest_results, 
                                         "Backtest Results",
                                         'results/backtest_metrics.txt')

        print("5. Running forward test...")
        test_period = 30
        forward_start = (current_time - timedelta(days=test_period)).strftime('%Y-%m-%d')
        forward_end = current_time.strftime('%Y-%m-%d')

        fw_results, fw_alerts = analyzer.forward_test_portfolio(
            portfolio['weights'], 
            forward_start, 
            forward_end
        )

        if fw_results is not None:
            analyzer.save_portfolio_performance(fw_results,
                                             "Forward Test Results",
                                             'results/forward_test_metrics.txt')

        # Calculate cumulative weights for notification
        sorted_holdings = portfolio['weights'].sort_values(ascending=False)
        cumulative_sum = 0
        holdings_text = "\nPortfolio Holdings (to 100%):\n"
        
        for stock, weight in sorted_holdings.items():
            cumulative_sum += weight
            holdings_text += f"- {stock}: {weight:.1%} (Cum: {cumulative_sum:.1%})\n"
            if cumulative_sum >= 1.0:
                break

        # Calculate cumulative sector exposure
        sorted_sectors = dict(sorted(portfolio['sector_exposure'].items(), 
                                   key=lambda x: x[1], reverse=True))
        cumulative_sum = 0
        sector_text = "\nSector Exposure (to 100%):\n"
        
        for sector, exposure in sorted_sectors.items():
            cumulative_sum += exposure
            sector_text += f"- {sector}: {exposure:.1%} (Cum: {cumulative_sum:.1%})\n"
            if cumulative_sum >= 1.0:
                break

        notification = analyzer.format_notification(portfolio, fw_alerts)
        analyzer.notify_via_telegram(notification)

    except Exception as e:
        print(f"Error formatting/sending notification: {e}")

if __name__ == "__main__":
    main()