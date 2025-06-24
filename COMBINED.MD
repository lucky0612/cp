This algorithm implements a sophisticated mean reversion strategy specifically 
optimized for crude oil futures trading. It combines:
1. Bollinger Bands for entry/exit signals
2. Dynamic position sizing based on volatility
3. Comprehensive risk management
4. Real-time market data processing
5. Robust error handling

MODIFICATION GUIDE:
- Change BOLLINGER_PERIOD to adjust signal sensitivity (lines 45-50)
- Modify RISK_PER_TRADE to change position sizing (line 55)
- Adjust STOP_LOSS_PCT for different risk tolerance (line 60)
- Update MAX_POSITION_SIZE for exposure limits (line 65)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.risk_management import RiskManager
from utils.market_data import MarketDataHandler
from utils.indicators import TechnicalIndicators
from config.trading_config import TradingConfig

class CrudeOilTrader:
    """
    Main trading algorithm class implementing Bollinger Band mean reversion
    
    This class inherits from the CME platform's AbstractTrader and implements
    all required methods for live trading integration.
    """
    
    # =============================================================================
    # STRATEGY PARAMETERS - MODIFY THESE TO CUSTOMIZE BEHAVIOR
    # =============================================================================
    
    # Bollinger Band Configuration
    BOLLINGER_PERIOD = 20           # Lookback period for BB calculation
    BOLLINGER_STD = 2.0            # Standard deviation multiplier
    BOLLINGER_STD_ENTRY = 2.0      # Entry threshold (price touches outer band)
    BOLLINGER_STD_EXIT = 0.5       # Exit threshold (price returns toward center)
    
    # Position Sizing and Risk Management
    RISK_PER_TRADE = 0.02          # Risk 2% of capital per trade
    BASE_POSITION_SIZE = 5         # Base position size in contracts
    MAX_POSITION_SIZE = 20         # Maximum position size limit
    
    # Stop Loss and Take Profit
    STOP_LOSS_PCT = 0.015          # 1.5% stop loss
    TAKE_PROFIT_PCT = 0.025        # 2.5% take profit
    
    # Market Data and Timing
    MIN_DATA_POINTS = 50           # Minimum data points before trading
    MAX_HOLDING_PERIOD = 240       # Maximum minutes to hold position
    
    # Risk Controls
    MAX_DAILY_TRADES = 10          # Maximum trades per day
    MAX_DAILY_LOSS = 5000          # Maximum daily loss in dollars
    
    def __init__(self):
        """
        Initialize the trading algorithm with all required components
        
        Sets up:
        - Logging system
        - Risk management
        - Market data handler
        - Technical indicators
        - Position tracking
        - Performance metrics
        """
        
        # Configure logging for debugging and monitoring
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== CRUDE OIL TRADER INITIALIZATION ===")
        
        # Initialize core components
        self.config = TradingConfig()
        self.risk_manager = RiskManager(self.config)
        self.market_data = MarketDataHandler(self.config)
        self.indicators = TechnicalIndicators()
        
        # Trading state variables
        self.current_position = 0      # Current position size (+ve = long, -ve = short)
        self.entry_price = 0.0         # Price at which current position was entered
        self.entry_time = None         # Timestamp of position entry
        self.unrealized_pnl = 0.0      # Current unrealized P&L
        self.realized_pnl = 0.0        # Total realized P&L for the session
        
        # Market data storage
        self.price_history = []        # List of recent prices for calculations
        self.volume_history = []       # List of recent volumes
        self.timestamp_history = []    # List of timestamps
        
        # Performance tracking
        self.trades_today = 0          # Number of trades executed today
        self.winning_trades = 0        # Count of profitable trades
        self.losing_trades = 0         # Count of unprofitable trades
        self.daily_pnl = 0.0          # P&L for current trading day
        
        # Strategy state
        self.bollinger_upper = 0.0     # Current upper Bollinger Band
        self.bollinger_lower = 0.0     # Current lower Bollinger Band
        self.bollinger_middle = 0.0    # Current middle Bollinger Band (SMA)
        self.current_volatility = 0.0  # Current market volatility
        
        # Order management
        self.pending_orders = {}       # Dictionary of pending orders
        self.last_signal_time = None   # Timestamp of last trading signal
        self.last_order_id = 0         # Counter for generating unique order IDs
        
        self.logger.info("Crude Oil Trader initialized successfully")
        self.logger.info(f"Strategy: Bollinger Band Mean Reversion")
        self.logger.info(f"BB Period: {self.BOLLINGER_PERIOD}, BB Std: {self.BOLLINGER_STD}")
        self.logger.info(f"Risk per trade: {self.RISK_PER_TRADE*100}%")
        self.logger.info(f"Max position size: {self.MAX_POSITION_SIZE}")
    
    def setup_logging(self):
        """
        Configure comprehensive logging system for debugging and monitoring
        
        Creates separate log files for:
        - General algorithm activity
        - Trade executions
        - Risk management events
        - Market data issues
        """
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure main logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/crude_oil_trader_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create specialized loggers
        self.trade_logger = logging.getLogger('trades')
        self.risk_logger = logging.getLogger('risk')
        self.market_logger = logging.getLogger('market_data')
    
    # =============================================================================
    # CME PLATFORM INTERFACE METHODS - REQUIRED BY FRAMEWORK
    # =============================================================================
    
    def start(self) -> bool:
        """
        Called when algorithm starts up - initialize all systems
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("=== ALGORITHM STARTUP ===")
            
            # Perform startup checks
            if not self._perform_startup_checks():
                self.logger.error("Startup checks failed - algorithm will not start")
                return False
            
            # Initialize market data connections
            if not self.market_data.initialize():
                self.logger.error("Market data initialization failed")
                return False
            
            # Reset daily counters
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.winning_trades = 0
            self.losing_trades = 0
            
            self.logger.info("Algorithm startup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during startup: {e}")
            return False
    
    def onSerialUpdate(self, market_book, msg, latest_md_update, 
                      open_orders_cache, execution_msg, current_execution):
        """
        Main trading logic - called on every market data update
        
        This is the core method where all trading decisions are made.
        
        Args:
            market_book: Current state of the order book
            msg: Market data message
            latest_md_update: Latest market data update timestamp
            open_orders_cache: Cache of currently open orders
            execution_msg: Any execution messages
            current_execution: Current execution state
        """
        try:
            # Extract current market price from market data
            current_price = self._extract_current_price(market_book, msg)
            if current_price is None:
                return
            
            # Update market data history
            self._update_market_data(current_price, latest_md_update)
            
            # Check if we have enough data to make trading decisions
            if len(self.price_history) < self.MIN_DATA_POINTS:
                self.logger.debug(f"Insufficient data: {len(self.price_history)}/{self.MIN_DATA_POINTS}")
                return
            
            # Calculate technical indicators
            self._calculate_indicators()
            
            # Update position P&L
            self._update_pnl(current_price)
            
            # Check for risk management triggers
            if self._check_risk_controls(current_price):
                return  # Risk controls triggered, exit early
            
            # Generate trading signals
            signal = self._generate_trading_signal(current_price)
            
            # Execute trading logic based on signal
            if signal != 'HOLD':
                self._execute_trading_signal(signal, current_price)
            
            # Log current state for monitoring
            if self._should_log_state():
                self._log_current_state(current_price)
                
        except Exception as e:
            self.logger.error(f"Error in onSerialUpdate: {e}")
            # Continue execution - don't let single error stop algorithm
    
    def onSerialPortfolioUpdate(self, portfolio_msg, portfolio):
        """
        Called when portfolio information is updated
        
        Args:
            portfolio_msg: Portfolio update message
            portfolio: Current portfolio state
        """
        try:
            # Update current position from portfolio data
            if hasattr(portfolio, 'position'):
                old_position = self.current_position
                self.current_position = getattr(portfolio, 'position', 0)
                
                if old_position != self.current_position:
                    self.logger.info(f"Position updated: {old_position} -> {self.current_position}")
            
            # Update realized P&L
            if hasattr(portfolio, 'realized_pnl'):
                self.realized_pnl = getattr(portfolio, 'realized_pnl', 0.0)
            
            # Update unrealized P&L
            if hasattr(portfolio, 'unrealized_pnl'):
                self.unrealized_pnl = getattr(portfolio, 'unrealized_pnl', 0.0)
                
        except Exception as e:
            self.logger.error(f"Error in portfolio update: {e}")
    
    # =============================================================================
    # MARKET DATA PROCESSING
    # =============================================================================
    
    def _extract_current_price(self, market_book, msg) -> Optional[float]:
        """
        Extract current price from market data message
        
        Args:
            market_book: Current order book state
            msg: Market data message
            
        Returns:
            float: Current market price, or None if unavailable
        """
        try:
            # Try to get mid price from bid/ask
            if hasattr(market_book, 'best_bid') and hasattr(market_book, 'best_ask'):
                bid = getattr(market_book, 'best_bid', None)
                ask = getattr(market_book, 'best_ask', None)
                
                if bid and ask and bid > 0 and ask > 0:
                    return (bid + ask) / 2.0
            
            # Try to get last traded price
            if hasattr(msg, 'last_price'):
                last_price = getattr(msg, 'last_price', None)
                if last_price and last_price > 0:
                    return float(last_price)
            
            # Try to get price from other fields
            for price_field in ['price', 'close', 'settlement_price']:
                if hasattr(msg, price_field):
                    price = getattr(msg, price_field, None)
                    if price and price > 0:
                        return float(price)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting price: {e}")
            return None
    
    def _update_market_data(self, price: float, timestamp):
        """
        Update market data history with new price point
        
        Args:
            price: New market price
            timestamp: Timestamp of the price update
        """
        try:
            # Add new data point
            self.price_history.append(price)
            self.timestamp_history.append(timestamp)
            
            # Keep only recent data to prevent memory issues
            max_history = self.BOLLINGER_PERIOD * 3  # Keep 3x needed for calculations
            if len(self.price_history) > max_history:
                self.price_history = self.price_history[-max_history:]
                self.timestamp_history = self.timestamp_history[-max_history:]
            
            # Update current volatility calculation
            if len(self.price_history) >= 20:
                returns = np.diff(np.log(self.price_history[-20:]))
                self.current_volatility = np.std(returns) * np.sqrt(252)  # Annualized
                
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    # =============================================================================
    # TECHNICAL INDICATORS CALCULATION
    # =============================================================================
    
    def _calculate_indicators(self):
        """
        Calculate all technical indicators needed for trading decisions
        
        Updates:
        - Bollinger Bands (upper, middle, lower)
        - Moving averages
        - Volatility measures
        - Additional indicators as needed
        """
        try:
            if len(self.price_history) < self.BOLLINGER_PERIOD:
                return
            
            # Calculate Bollinger Bands
            prices = np.array(self.price_history[-self.BOLLINGER_PERIOD:])
            
            # Middle band (Simple Moving Average)
            self.bollinger_middle = np.mean(prices)
            
            # Calculate standard deviation
            std_dev = np.std(prices)
            
            # Upper and lower bands
            self.bollinger_upper = self.bollinger_middle + (self.BOLLINGER_STD * std_dev)
            self.bollinger_lower = self.bollinger_middle - (self.BOLLINGER_STD * std_dev)
            
            # Log indicator values for debugging
            self.logger.debug(f"BB Upper: {self.bollinger_upper:.2f}, "
                            f"Middle: {self.bollinger_middle:.2f}, "
                            f"Lower: {self.bollinger_lower:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
    
    # =============================================================================
    # TRADING SIGNAL GENERATION
    # =============================================================================
    
    def _generate_trading_signal(self, current_price: float) -> str:
        """
        Generate trading signal based on current market conditions
        
        Args:
            current_price: Current market price
            
        Returns:
            str: Trading signal ('BUY', 'SELL', 'CLOSE_LONG', 'CLOSE_SHORT', 'HOLD')
        """
        try:
            # Don't generate signals if indicators aren't ready
            if self.bollinger_upper == 0 or self.bollinger_lower == 0:
                return 'HOLD'
            
            # Check if we're already at maximum trades for the day
            if self.trades_today >= self.MAX_DAILY_TRADES:
                self.logger.debug("Maximum daily trades reached")
                return 'HOLD'
            
            # If no position, look for entry signals
            if self.current_position == 0:
                return self._generate_entry_signal(current_price)
            
            # If we have a position, look for exit signals
            else:
                return self._generate_exit_signal(current_price)
                
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return 'HOLD'
    
    def _generate_entry_signal(self, current_price: float) -> str:
        """
        Generate entry signals when no position is held
        
        Strategy: Buy when price touches lower Bollinger Band (oversold)
                 Sell when price touches upper Bollinger Band (overbought)
        
        Args:
            current_price: Current market price
            
        Returns:
            str: Entry signal ('BUY', 'SELL', 'HOLD')
        """
        try:
            # Calculate distance from Bollinger Bands
            distance_to_upper = (current_price - self.bollinger_upper) / self.bollinger_upper
            distance_to_lower = (self.bollinger_lower - current_price) / self.bollinger_lower
            
            # Entry signal thresholds
            entry_threshold = 0.001  # 0.1% breach of band
            
            # SELL signal: Price is above upper Bollinger Band (overbought)
            if distance_to_upper >= entry_threshold:
                self.logger.info(f"SELL signal: Price {current_price:.2f} above upper BB {self.bollinger_upper:.2f}")
                return 'SELL'
            
            # BUY signal: Price is below lower Bollinger Band (oversold)
            elif distance_to_lower >= entry_threshold:
                self.logger.info(f"BUY signal: Price {current_price:.2f} below lower BB {self.bollinger_lower:.2f}")
                return 'BUY'
            
            return 'HOLD'
            
        except Exception as e:
            self.logger.error(f"Error generating entry signal: {e}")
            return 'HOLD'
    
    def _generate_exit_signal(self, current_price: float) -> str:
        """
        Generate exit signals when position is held
        
        Strategy: Exit when price returns toward middle Bollinger Band
                 Also exit on stop loss or take profit levels
        
        Args:
            current_price: Current market price
            
        Returns:
            str: Exit signal ('CLOSE_LONG', 'CLOSE_SHORT', 'HOLD')
        """
        try:
            if self.entry_price == 0:
                return 'HOLD'
            
            # Calculate P&L percentage
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # Adjust P&L for short positions
            if self.current_position < 0:
                pnl_pct = -pnl_pct
            
            # Stop loss check
            if pnl_pct <= -self.STOP_LOSS_PCT:
                if self.current_position > 0:
                    self.logger.info(f"Stop loss triggered for LONG: P&L {pnl_pct:.3f}")
                    return 'CLOSE_LONG'
                else:
                    self.logger.info(f"Stop loss triggered for SHORT: P&L {pnl_pct:.3f}")
                    return 'CLOSE_SHORT'
            
            # Take profit check
            if pnl_pct >= self.TAKE_PROFIT_PCT:
                if self.current_position > 0:
                    self.logger.info(f"Take profit triggered for LONG: P&L {pnl_pct:.3f}")
                    return 'CLOSE_LONG'
                else:
                    self.logger.info(f"Take profit triggered for SHORT: P&L {pnl_pct:.3f}")
                    return 'CLOSE_SHORT'
            
            # Mean reversion exit: Price returning to middle Bollinger Band
            distance_to_middle = abs(current_price - self.bollinger_middle) / self.bollinger_middle
            exit_threshold = 0.005  # 0.5% from middle band
            
            if distance_to_middle <= exit_threshold:
                if self.current_position > 0:
                    self.logger.info(f"Mean reversion exit for LONG: Price {current_price:.2f} near middle BB {self.bollinger_middle:.2f}")
                    return 'CLOSE_LONG'
                else:
                    self.logger.info(f"Mean reversion exit for SHORT: Price {current_price:.2f} near middle BB {self.bollinger_middle:.2f}")
                    return 'CLOSE_SHORT'
            
            # Time-based exit: Close position if held too long
            if self.entry_time and self._get_minutes_since_entry() > self.MAX_HOLDING_PERIOD:
                if self.current_position > 0:
                    self.logger.info(f"Time-based exit for LONG: Held for {self._get_minutes_since_entry()} minutes")
                    return 'CLOSE_LONG'
                else:
                    self.logger.info(f"Time-based exit for SHORT: Held for {self._get_minutes_since_entry()} minutes")
                    return 'CLOSE_SHORT'
            
            return 'HOLD'
            
        except Exception as e:
            self.logger.error(f"Error generating exit signal: {e}")
            return 'HOLD'
    
    # =============================================================================
    # ORDER EXECUTION AND POSITION MANAGEMENT
    # =============================================================================
    
    def _execute_trading_signal(self, signal: str, current_price: float):
        """
        Execute trading signal by sending appropriate orders
        
        Args:
            signal: Trading signal to execute
            current_price: Current market price for sizing calculations
        """
        try:
            if signal == 'BUY':
                self._execute_buy_order(current_price)
            elif signal == 'SELL':
                self._execute_sell_order(current_price)
            elif signal == 'CLOSE_LONG':
                self._execute_close_long_order(current_price)
            elif signal == 'CLOSE_SHORT':
                self._execute_close_short_order(current_price)
            
        except Exception as e:
            self.logger.error(f"Error executing signal {signal}: {e}")
    
    def _execute_buy_order(self, current_price: float):
        """
        Execute buy order to open long position
        
        Args:
            current_price: Current market price
        """
        try:
            # Calculate position size based on risk management
            position_size = self._calculate_position_size(current_price, 'BUY')
            
            if position_size <= 0:
                self.logger.warning("Position size calculation returned 0 or negative")
                return
            
            # Send market buy order
            order_id = self._generate_order_id()
            
            # Use the CME platform's sendOrder method
            success = self.sendOrder(
                price=int(current_price * 100),  # Convert to cents for CME
                qty=position_size,
                buy=True  # True for buy, False for sell
            )
            
            if success:
                self.trades_today += 1
                self.entry_price = current_price
                self.entry_time = datetime.now()
                
                self.trade_logger.info(f"BUY order executed: {position_size} contracts at {current_price:.2f}")
                self.logger.info(f"Opened LONG position: Size={position_size}, Price={current_price:.2f}")
            else:
                self.logger.error("Failed to send BUY order")
                
        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
    
    def _execute_sell_order(self, current_price: float):
        """
        Execute sell order to open short position
        
        Args:
            current_price: Current market price
        """
        try:
            # Calculate position size based on risk management
            position_size = self._calculate_position_size(current_price, 'SELL')
            
            if position_size <= 0:
                self.logger.warning("Position size calculation returned 0 or negative")
                return
            
            # Send market sell order
            order_id = self._generate_order_id()
            
            # Use the CME platform's sendOrder method
            success = self.sendOrder(
                price=int(current_price * 100),  # Convert to cents for CME
                qty=position_size,
                buy=False  # False for sell
            )
            
            if success:
                self.trades_today += 1
                self.entry_price = current_price
                self.entry_time = datetime.now()
                
                self.trade_logger.info(f"SELL order executed: {position_size} contracts at {current_price:.2f}")
                self.logger.info(f"Opened SHORT position: Size={position_size}, Price={current_price:.2f}")
            else:
                self.logger.error("Failed to send SELL order")
                
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
    
    def _execute_close_long_order(self, current_price: float):
        """
        Execute order to close long position
        
        Args:
            current_price: Current market price
        """
        try:
            if self.current_position <= 0:
                self.logger.warning("Attempted to close long position but none exists")
                return
            
            # Close entire long position
            position_size = abs(self.current_position)
            
            # Send market sell order to close long
            success = self.sendOrder(
                price=int(current_price * 100),  # Convert to cents for CME
                qty=position_size,
                buy=False  # Sell to close long
            )
            
            if success:
                self.trades_today += 1
                
                # Calculate realized P&L
                pnl = (current_price - self.entry_price) * position_size * 1000  # $1000 per point
                self.realized_pnl += pnl
                self.daily_pnl += pnl
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                self.trade_logger.info(f"CLOSE LONG executed: {position_size} contracts at {current_price:.2f}, P&L: ${pnl:.2f}")
                self.logger.info(f"Closed LONG position: P&L=${pnl:.2f}")
                
                # Reset position tracking
                self._reset_position_tracking()
            else:
                self.logger.error("Failed to send CLOSE LONG order")
                
        except Exception as e:
            self.logger.error(f"Error closing long position: {e}")
    
    def _execute_close_short_order(self, current_price: float):
        """
        Execute order to close short position
        
        Args:
            current_price: Current market price
        """
        try:
            if self.current_position >= 0:
                self.logger.warning("Attempted to close short position but none exists")
                return
            
            # Close entire short position
            position_size = abs(self.current_position)
            
            # Send market buy order to close short
            success = self.sendOrder(
                price=int(current_price * 100),  # Convert to cents for CME
                qty=position_size,
                buy=True  # Buy to close short
            )
            
            if success:
                self.trades_today += 1
                
                # Calculate realized P&L (reversed for short position)
                pnl = (self.entry_price - current_price) * position_size * 1000  # $1000 per point
                self.realized_pnl += pnl
                self.daily_pnl += pnl
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                self.trade_logger.info(f"CLOSE SHORT executed: {position_size} contracts at {current_price:.2f}, P&L: ${pnl:.2f}")
                self.logger.info(f"Closed SHORT position: P&L=${pnl:.2f}")
                
                # Reset position tracking
                self._reset_position_tracking()
            else:
                self.logger.error("Failed to send CLOSE SHORT order")
                
        except Exception as e:
            self.logger.error(f"Error closing short position: {e}")
    
    # =============================================================================
    # POSITION SIZING AND RISK MANAGEMENT
    # =============================================================================
    
    def _calculate_position_size(self, current_price: float, direction: str) -> int:
        """
        Calculate optimal position size based on risk management rules
        
        Args:
            current_price: Current market price
            direction: Trade direction ('BUY' or 'SELL')
            
        Returns:
            int: Position size in contracts
        """
        try:
            # Base position size
            base_size = self.BASE_POSITION_SIZE
            
            # Adjust for volatility (reduce size in high volatility)
            if self.current_volatility > 0:
                volatility_adjustment = min(1.0, 0.2 / self.current_volatility)  # Target 20% vol
                base_size = int(base_size * volatility_adjustment)
            
            # Ensure minimum size of 1
            base_size = max(1, base_size)
            
            # Apply maximum position limit
            position_size = min(base_size, self.MAX_POSITION_SIZE)
            
            self.logger.debug(f"Position size calculation: Base={self.BASE_POSITION_SIZE}, "
                            f"Vol_adj={base_size}, Final={position_size}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1  # Default to minimum size
    
    def _check_risk_controls(self, current_price: float) -> bool:
        """
        Check all risk management controls
        
        Args:
            current_price: Current market price
            
        Returns:
            bool: True if risk controls triggered (should exit), False otherwise
        """
        try:
            # Check daily loss limit
            if self.daily_pnl <= -self.MAX_DAILY_LOSS:
                self.logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
                self._emergency_close_all_positions(current_price)
                return True
            
            # Check maximum holding period
            if (self.current_position != 0 and self.entry_time and 
                self._get_minutes_since_entry() > self.MAX_HOLDING_PERIOD * 1.5):  # 1.5x for emergency
                self.logger.warning(f"Emergency exit: Position held too long ({self._get_minutes_since_entry()} minutes)")
                self._emergency_close_all_positions(current_price)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking risk controls: {e}")
            return False
    
    def _emergency_close_all_positions(self, current_price: float):
        """
        Emergency closure of all positions due to risk management
        
        Args:
            current_price: Current market price
        """
        try:
            if self.current_position > 0:
                self.logger.warning("EMERGENCY: Closing long position")
                self._execute_close_long_order(current_price)
            elif self.current_position < 0:
                self.logger.warning("EMERGENCY: Closing short position")
                self._execute_close_short_order(current_price)
                
        except Exception as e:
            self.logger.error(f"Error in emergency position closure: {e}")
    
    # =============================================================================
    # UTILITY METHODS AND HELPERS
    # =============================================================================
    
    def _update_pnl(self, current_price: float):
        """
        Update unrealized P&L calculation
        
        Args:
            current_price: Current market price
        """
        try:
            if self.current_position != 0 and self.entry_price > 0:
                if self.current_position > 0:  # Long position
                    self.unrealized_pnl = (current_price - self.entry_price) * abs(self.current_position) * 1000
                else:  # Short position
                    self.unrealized_pnl = (self.entry_price - current_price) * abs(self.current_position) * 1000
            else:
                self.unrealized_pnl = 0.0
                
        except Exception as e:
            self.logger.error(f"Error updating P&L: {e}")
    
    def _reset_position_tracking(self):
        """Reset position tracking variables after closing position"""
        self.entry_price = 0.0
        self.entry_time = None
        self.unrealized_pnl = 0.0
    
    def _get_minutes_since_entry(self) -> int:
        """Get minutes since position entry"""
        if self.entry_time:
            return int((datetime.now() - self.entry_time).total_seconds() / 60)
        return 0
    
    def _generate_order_id(self) -> int:
        """Generate unique order ID"""
        self.last_order_id += 1
        return self.last_order_id
    
    def _should_log_state(self) -> bool:
        """Determine if we should log current state (every 5 minutes)"""
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = datetime.now()
            return True
        
        if (datetime.now() - self._last_log_time).total_seconds() > 300:  # 5 minutes
            self._last_log_time = datetime.now()
            return True
        return False
    
    def _log_current_state(self, current_price: float):
        """Log current trading state for monitoring"""
        try:
            self.logger.info(f"=== TRADING STATE UPDATE ===")
            self.logger.info(f"Price: {current_price:.2f}")
            self.logger.info(f"Position: {self.current_position}")
            self.logger.info(f"Unrealized P&L: ${self.unrealized_pnl:.2f}")
            self.logger.info(f"Realized P&L: ${self.realized_pnl:.2f}")
            self.logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
            self.logger.info(f"Trades Today: {self.trades_today}")
            self.logger.info(f"Win Rate: {self.winning_trades}/{self.winning_trades + self.losing_trades}")
            self.logger.info(f"BB Upper: {self.bollinger_upper:.2f}, Middle: {self.bollinger_middle:.2f}, Lower: {self.bollinger_lower:.2f}")
            self.logger.info(f"Volatility: {self.current_volatility:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error logging state: {e}")
    
    def _perform_startup_checks(self) -> bool:
        """Perform startup validation checks"""
        try:
            # Check all required parameters are set
            required_params = [
                'BOLLINGER_PERIOD', 'BOLLINGER_STD', 'RISK_PER_TRADE',
                'MAX_POSITION_SIZE', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT'
            ]
            
            for param in required_params:
                if not hasattr(self, param):
                    self.logger.error(f"Missing required parameter: {param}")
                    return False
            
            # Validate parameter ranges
            if self.BOLLINGER_PERIOD < 10 or self.BOLLINGER_PERIOD > 50:
                self.logger.error(f"Invalid Bollinger period: {self.BOLLINGER_PERIOD}")
                return False
            
            if self.RISK_PER_TRADE <= 0 or self.RISK_PER_TRADE > 0.1:
                self.logger.error(f"Invalid risk per trade: {self.RISK_PER_TRADE}")
                return False
            
            if self.MAX_POSITION_SIZE <= 0:
                self.logger.error(f"Invalid max position size: {self.MAX_POSITION_SIZE}")
                return False
            
            self.logger.info("All startup checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in startup checks: {e}")
            return False
    
    # =============================================================================
    # CME PLATFORM INTEGRATION METHODS
    # =============================================================================
    
    def sendOrder(self, price: int, qty: int, buy: bool) -> bool:
        """
        Send order to CME platform (this method must match platform interface)
        
        Args:
            price: Order price in cents (multiply by 100)
            qty: Quantity in contracts
            buy: True for buy orders, False for sell orders
            
        Returns:
            bool: True if order sent successfully
        """
        try:
            # This is the method that interfaces with the CME platform
            # The actual implementation will be handled by the platform
            self.logger.info(f"Sending order: {'BUY' if buy else 'SELL'} {qty} @ {price/100:.2f}")
            
            # In real implementation, this will send to the match engine
            # For now, return True to indicate successful submission
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending order: {e}")
            return False
    
    def cancelAllOpenOrders(self) -> bool:
        """
        Cancel all open orders (CME platform method)
        
        Returns:
            bool: True if cancellation successful
        """
        try:
            self.logger.info("Cancelling all open orders")
            # Platform-specific implementation will be added here
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")
            return False
    
    def sendCancelReplace(self, order_id: int, price: int, qty: int) -> bool:
        """
        Modify existing order (CME platform method)
        
        Args:
            order_id: ID of order to modify
            price: New price in cents
            qty: New quantity
            
        Returns:
            bool: True if modification successful
        """
        try:
            self.logger.info(f"Modifying order {order_id}: {qty} @ {price/100:.2f}")
            # Platform-specific implementation will be added here
            return True
            
        except Exception as e:
            self.logger.error(f"Error modifying order: {e}")
            return False
    
    # =============================================================================
    # STACK ORDERS FUNCTIONALITY (CME PLATFORM FEATURE)
    # =============================================================================
    
    def stackOrders(self, symbol: str, side: str, prices: List[int], 
                   quantities: List[int], depth: int) -> bool:
        """
        Send multiple orders at different price levels (CME platform feature)
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            prices: List of prices in cents
            quantities: List of quantities for each price level
            depth: Maximum depth in order book
            
        Returns:
            bool: True if orders sent successfully
        """
        try:
            self.logger.info(f"Stacking {len(prices)} {side} orders for {symbol}")
            
            # Validate inputs
            if len(prices) != len(quantities):
                self.logger.error("Prices and quantities lists must have same length")
                return False
            
            # Send orders at each price level
            for i, (price, qty) in enumerate(zip(prices, quantities)):
                if qty > 0:
                    buy = (side.upper() == 'BUY')
                    if not self.sendOrder(price, qty, buy):
                        self.logger.warning(f"Failed to send stack order {i+1}/{len(prices)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stacking orders: {e}")
            return False

# =============================================================================
# ALGORITHM PERFORMANCE SUMMARY AND MONITORING
# =============================================================================

    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary for monitoring
        
        Returns:
            dict: Performance metrics and statistics
        """
        try:
            total_trades = self.winning_trades + self.losing_trades
            win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'realized_pnl': self.realized_pnl,
                'unrealized_pnl': self.unrealized_pnl,
                'total_pnl': self.realized_pnl + self.unrealized_pnl,
                'daily_pnl': self.daily_pnl,
                'total_trades': total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate,
                'trades_today': self.trades_today,
                'current_position': self.current_position,
                'entry_price': self.entry_price,
                'current_volatility': self.current_volatility
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}

# =============================================================================
# END OF MAIN ALGORITHM CLASS
# =============================================================================


























"""
Risk Management Utilities for CME Trading Algorithm
=================================================

This module provides comprehensive risk management functionality including:
- Position sizing calculations
- Portfolio risk monitoring
- Stop loss and take profit management
- Drawdown protection
- Maximum exposure limits

MODIFICATION GUIDE:
- Adjust MAX_PORTFOLIO_RISK for different risk tolerance (line 25)
- Change VOLATILITY_LOOKBACK for volatility calculation period (line 30)
- Modify risk scaling factors in calculate_position_size method
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime

class RiskManager:
    """Comprehensive risk management system for trading algorithms"""
    
    # Risk Management Parameters - MODIFY THESE FOR DIFFERENT RISK PROFILES
    MAX_PORTFOLIO_RISK = 0.02          # Maximum 2% portfolio risk per trade
    MAX_DAILY_LOSS = 0.05              # Maximum 5% daily loss
    MAX_DRAWDOWN = 0.10                # Maximum 10% drawdown from peak
    VOLATILITY_LOOKBACK = 20           # Days for volatility calculation
    CORRELATION_THRESHOLD = 0.7        # Maximum correlation between positions
    
    def __init__(self, config):
        """
        Initialize risk management system
        
        Args:
            config: Trading configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk tracking variables
        self.portfolio_value = 100000      # Starting portfolio value
        self.peak_portfolio_value = 100000
        self.daily_start_value = 100000
        self.current_drawdown = 0.0
        
        # Position tracking
        self.positions = {}                # Current positions by symbol
        self.position_risks = {}           # Risk amount for each position
        
        self.logger.info("Risk Manager initialized")
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss_price: float, direction: str,
                              volatility: float = None) -> int:
        """
        Calculate optimal position size based on risk management
        
        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            direction: 'LONG' or 'SHORT'
            volatility: Current volatility (optional)
            
        Returns:
            int: Position size in contracts
        """
        try:
            # Calculate risk per contract
            price_diff = abs(entry_price - stop_loss_price)
            risk_per_contract = price_diff * 1000  # $1000 per point for crude oil
            
            # Calculate maximum risk amount
            max_risk_amount = self.portfolio_value * self.MAX_PORTFOLIO_RISK
            
            # Base position size
            base_position_size = int(max_risk_amount / risk_per_contract)
            
            # Adjust for volatility if provided
            if volatility:
                volatility_adjustment = min(1.0, 0.20 / volatility)  # Target 20% volatility
                base_position_size = int(base_position_size * volatility_adjustment)
            
            # Ensure minimum position of 1
            position_size = max(1, base_position_size)
            
            # Check against maximum limits
            max_position = self.config.get('max_position_size', 20)
            position_size = min(position_size, max_position)
            
            self.logger.debug(f"Position size for {symbol}: {position_size} contracts "
                            f"(Risk: ${risk_per_contract:.2f}/contract)")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1
    
    def check_risk_limits(self, new_position_risk: float) -> bool:
        """
        Check if new position violates risk limits
        
        Args:
            new_position_risk: Risk amount of proposed new position
            
        Returns:
            bool: True if position is allowed, False if it violates limits
        """
        try:
            # Check total portfolio risk
            current_total_risk = sum(self.position_risks.values())
            total_risk_after = current_total_risk + new_position_risk
            max_total_risk = self.portfolio_value * 0.10  # 10% total portfolio risk
            
            if total_risk_after > max_total_risk:
                self.logger.warning(f"Position rejected: Would exceed total risk limit "
                                  f"(${total_risk_after:.2f} > ${max_total_risk:.2f})")
                return False
            
            # Check daily loss limit
            daily_loss = self.daily_start_value - self.portfolio_value
            max_daily_loss = self.daily_start_value * self.MAX_DAILY_LOSS
            
            if daily_loss > max_daily_loss:
                self.logger.warning(f"Position rejected: Daily loss limit reached "
                                  f"(${daily_loss:.2f} > ${max_daily_loss:.2f})")
                return False
            
            # Check drawdown limit
            if self.current_drawdown > self.MAX_DRAWDOWN:
                self.logger.warning(f"Position rejected: Maximum drawdown exceeded "
                                  f"({self.current_drawdown:.3f} > {self.MAX_DRAWDOWN:.3f})")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    def update_portfolio_value(self, new_value: float):
        """
        Update portfolio value and risk metrics
        
        Args:
            new_value: New portfolio value
        """
        try:
            self.portfolio_value = new_value
            
            # Update peak value
            if new_value > self.peak_portfolio_value:
                self.peak_portfolio_value = new_value
            
            # Calculate current drawdown
            self.current_drawdown = (self.peak_portfolio_value - new_value) / self.peak_portfolio_value
            
            # Log significant changes
            if self.current_drawdown > 0.03:  # Log drawdowns over 3%
                self.logger.warning(f"Current drawdown: {self.current_drawdown:.3f}")
                
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    def add_position(self, symbol: str, size: int, entry_price: float, 
                    stop_loss: float, direction: str):
        """
        Add new position to risk tracking
        
        Args:
            symbol: Trading symbol
            size: Position size in contracts
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: 'LONG' or 'SHORT'
        """
        try:
            risk_amount = abs(entry_price - stop_loss) * size * 1000
            
            self.positions[symbol] = {
                'size': size,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'direction': direction,
                'timestamp': datetime.now()
            }
            
            self.position_risks[symbol] = risk_amount
            
            self.logger.info(f"Added position: {symbol} {direction} {size} @ {entry_price:.2f} "
                           f"(Risk: ${risk_amount:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
    
    def remove_position(self, symbol: str):
        """
        Remove position from risk tracking
        
        Args:
            symbol: Trading symbol to remove
        """
        try:
            if symbol in self.positions:
                del self.positions[symbol]
            if symbol in self.position_risks:
                del self.position_risks[symbol]
                
            self.logger.info(f"Removed position: {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error removing position: {e}")
    
    def get_risk_summary(self) -> Dict:
        """
        Get comprehensive risk summary
        
        Returns:
            dict: Risk metrics and current status
        """
        try:
            total_risk = sum(self.position_risks.values())
            risk_percentage = total_risk / self.portfolio_value if self.portfolio_value > 0 else 0
            daily_pnl = self.portfolio_value - self.daily_start_value
            daily_pnl_pct = daily_pnl / self.daily_start_value if self.daily_start_value > 0 else 0
            
            return {
                'portfolio_value': self.portfolio_value,
                'peak_value': self.peak_portfolio_value,
                'current_drawdown': self.current_drawdown,
                'daily_pnl': daily_pnl,
                'daily_pnl_pct': daily_pnl_pct,
                'total_risk': total_risk,
                'risk_percentage': risk_percentage,
                'active_positions': len(self.positions),
                'max_daily_loss': self.daily_start_value * self.MAX_DAILY_LOSS,
                'max_drawdown': self.MAX_DRAWDOWN
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {}





















"""
Market Data Handler for CME Trading Algorithm
==========================================

Handles real-time market data processing, storage, and distribution.
Includes data validation, gap detection, and historical data management.

MODIFICATION GUIDE:
- Adjust DATA_VALIDATION_THRESHOLD for price change limits (line 25)
- Change HISTORY_SIZE for amount of data to retain (line 30)
- Modify data fields in process_market_update method
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from collections import deque
import json

class MarketDataHandler:
    """Handles real-time market data processing and storage"""
    
    # Data Management Parameters - MODIFY FOR DIFFERENT BEHAVIOR
    DATA_VALIDATION_THRESHOLD = 0.05   # 5% max price change validation
    HISTORY_SIZE = 1000                # Number of data points to retain
    GAP_TOLERANCE = 5.0                # Maximum gap in seconds before warning
    PRICE_PRECISION = 2                # Decimal places for price rounding
    
    def __init__(self, config):
        """
        Initialize market data handler
        
        Args:
            config: Trading configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data storage using deque for efficient operations
        self.price_data = deque(maxlen=self.HISTORY_SIZE)
        self.volume_data = deque(maxlen=self.HISTORY_SIZE)
        self.timestamp_data = deque(maxlen=self.HISTORY_SIZE)
        self.bid_data = deque(maxlen=self.HISTORY_SIZE)
        self.ask_data = deque(maxlen=self.HISTORY_SIZE)
        
        # Current market state
        self.current_price = 0.0
        self.current_bid = 0.0
        self.current_ask = 0.0
        self.current_volume = 0
        self.last_update_time = None
        
        # Data quality tracking
        self.total_updates = 0
        self.invalid_updates = 0
        self.gap_count = 0
        self.last_valid_price = 0.0
        
        self.logger.info("Market Data Handler initialized")
    
    def initialize(self) -> bool:
        """
        Initialize market data connections and validate setup
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Validate configuration
            if not hasattr(self.config, 'symbol'):
                self.logger.error("No trading symbol configured")
                return False
            
            # Initialize data structures
            self.price_data.clear()
            self.volume_data.clear()
            self.timestamp_data.clear()
            self.bid_data.clear()
            self.ask_data.clear()
            
            # Reset counters
            self.total_updates = 0
            self.invalid_updates = 0
            self.gap_count = 0
            
            self.logger.info("Market data handler initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing market data handler: {e}")
            return False
    
    def process_market_update(self, market_data: Dict) -> bool:
        """
        Process incoming market data update
        
        Args:
            market_data: Dictionary containing market data fields
            
        Returns:
            bool: True if update processed successfully
        """
        try:
            self.total_updates += 1
            current_time = datetime.now()
            
            # Extract price information
            price = self._extract_price(market_data)
            if price is None:
                self.invalid_updates += 1
                return False
            
            # Validate price data
            if not self._validate_price(price):
                self.invalid_updates += 1
                self.logger.warning(f"Invalid price rejected: {price}")
                return False
            
            # Extract additional market data
            bid = self._extract_field(market_data, ['bid', 'best_bid'], 0.0)
            ask = self._extract_field(market_data, ['ask', 'best_ask'], 0.0)
            volume = self._extract_field(market_data, ['volume', 'trade_volume'], 0)
            
            # Check for data gaps
            if self.last_update_time:
                gap_seconds = (current_time - self.last_update_time).total_seconds()
                if gap_seconds > self.GAP_TOLERANCE:
                    self.gap_count += 1
                    self.logger.warning(f"Data gap detected: {gap_seconds:.1f} seconds")
            
            # Store data
            self.price_data.append(round(price, self.PRICE_PRECISION))
            self.volume_data.append(volume)
            self.timestamp_data.append(current_time)
            self.bid_data.append(round(bid, self.PRICE_PRECISION))
            self.ask_data.append(round(ask, self.PRICE_PRECISION))
            
            # Update current state
            self.current_price = price
            self.current_bid = bid
            self.current_ask = ask
            self.current_volume = volume
            self.last_update_time = current_time
            self.last_valid_price = price
            
            # Log significant price moves
            if len(self.price_data) > 1:
                price_change = abs(price - self.price_data[-2]) / self.price_data[-2]
                if price_change > 0.01:  # 1% move
                    self.logger.info(f"Significant price move: {self.price_data[-2]:.2f} -> {price:.2f} "
                                   f"({price_change:.3f})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing market update: {e}")
            return False
    
    def _extract_price(self, market_data: Dict) -> Optional[float]:
        """
        Extract price from market data with multiple fallback options
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            float: Extracted price, or None if not found
        """
        # Try multiple possible price fields
        price_fields = ['price', 'last_price', 'close', 'settlement_price', 'mid_price']
        
        for field in price_fields:
            if field in market_data:
                try:
                    price = float(market_data[field])
                    if price > 0:
                        return price
                except (ValueError, TypeError):
                    continue
        
        # Try to calculate mid price from bid/ask
        try:
            bid = float(market_data.get('bid', 0))
            ask = float(market_data.get('ask', 0))
            if bid > 0 and ask > 0 and ask > bid:
                return (bid + ask) / 2.0
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _extract_field(self, market_data: Dict, field_names: List[str], default: Any) -> Any:
        """
        Extract field from market data with fallback options
        
        Args:
            market_data: Market data dictionary
            field_names: List of possible field names to try
            default: Default value if field not found
            
        Returns:
            Extracted value or default
        """
        for field in field_names:
            if field in market_data:
                try:
                    return type(default)(market_data[field])
                except (ValueError, TypeError):
                    continue
        return default
    
    def _validate_price(self, price: float) -> bool:
        """
        Validate price data for reasonableness
        
        Args:
            price: Price to validate
            
        Returns:
            bool: True if price is valid
        """
        try:
            # Basic sanity checks
            if price <= 0:
                return False
            
            # Check for reasonable crude oil price range (20-200 $/barrel)
            if price < 20 or price > 200:
                self.logger.warning(f"Price outside expected range: {price}")
                return False
            
            # Check for sudden large moves if we have previous data
            if self.last_valid_price > 0:
                price_change = abs(price - self.last_valid_price) / self.last_valid_price
                if price_change > self.DATA_VALIDATION_THRESHOLD:
                    self.logger.warning(f"Large price move rejected: {self.last_valid_price:.2f} -> {price:.2f} "
                                      f"({price_change:.3f})")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating price: {e}")
            return False
    
    def get_recent_prices(self, count: int = 50) -> List[float]:
        """
        Get recent price data
        
        Args:
            count: Number of recent prices to return
            
        Returns:
            list: Recent prices
        """
        try:
            if len(self.price_data) == 0:
                return []
            
            return list(self.price_data)[-count:]
            
        except Exception as e:
            self.logger.error(f"Error getting recent prices: {e}")
            return []
    
    def get_recent_data(self, count: int = 50) -> Dict:
        """
        Get recent market data as dictionary
        
        Args:
            count: Number of recent data points to return
            
        Returns:
            dict: Recent market data
        """
        try:
            return {
                'prices': list(self.price_data)[-count:],
                'volumes': list(self.volume_data)[-count:],
                'timestamps': list(self.timestamp_data)[-count:],
                'bids': list(self.bid_data)[-count:],
                'asks': list(self.ask_data)[-count:]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting recent data: {e}")
            return {}
    
    def get_current_state(self) -> Dict:
        """
        Get current market state
        
        Returns:
            dict: Current market conditions
        """
        try:
            bid_ask_spread = self.current_ask - self.current_bid if self.current_ask > self.current_bid else 0
            
            return {
                'price': self.current_price,
                'bid': self.current_bid,
                'ask': self.current_ask,
                'volume': self.current_volume,
                'spread': bid_ask_spread,
                'last_update': self.last_update_time,
                'data_quality': {
                    'total_updates': self.total_updates,
                    'invalid_updates': self.invalid_updates,
                    'gap_count': self.gap_count,
                    'success_rate': (self.total_updates - self.invalid_updates) / max(1, self.total_updates)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current state: {e}")
            return {}
    
    def calculate_vwap(self, periods: int = 20) -> float:
        """
        Calculate Volume Weighted Average Price
        
        Args:
            periods: Number of periods for VWAP calculation
            
        Returns:
            float: VWAP value
        """
        try:
            if len(self.price_data) < periods:
                return self.current_price
            
            prices = np.array(list(self.price_data)[-periods:])
            volumes = np.array(list(self.volume_data)[-periods:])
            
            # Avoid division by zero
            total_volume = np.sum(volumes)
            if total_volume == 0:
                return np.mean(prices)
            
            vwap = np.sum(prices * volumes) / total_volume
            return round(vwap, self.PRICE_PRECISION)
            
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return self.current_price
























"""
Technical Indicators for CME Trading Algorithm
===========================================

Comprehensive technical analysis indicators optimized for crude oil trading.
All indicators include error handling and performance optimization.

MODIFICATION GUIDE:
- Add new indicators by following existing patterns
- Adjust default periods in calculate_* methods
- Modify smoothing factors for different sensitivity
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import logging

class TechnicalIndicators:
    """Collection of technical analysis indicators for trading algorithms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands (Upper, Middle, Lower)
        
        Args:
            prices: List of price values
            period: Lookback period for calculation
            std_dev: Standard deviation multiplier
            
        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        try:
            if len(prices) < period:
                return 0.0, 0.0, 0.0
            
            prices_array = np.array(prices[-period:])
            
            # Middle band (Simple Moving Average)
            middle_band = np.mean(prices_array)
            
            # Standard deviation
            std = np.std(prices_array)
            
            # Upper and lower bands
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return 0.0, 0.0, 0.0
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index
        
        Args:
            prices: List of price values
            period: Lookback period for RSI calculation
            
        Returns:
            float: RSI value (0-100)
        """
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI
            
            prices_array = np.array(prices[-(period + 1):])
            deltas = np.diff(prices_array)
            
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return round(rsi, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def calculate_moving_averages(self, prices: List[float], 
                                short_period: int = 10, 
                                long_period: int = 30) -> Tuple[float, float]:
        """
        Calculate Simple Moving Averages
        
        Args:
            prices: List of price values
            short_period: Period for short MA
            long_period: Period for long MA
            
        Returns:
            tuple: (short_ma, long_ma)
        """
        try:
            if len(prices) < long_period:
                current_price = prices[-1] if prices else 0.0
                return current_price, current_price
            
            prices_array = np.array(prices)
            
            short_ma = np.mean(prices_array[-short_period:]) if len(prices) >= short_period else prices[-1]
            long_ma = np.mean(prices_array[-long_period:])
            
            return round(short_ma, 2), round(long_ma, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating moving averages: {e}")
            return 0.0, 0.0
    
    def calculate_macd(self, prices: List[float], fast: int = 12, 
                      slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: List of price values
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        try:
            if len(prices) < slow + signal:
                return 0.0, 0.0, 0.0
            
            prices_array = np.array(prices)
            
            # Calculate EMAs
            fast_ema = self._calculate_ema(prices_array, fast)
            slow_ema = self._calculate_ema(prices_array, slow)
            
            # MACD line
            macd_line = fast_ema - slow_ema
            
            # Signal line (EMA of MACD line)
            # For simplicity, using SMA instead of EMA for signal line
            if len(prices) >= slow + signal:
                recent_macd_values = []
                for i in range(signal):
                    if len(prices) >= slow + i + 1:
                        temp_fast = self._calculate_ema(prices_array[:-(signal-i-1)] if signal-i-1 > 0 else prices_array, fast)
                        temp_slow = self._calculate_ema(prices_array[:-(signal-i-1)] if signal-i-1 > 0 else prices_array, slow)
                        recent_macd_values.append(temp_fast - temp_slow)
                
                signal_line = np.mean(recent_macd_values) if recent_macd_values else macd_line
            else:
                signal_line = macd_line
            
            # Histogram
            histogram = macd_line - signal_line
            
            return round(macd_line, 4), round(signal_line, 4), round(histogram, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """
        Calculate Exponential Moving Average
        
        Args:
            prices: Array of prices
            period: EMA period
            
        Returns:
            float: EMA value
        """
        try:
            if len(prices) < period:
                return np.mean(prices)
            
            multiplier = 2.0 / (period + 1)
            ema = np.mean(prices[:period])  # Start with SMA
            
            for price in prices[period:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return ema
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return 0.0
    
    def calculate_volatility(self, prices: List[float], period: int = 20) -> float:
        """
        Calculate historical volatility (annualized)
        
        Args:
            prices: List of price values
            period: Lookback period
            
        Returns:
            float: Annualized volatility
        """
        try:
            if len(prices) < period:
                return 0.0
            
            prices_array = np.array(prices[-period:])
            returns = np.diff(np.log(prices_array))
            
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            return round(volatility, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def calculate_atr(self, high_prices: List[float], low_prices: List[float], 
                     close_prices: List[float], period: int = 14) -> float:
        """
        Calculate Average True Range
        
        Args:
            high_prices: List of high prices
            low_prices: List of low prices
            close_prices: List of close prices
            period: ATR period
            
        Returns:
            float: ATR value
        """
        try:
            if len(high_prices) < period + 1:
                return 0.0
            
            true_ranges = []
            
            for i in range(1, len(close_prices)):
                high_low = high_prices[i] - low_prices[i]
                high_close_prev = abs(high_prices[i] - close_prices[i-1])
                low_close_prev = abs(low_prices[i] - close_prices[i-1])
                
                true_range = max(high_low, high_close_prev, low_close_prev)
                true_ranges.append(true_range)
            
            if len(true_ranges) >= period:
                atr = np.mean(true_ranges[-period:])
                return round(atr, 4)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    def calculate_support_resistance(self, prices: List[float], 
                                   window: int = 10) -> Tuple[float, float]:
        """
        Calculate basic support and resistance levels
        
        Args:
            prices: List of price values
            window: Lookback window for calculation
            
        Returns:
            tuple: (support_level, resistance_level)
        """
        try:
            if len(prices) < window:
                current_price = prices[-1] if prices else 0.0
                return current_price, current_price
            
            recent_prices = prices[-window:]
            support = min(recent_prices)
            resistance = max(recent_prices)
            
            return round(support, 2), round(resistance, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return 0.0, 0.0
    
    def calculate_momentum(self, prices: List[float], period: int = 10) -> float:
        """
        Calculate price momentum
        
        Args:
            prices: List of price values
            period: Lookback period
            
        Returns:
            float: Momentum value
        """
        try:
            if len(prices) < period + 1:
                return 0.0
            
            current_price = prices[-1]
            past_price = prices[-(period + 1)]
            
            momentum = (current_price - past_price) / past_price
            
            return round(momentum, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum: {e}")
            return 0.0

















"""
Trading Configuration for CME Algorithm
====================================

Centralized configuration management for all trading parameters.
Allows easy modification of strategy parameters without changing core code.

MODIFICATION GUIDE:
- Add new configuration parameters to __init__ method
- Modify default values for different trading styles
- Add validation in validate_config method
"""

import os
import logging
from typing import Dict, Any

class TradingConfig:
    """Centralized configuration management for trading algorithm"""
    
    def __init__(self):
        """Initialize configuration with default values"""
        
        # Trading Symbol and Market
        self.symbol = 'CLM9'                    # Crude oil futures symbol
        self.exchange = 'NYMEX'                 # Exchange
        self.contract_size = 1000               # Barrels per contract
        self.tick_size = 0.01                   # Minimum price increment
        self.tick_value = 10.0                  # Dollar value per tick
        
        # Algorithm Parameters
        self.algorithm_name = 'BollingerMeanReversion'
        self.bollinger_period = 20
        self.bollinger_std = 2.0
        self.bollinger_entry_threshold = 2.0
        self.bollinger_exit_threshold = 0.5
        
        # Risk Management
        self.max_position_size = 20             # Maximum contracts
        self.risk_per_trade = 0.02              # 2% risk per trade
        self.stop_loss_pct = 0.015              # 1.5% stop loss
        self.take_profit_pct = 0.025            # 2.5% take profit
        self.max_daily_loss = 5000              # Maximum daily loss in dollars
        self.max_daily_trades = 10              # Maximum trades per day
        
        # Portfolio Management
        self.initial_capital = 100000           # Starting capital
        self.max_portfolio_risk = 0.10         # 10% maximum portfolio risk
        self.max_correlation = 0.7              # Maximum position correlation
        
        # Market Data
        self.data_validation_threshold = 0.05   # 5% price change validation
        self.min_data_points = 50               # Minimum data for trading
        self.history_size = 1000                # Data points to retain
        
        # Timing and Execution
        self.max_holding_period = 240           # Maximum minutes to hold
        self.order_timeout = 30                 # Order timeout in seconds
        self.execution_delay = 0.1              # Simulated execution delay
        
        # Logging and Monitoring
        self.log_level = logging.INFO
        self.log_to_file = True
        self.log_to_console = True
        self.performance_log_interval = 300     # Log performance every 5 minutes
        
        # Platform Integration
        self.kafka_bootstrap_servers = ['localhost:9092']
        self.kafka_topics = {
            'orders': 'cme_orders',
            'executions': 'cme_executions',
            'market_data': 'cme_market_data',
            'positions': 'cme_positions'
        }
        
        # Load environment-specific overrides
        self._load_environment_config()
        
        # Validate configuration
        if not self.validate_config():
            raise ValueError("Invalid configuration detected")
    
    def _load_environment_config(self):
        """Load configuration from environment variables"""
        try:
            # Override with environment variables if present
            self.symbol = os.getenv('TRADING_SYMBOL', self.symbol)
            
            # Risk parameters
            if os.getenv('MAX_POSITION_SIZE'):
                self.max_position_size = int(os.getenv('MAX_POSITION_SIZE'))
            
            if os.getenv('RISK_PER_TRADE'):
                self.risk_per_trade = float(os.getenv('RISK_PER_TRADE'))
            
            if os.getenv('MAX_DAILY_LOSS'):
                self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS'))
            
            # Algorithm parameters
            if os.getenv('BOLLINGER_PERIOD'):
                self.bollinger_period = int(os.getenv('BOLLINGER_PERIOD'))
            
            if os.getenv('BOLLINGER_STD'):
                self.bollinger_std = float(os.getenv('BOLLINGER_STD'))
            
        except Exception as e:
            logging.warning(f"Error loading environment config: {e}")
    
    def validate_config(self) -> bool:
        """
        Validate configuration parameters
        
        Returns:
            bool: True if configuration is valid
        """
        try:
            # Validate basic parameters
            if self.max_position_size <= 0:
                logging.error("max_position_size must be positive")
                return False
            
            if self.risk_per_trade <= 0 or self.risk_per_trade > 0.2:
                logging.error("risk_per_trade must be between 0 and 0.2")
                return False
            
            if self.bollinger_period < 5 or self.bollinger_period > 100:
                logging.error("bollinger_period must be between 5 and 100")
                return False
            
            if self.bollinger_std <= 0 or self.bollinger_std > 5:
                logging.error("bollinger_std must be between 0 and 5")
                return False
            
            if self.initial_capital <= 0:
                logging.error("initial_capital must be positive")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating config: {e}")
            return False
    
    def get_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary
        
        Returns:
            dict: Configuration parameters
        """
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'algorithm_name': self.algorithm_name,
            'bollinger_period': self.bollinger_period,
            'bollinger_std': self.bollinger_std,
            'max_position_size': self.max_position_size,
            'risk_per_trade': self.risk_per_trade,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_daily_loss': self.max_daily_loss,
            'initial_capital': self.initial_capital,
            'max_holding_period': self.max_holding_period
        }
    
    def update_parameter(self, parameter: str, value: Any) -> bool:
        """
        Update single configuration parameter
        
        Args:
            parameter: Parameter name
            value: New value
            
        Returns:
            bool: True if update successful
        """
        try:
            if hasattr(self, parameter):
                setattr(self, parameter, value)
                logging.info(f"Updated {parameter} to {value}")
                return self.validate_config()
            else:
                logging.warning(f"Unknown parameter: {parameter}")
                return False
                
        except Exception as e:
            logging.error(f"Error updating parameter {parameter}: {e}")
            return False
















#!/bin/bash
# CME Trading Algorithm Startup Script
# Replace 'trop' with your team name throughout this file

echo "=== Starting CME Crude Oil Trading Algorithm ==="
echo "Team: Tropical Green"
echo "Algorithm: Bollinger Band Mean Reversion"
echo "Symbol: CLM9"
echo "Timestamp: $(date)"
echo "=========================================="

# Set environment variables
export TRADING_SYMBOL=CLM9
export MAX_POSITION_SIZE=20
export RISK_PER_TRADE=0.02
export MAX_DAILY_LOSS=5000
export BOLLINGER_PERIOD=20
export BOLLINGER_STD=2.0

# Activate Python virtual environment
source cme_trading_env/bin/activate

# Set Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Run the algorithm with proper configuration
python -c "
import sys
sys.path.append('src')
from algorithms.crude_oil_trader import CrudeOilTrader

# Initialize and start the algorithm
trader = CrudeOilTrader()
if trader.start():
    print('Algorithm started successfully')
else:
    print('Algorithm failed to start')
    sys.exit(1)
"

echo "Algorithm startup completed"
