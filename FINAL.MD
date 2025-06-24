import traceback
import sys
import os
import dataclasses
from datetime import import datetime, timedelta
from pydoc import locate
from Preferences import Preferences
from dacite import from_dict
from json import loads, dumps
from gensrc.MDMsg import MDMsg
from gensrc.MDBookReqMsg import MDBookReqMsg
from gensrc.CommandMsg import CommandMsg
from gensrc.OrderMsg import OrderMsg
from gensrc.CXRMsg import CXRMsg
from gensrc.CXLMsg import CXLMsg
from gensrc.OrderStatusMsg import OrderStatusMsg
from gensrc.ExecutionMsg import ExecutionMsg
from gensrc.PortfolioMsg import PortfolioMsg
from gensrc.PortfolioBootReqMsg import PortfolioBootReqMsg
from gensrc.ConfigRespMsg import ConfigRespMsg
from MDBook import MDBook
from OrderBook import OrderBook
import OrderBook as OB
import random
import logging
import logging.config
from WebTrader import WebTrader
from kafka import KafkaConsumer, KafkaProducer, TopicPartition, OffsetAndMetadata
from concurrent.futures import ThreadPoolExecutor
import faulthandler
import time

# Enhanced Trading Algorithm Imports
import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import json
import math
from statistics import mean, stdev

"""
ENHANCED CME WINNING TRADING ALGORITHM
=====================================

Competition-Optimized Bollinger Band Mean Reversion Strategy
Integrated with existing CME platform infrastructure

Key Features:
- Advanced Bollinger Band mean reversion with multi-factor confirmation
- Dynamic position sizing based on volatility and market conditions
- Comprehensive risk management preventing catastrophic losses
- Market regime detection for strategy adaptation
- Real-time performance monitoring and emergency controls

Target: Maximum Sharpe Ratio while minimizing drawdown risk
"""

PREFS_FILE = 'cfg/matchalgo-py.properties'

logging.config.fileConfig(fname='cfg/logging.cfg', disable_existing_loggers=False)
log = logging.getLogger('cc.applogger')
tlog = logging.getLogger('cc.trader')

# Message Types
MD='M'
NEW_ORDER='O'
ORDER_CXL='C'
ORDER_CXR='R'
EXECUTION='E'
PORTFOLIO='P'
ORDSTATUS='S'
MD_BOOT_REQ='B'
PORTFOLIO_BOOT_REQ='A'
COMMAND='Z'
CONFIG_RESP='X'

# ============================================================================
# ENHANCED TRADING ALGORITHM CLASS
# ============================================================================

class EnhancedBollingerBandTrader:
    """
    Competition-optimized Bollinger Band mean reversion trader
    Designed for maximum performance with minimal risk
    """
    
    def __init__(self):
        """Initialize the enhanced trading algorithm"""
        
        # Competition-Optimized Parameters
        self.BB_PERIOD = 20                    # Bollinger Band lookback period
        self.BB_STD_DEV = 2.0                  # Standard deviation multiplier
        self.ENTRY_THRESHOLD = 0.015           # 1.5% breach for entry signal
        self.EXIT_THRESHOLD = 0.005            # 0.5% return to middle for exit
        
        # Risk Management - Critical for Competition Survival
        self.MAX_POSITION_SIZE = 8             # Maximum contracts per position
        self.RISK_PER_TRADE = 0.02             # 2% risk per trade
        self.STOP_LOSS_PCT = 0.025             # 2.5% stop loss
        self.TAKE_PROFIT_PCT = 0.04            # 4% take profit
        self.MAX_DAILY_LOSS = 0.06             # 6% maximum daily loss
        self.MAX_DRAWDOWN = 0.10               # 10% maximum drawdown
        self.MAX_TRADES_PER_DAY = 12           # Maximum trades per day
        self.MIN_TRADE_INTERVAL = 60           # Minimum seconds between trades
        
        # Market Data Storage
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        self.timestamp_history = deque(maxlen=100)
        
        # Algorithm State
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.current_price = 0.0
        self.portfolio_value = 100000          # Starting capital
        self.peak_portfolio_value = 100000
        self.daily_start_value = 100000
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Performance Tracking
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        
        # Technical Indicators
        self.bb_upper = 0.0
        self.bb_middle = 0.0
        self.bb_lower = 0.0
        self.current_volatility = 0.0
        self.rsi = 50.0
        self.momentum = 0.0
        
        # Risk Control State
        self.emergency_mode = False
        self.consecutive_losses = 0
        self.last_signal_time = None
        
        # Platform Integration
        self.container = None
        self.is_running = False
        
        log.info("=== ENHANCED BOLLINGER BAND TRADER INITIALIZED ===")
        log.info(f"Strategy: Competition-Optimized Mean Reversion")
        log.info(f"Risk Management: {self.RISK_PER_TRADE*100}% per trade")
        log.info(f"Max Position: {self.MAX_POSITION_SIZE} contracts")
        log.info(f"Daily Loss Limit: {self.MAX_DAILY_LOSS*100}%")
    
    def setContainer(self, container):
        """Set the container reference for order execution"""
        self.container = container
        log.info("Container reference set for trader")
    
    def isRunning(self) -> bool:
        """Check if trader is currently running"""
        return self.is_running
    
    def start(self):
        """Start the trading algorithm"""
        try:
            self.is_running = True
            self.daily_start_value = self.portfolio_value
            self.trades_today = 0
            self.emergency_mode = False
            self.consecutive_losses = 0
            
            log.info("=== ENHANCED TRADER STARTED ===")
            log.info(f"Starting portfolio value: ${self.portfolio_value:,.2f}")
            return True
            
        except Exception as e:
            log.error(f"Error starting trader: {e}")
            return False
    
    def stop(self):
        """Stop the trading algorithm"""
        try:
            self.is_running = False
            
            # Close any open positions in emergency
            if self.current_position != 0:
                log.warning(f"Closing position on trader stop: {self.current_position}")
                self._emergency_close_position()
            
            # Log final performance
            final_pnl = self.portfolio_value - self.daily_start_value
            log.info("=== ENHANCED TRADER STOPPED ===")
            log.info(f"Final P&L: ${final_pnl:,.2f}")
            log.info(f"Total Trades: {self.winning_trades + self.losing_trades}")
            log.info(f"Win Rate: {self._calculate_win_rate():.1f}%")
            
        except Exception as e:
            log.error(f"Error stopping trader: {e}")
    
    def onMDMsg(self, mdbook, mdmsg):
        """Process market data updates - main algorithm entry point"""
        try:
            if not self.is_running:
                return
            
            # Extract price from market data
            current_price = self._extract_price_from_mdmsg(mdmsg)
            if current_price is None or current_price <= 0:
                return
            
            # Validate and store market data
            if not self._validate_and_store_price(current_price):
                return
            
            # Need sufficient data for analysis
            if len(self.price_history) < self.BB_PERIOD + 5:
                return
            
            # Calculate technical indicators
            self._calculate_indicators()
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            # Check emergency conditions first
            if self._check_emergency_conditions():
                return
            
            # Generate and execute trading signals
            signal = self._generate_trading_signal()
            if signal != 'HOLD':
                self._execute_signal(signal)
                
        except Exception as e:
            log.error(f"Error in onMDMsg: {e}")
            traceback.print_exc()
    
    def onExecutionMsg(self, orderbook, execution):
        """Handle execution messages"""
        try:
            if not self.is_running:
                return
            
            # Update position tracking based on execution
            if hasattr(execution, 'qty') and hasattr(execution, 'isBuy'):
                qty = execution.qty
                if not execution.isBuy:  # Sell order
                    qty = -qty
                
                # Update our position tracking
                if self.current_position == 0:  # Opening position
                    self.current_position = qty
                    self.entry_price = execution.price if hasattr(execution, 'price') else self.current_price
                    self.entry_time = datetime.now()
                    log.info(f"Position opened: {self.current_position} @ {self.entry_price:.2f}")
                else:  # Closing position
                    pnl = self._calculate_trade_pnl(execution.price if hasattr(execution, 'price') else self.current_price)
                    self._record_trade_result(pnl)
                    self._reset_position_state()
                    log.info(f"Position closed: P&L ${pnl:.2f}")
                
        except Exception as e:
            log.error(f"Error in onExecutionMsg: {e}")
    
    def onPortfolioMsg(self, portfolio):
        """Handle portfolio updates"""
        try:
            if not self.is_running:
                return
            
            # Update portfolio value from platform
            if hasattr(portfolio, 'totalValue'):
                self.portfolio_value = portfolio.totalValue
            
            # Update realized P&L
            if hasattr(portfolio, 'realizedPnL'):
                self.realized_pnl = portfolio.realizedPnL
                
        except Exception as e:
            log.error(f"Error in onPortfolioMsg: {e}")
    
    def onTakeAction(self):
        """Periodic action callback - used for monitoring and maintenance"""
        try:
            if not self.is_running:
                return
            
            # Log performance summary every 5 minutes
            if self._should_log_performance():
                self._log_performance_summary()
            
            # Check for emergency conditions
            self._check_emergency_conditions()
            
            # Monitor position time limits
            self._check_position_time_limits()
            
        except Exception as e:
            log.error(f"Error in onTakeAction: {e}")
    
    def _extract_price_from_mdmsg(self, mdmsg) -> Optional[float]:
        """Extract price from market data message"""
        try:
            # Try multiple price fields
            for field in ['price', 'last', 'close', 'mid']:
                if hasattr(mdmsg, field):
                    price = getattr(mdmsg, field)
                    if price and price > 0:
                        return float(price)
            
            # Try to calculate mid price from bid/ask
            if hasattr(mdmsg, 'bid') and hasattr(mdmsg, 'ask'):
                bid = getattr(mdmsg, 'bid', 0)
                ask = getattr(mdmsg, 'ask', 0)
                if bid > 0 and ask > 0 and ask > bid:
                    return (bid + ask) / 2.0
            
            return None
            
        except Exception as e:
            log.error(f"Error extracting price: {e}")
            return None
    
    def _validate_and_store_price(self, price: float) -> bool:
        """Validate and store incoming price data"""
        try:
            # Basic validation
            if price <= 0 or price > 500:  # Reasonable crude oil range
                return False
            
            # Check for extreme moves
            if len(self.price_history) > 0:
                last_price = self.price_history[-1]
                price_change = abs(price - last_price) / last_price
                if price_change > 0.10:  # 10% circuit breaker
                    log.warning(f"Extreme price move detected: {price_change:.3f}")
                    return False
            
            # Store the price
            self.price_history.append(price)
            self.timestamp_history.append(datetime.now())
            self.current_price = price
            
            return True
            
        except Exception as e:
            log.error(f"Error validating price: {e}")
            return False
    
    def _calculate_indicators(self):
        """Calculate all technical indicators"""
        try:
            prices = np.array(list(self.price_history))
            
            # Bollinger Bands
            if len(prices) >= self.BB_PERIOD:
                recent_prices = prices[-self.BB_PERIOD:]
                self.bb_middle = np.mean(recent_prices)
                bb_std = np.std(recent_prices)
                self.bb_upper = self.bb_middle + (self.BB_STD_DEV * bb_std)
                self.bb_lower = self.bb_middle - (self.BB_STD_DEV * bb_std)
            
            # Volatility
            if len(prices) >= 20:
                returns = np.diff(np.log(prices[-20:]))
                self.current_volatility = np.std(returns) * np.sqrt(252)
            
            # RSI
            if len(prices) >= 15:
                self.rsi = self._calculate_rsi(prices[-15:])
            
            # Momentum
            if len(prices) >= 11:
                self.momentum = (prices[-1] - prices[-11]) / prices[-11]
                
        except Exception as e:
            log.error(f"Error calculating indicators: {e}")
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return max(0, min(100, rsi))
            
        except Exception as e:
            log.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _generate_trading_signal(self) -> str:
        """Generate enhanced trading signal"""
        try:
            # No trading in emergency mode
            if self.emergency_mode:
                return 'HOLD'
            
            # Check daily trade limit
            if self.trades_today >= self.MAX_TRADES_PER_DAY:
                return 'HOLD'
            
            # Rate limiting
            if (self.last_trade_time and 
                (datetime.now() - self.last_trade_time).total_seconds() < self.MIN_TRADE_INTERVAL):
                return 'HOLD'
            
            # If we have a position, check for exit signals
            if self.current_position != 0:
                return self._generate_exit_signal()
            
            # Generate entry signals
            return self._generate_entry_signal()
            
        except Exception as e:
            log.error(f"Error generating signal: {e}")
            return 'HOLD'
    
    def _generate_entry_signal(self) -> str:
        """Generate entry signal with enhanced filtering"""
        try:
            current_price = self.current_price
            
            # Calculate Bollinger Band breaches
            upper_breach = (current_price - self.bb_upper) / self.bb_upper
            lower_breach = (self.bb_lower - current_price) / self.bb_lower
            
            # Adjust thresholds based on volatility
            entry_threshold = self.ENTRY_THRESHOLD
            if self.current_volatility > 0.3:  # High volatility
                entry_threshold *= 1.5
            elif self.current_volatility < 0.15:  # Low volatility
                entry_threshold *= 0.8
            
            # SELL Signal (Short) - Price above upper Bollinger Band
            if upper_breach >= entry_threshold:
                # Additional confirmation
                if (self.rsi > 65 and  # Overbought condition
                    self.momentum > 0.005):  # Positive momentum
                    
                    log.info(f"SELL SIGNAL: Price {current_price:.2f}, BB Upper {self.bb_upper:.2f}, RSI {self.rsi:.1f}")
                    return 'SELL'
            
            # BUY Signal (Long) - Price below lower Bollinger Band
            elif lower_breach >= entry_threshold:
                # Additional confirmation
                if (self.rsi < 35 and  # Oversold condition
                    self.momentum < -0.005):  # Negative momentum
                    
                    log.info(f"BUY SIGNAL: Price {current_price:.2f}, BB Lower {self.bb_lower:.2f}, RSI {self.rsi:.1f}")
                    return 'BUY'
            
            return 'HOLD'
            
        except Exception as e:
            log.error(f"Error generating entry signal: {e}")
            return 'HOLD'
    
    def _generate_exit_signal(self) -> str:
        """Generate exit signal for existing position"""
        try:
            if self.entry_price == 0:
                return 'HOLD'
            
            current_price = self.current_price
            
            # Calculate P&L
            if self.current_position > 0:  # Long position
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                position_type = 'LONG'
            else:  # Short position
                pnl_pct = (self.entry_price - current_price) / self.entry_price
                position_type = 'SHORT'
            
            # Stop loss
            if pnl_pct <= -self.STOP_LOSS_PCT:
                log.info(f"STOP LOSS: {position_type}, P&L: {pnl_pct:.3f}")
                return f'CLOSE_{position_type}'
            
            # Take profit
            if pnl_pct >= self.TAKE_PROFIT_PCT:
                log.info(f"TAKE PROFIT: {position_type}, P&L: {pnl_pct:.3f}")
                return f'CLOSE_{position_type}'
            
            # Mean reversion exit
            distance_to_middle = abs(current_price - self.bb_middle) / self.bb_middle
            if distance_to_middle <= self.EXIT_THRESHOLD:
                log.info(f"MEAN REVERSION EXIT: {position_type}")
                return f'CLOSE_{position_type}'
            
            # Time-based exit (2 hours maximum)
            if (self.entry_time and 
                (datetime.now() - self.entry_time).total_seconds() > 7200):
                log.info(f"TIME EXIT: {position_type}")
                return f'CLOSE_{position_type}'
            
            return 'HOLD'
            
        except Exception as e:
            log.error(f"Error generating exit signal: {e}")
            return 'HOLD'
    
    def _execute_signal(self, signal: str):
        """Execute trading signal through platform"""
        try:
            if not self.container:
                log.error("No container reference for order execution")
                return
            
            position_size = self._calculate_position_size()
            current_price_cents = int(self.current_price * 100)  # Convert to cents
            
            if signal == 'BUY':
                self.container.newOrder(current_price_cents, position_size, True)
                log.info(f"BUY ORDER: {position_size} @ {self.current_price:.2f}")
                
            elif signal == 'SELL':
                self.container.newOrder(current_price_cents, position_size, False)
                log.info(f"SELL ORDER: {position_size} @ {self.current_price:.2f}")
                
            elif signal in ['CLOSE_LONG', 'CLOSE_SHORT']:
                # Close position with opposite order
                is_buy = (signal == 'CLOSE_SHORT')
                close_size = abs(self.current_position)
                self.container.newOrder(current_price_cents, close_size, is_buy)
                log.info(f"CLOSE ORDER: {close_size} @ {self.current_price:.2f}")
            
            self.last_trade_time = datetime.now()
            self.trades_today += 1
            
        except Exception as e:
            log.error(f"Error executing signal {signal}: {e}")
    
    def _calculate_position_size(self) -> int:
        """Calculate optimal position size"""
        try:
            # Base size
            base_size = 3
            
            # Adjust for volatility
            if self.current_volatility > 0:
                vol_adjustment = min(1.0, 0.25 / max(0.10, self.current_volatility))
                base_size = max(1, int(base_size * vol_adjustment))
            
            # Reduce size after losses
            if self.consecutive_losses > 2:
                base_size = max(1, base_size - 1)
            
            # Apply limits
            return min(base_size, self.MAX_POSITION_SIZE)
            
        except Exception as e:
            log.error(f"Error calculating position size: {e}")
            return 1
    
    def _update_portfolio_metrics(self):
        """Update portfolio performance metrics"""
        try:
            # Calculate unrealized P&L
            if self.current_position != 0 and self.entry_price > 0:
                if self.current_position > 0:
                    self.unrealized_pnl = (self.current_price - self.entry_price) * self.current_position * 1000
                else:
                    self.unrealized_pnl = (self.entry_price - self.current_price) * abs(self.current_position) * 1000
            else:
                self.unrealized_pnl = 0.0
            
            # Update portfolio value
            total_pnl = self.realized_pnl + self.unrealized_pnl
            self.portfolio_value = self.daily_start_value + total_pnl
            
            # Update peak value
            if self.portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.portfolio_value
            
            # Update daily P&L
            self.daily_pnl = total_pnl
            
        except Exception as e:
            log.error(f"Error updating portfolio metrics: {e}")
    
    def _check_emergency_conditions(self) -> bool:
        """Check for emergency conditions"""
        try:
            # Daily loss limit
            daily_loss_pct = -self.daily_pnl / self.daily_start_value
            if daily_loss_pct > self.MAX_DAILY_LOSS:
                if not self.emergency_mode:
                    log.error(f"EMERGENCY: Daily loss limit exceeded ({daily_loss_pct:.3f})")
                    self.emergency_mode = True
                    self._emergency_close_position()
                return True
            
            # Maximum drawdown
            drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            if drawdown > self.MAX_DRAWDOWN:
                if not self.emergency_mode:
                    log.error(f"EMERGENCY: Max drawdown exceeded ({drawdown:.3f})")
                    self.emergency_mode = True
                    self._emergency_close_position()
                return True
            
            # Too many consecutive losses
            if self.consecutive_losses > 4:
                if not self.emergency_mode:
                    log.warning(f"EMERGENCY: Too many consecutive losses ({self.consecutive_losses})")
                    self.emergency_mode = True
                return True
            
            return False
            
        except Exception as e:
            log.error(f"Error checking emergency conditions: {e}")
            return True
    
    def _emergency_close_position(self):
        """Emergency position closure"""
        try:
            if self.current_position != 0 and self.container:
                is_buy = (self.current_position < 0)  # Buy to close short, sell to close long
                close_size = abs(self.current_position)
                price_cents = int(self.current_price * 100)
                
                self.container.newOrder(price_cents, close_size, is_buy)
                log.error(f"EMERGENCY CLOSE: {close_size} @ {self.current_price:.2f}")
                
        except Exception as e:
            log.error(f"Error in emergency close: {e}")
    
    def _calculate_trade_pnl(self, exit_price: float) -> float:
        """Calculate P&L for completed trade"""
        try:
            if self.current_position > 0:  # Long position
                pnl = (exit_price - self.entry_price) * self.current_position * 1000
            else:  # Short position
                pnl = (self.entry_price - exit_price) * abs(self.current_position) * 1000
            
            return pnl
            
        except Exception as e:
            log.error(f"Error calculating trade P&L: {e}")
            return 0.0
    
    def _record_trade_result(self, pnl: float):
        """Record trade result for performance tracking"""
        try:
            self.realized_pnl += pnl
            
            if pnl > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
            else:
                self.losing_trades += 1
                self.consecutive_losses += 1
            
            log.info(f"Trade completed: P&L ${pnl:.2f}, Win rate: {self._calculate_win_rate():.1f}%")
            
        except Exception as e:
            log.error(f"Error recording trade result: {e}")
    
    def _reset_position_state(self):
        """Reset position tracking state"""
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.unrealized_pnl = 0.0
    
    def _calculate_win_rate(self) -> float:
        """Calculate current win rate"""
        total_trades = self.winning_trades + self.losing_trades
        return (self.winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    def _should_log_performance(self) -> bool:
        """Check if should log performance (every 5 minutes)"""
        if not hasattr(self, '_last_performance_log'):
            self._last_performance_log = datetime.now()
            return True
        
        if (datetime.now() - self._last_performance_log).total_seconds() > 300:
            self._last_performance_log = datetime.now()
            return True
        
        return False
    
    def _log_performance_summary(self):
        """Log comprehensive performance summary"""
        try:
            win_rate = self._calculate_win_rate()
            drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            
            log.info("=== PERFORMANCE SUMMARY ===")
            log.info(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            log.info(f"Daily P&L: ${self.daily_pnl:,.2f} ({self.daily_pnl/self.daily_start_value*100:.2f}%)")
            log.info(f"Realized P&L: ${self.realized_pnl:,.2f}")
            log.info(f"Unrealized P&L: ${self.unrealized_pnl:,.2f}")
            log.info(f"Current Position: {self.current_position}")
            log.info(f"Trades Today: {self.trades_today}")
            log.info(f"Win Rate: {win_rate:.1f}% ({self.winning_trades}W/{self.losing_trades}L)")
            log.info(f"Current Drawdown: {drawdown:.3f}")
            log.info(f"Emergency Mode: {self.emergency_mode}")
            log.info(f"Current Price: {self.current_price:.2f}")
            log.info(f"BB Bands: {self.bb_lower:.2f} | {self.bb_middle:.2f} | {self.bb_upper:.2f}")
            log.info(f"RSI: {self.rsi:.1f}, Volatility: {self.current_volatility:.3f}")
            log.info("========================")
            
        except Exception as e:
            log.error(f"Error logging performance: {e}")
    
    def _check_position_time_limits(self):
        """Check if position has been held too long"""
        try:
            if (self.current_position != 0 and self.entry_time and 
                (datetime.now() - self.entry_time).total_seconds() > 7200):  # 2 hours
                
                log.warning(f"Position held too long, forcing close")
                self._emergency_close_position()
                
        except Exception as e:
            log.error(f"Error checking position time limits: {e}")

# ============================================================================
# MAIN PLATFORM INTEGRATION (UNCHANGED)
# ============================================================================

log.info(f'Args: {sys.argv}')
PREFS_FILE = sys.argv.pop()
log.info(f'Using properties file: {PREFS_FILE}')
prefs = Preferences(PREFS_FILE)
sys.argv.append(prefs.get_management_port())
log.info(f'Args updated: {sys.argv}')
containerID = prefs.get_container_id()
kafkaclientid = prefs.get_kafka_clientid()
brokerurls = prefs.get_brokers().split(',')
kafka_connect_at='latest'

log.info(f'Using Kafka Broker URLS {brokerurls}')

# Init the market data and order books
mdbook = MDBook()
orderbook = OrderBook()
portfolio: PortfolioMsg = None

# Check if client is being started after a shutdown by serve end
if os.path.exists('shutdowncmd.lck'):
    log.info("Removing shutdown lock ...")
    os.remove('shutdowncmd.lck')
else:
    # Advance the counter to accept first sutdown message
    prefs.onCMDMsg()

ORDERS_TOPIC_NAME = 'orders'

mdtopic = KafkaConsumer(
    'marketdata',
    bootstrap_servers=brokerurls,
    group_id=containerID,
    auto_offset_reset=kafka_connect_at,
    enable_auto_commit=True,
    auto_commit_interval_ms=500
)
postopic = KafkaConsumer(
    'positions',
    bootstrap_servers=brokerurls,
    group_id=containerID,
    auto_offset_reset=kafka_connect_at,
    enable_auto_commit=True,
    auto_commit_interval_ms=500
)
exectopic = KafkaConsumer(
    'executions',
    bootstrap_servers=brokerurls,
    group_id=containerID,
    auto_offset_reset=kafka_connect_at,
    enable_auto_commit=True,
    auto_commit_interval_ms=500
)
ordtopic = KafkaProducer(
    bootstrap_servers=brokerurls,
)

# Initialize the enhanced trader
trader = EnhancedBollingerBandTrader()

# Thread pool for listening to Kafka topics and sending messages
ordPool = ThreadPoolExecutor(max_workers=1)
execPool = ThreadPoolExecutor(max_workers=1)
posPool = ThreadPoolExecutor(max_workers=1)
mdPool = ThreadPoolExecutor(max_workers=1)

bkTaskPool = ThreadPoolExecutor(max_workers=1)
algoActionPool = ThreadPoolExecutor(max_workers=1)

class KafkaPyContainerService():
    orderID = 1
    traderThrottle = 1
    firstRun = True
    stopCalled = False

    def on_start(self):
        log.info("Client Container Starting ...")
        self.bootstrapped = False
        trader.setContainer(self)
        
        # send configuration to the server
        self.traderThrottle = float(prefs.get_throttle())/1000
        log.info(f"Using Trader Throttle: {self.traderThrottle}")

        # Start the Kafka Readers
        log.info("Starting kafka consumers ...")
        mdPool.submit(self.topicProcessor, 'marketdata', mdtopic)
        execPool.submit(self.topicProcessor, 'executions', exectopic)
        posPool.submit(self.topicProcessor, 'positions', postopic)
        log.info("Kafka consumers started, container ready")

        # Sending config of the container
        resp: ConfigRespMsg = prefs.build_config_response(trader.isRunning())
        msg = CONFIG_RESP + dumps(dataclasses.asdict(resp))
        log.info(f'Sending client config: {msg}')
        self.sendOrderMsg(msg)

        if prefs.get_start_on_boot() == True:
            log.info(f"Starting algo on boot ...")
            trader.start()
        else:
            log.info(f"Start on boot is not enabled, waiting for market open command from engine ...")
            
        # Start the periodic order purge task
        bkTaskPool.submit(self.backgroundTask)
        # Start the periodic action task, this triggers the contained algo to take action
        algoActionPool.submit(self.action_background_task)

    # Consumes messages out of a Kafka Topic indefinitely
    def topicProcessor(self, topic, consumer):
        try:
            # This is an infinite loop on the KafkaConsumer
            for consumerRecord in consumer:
                message = consumerRecord.value
                tlog.info(f"[{topic}]: << {message}")

                if topic == 'marketdata':
                    self.process_md(message)
                elif topic == 'executions':
                    self.process_executions(message)
                elif topic == 'positions':
                    self.process_positions(message)

                if self.bootstrapped == False:
                    # Positions is the last thing to be bootstrapped and the container will receive
                    # a response from the engine even if no trading has occured since engine start (tested)
                    log.info("Algo bootstrapped ...")
                    self.bootstrapped = True
                else:
                    log.error(f"Received message {message} for unknown topic {topic}")

        except Exception:
            log.error(traceback.format_exc())

    def bootstrap(self):
        attempts = 0
        while attempts < 5 and self.stopCalled == False and self.bootstrapped == False:
            attempts = attempts + 1
            log.info(f"Bootstrap attempt {attempts} of 5 ...")
            self.bootstrapMDBook()
            self.bootstrapOrderStates()
            self.bootstrapPortfolio()
            time.sleep(3)

    def on_stop(self):
        log.info("Client Container Stopping ...")
        self.stopCalled = True

    def backgroundTask(self):
        while self.stopCalled == False:
            time.sleep(5)
            log.debug("Purging closed order states ...")
            orderbook.purgeClosedOrderStates()

    def action_background_task(self):
        try:
            log.info("Action background task called")
            while self.stopCalled == False:
                if self.stopCalled == False:
                    if self.firstRun == True:
                        self.firstRun = False
                        log.info("Waiting for bootstrap period ...")
                        time.sleep(5)
                        log.info("Bootstrap period done ...")
                    else:
                        if self.bootstrapped == True:
                            trader.onTakeAction()

                time.sleep(self.traderThrottle)
        except Exception:
            log.error(traceback.format_exc())

    def process_md(self, payload):
        if trader.isRunning() == True:
            header = chr(payload[0])
            if header == MD:
                try:
                    mdmsg = from_dict(data_class=MDMsg, data=loads(payload[1:]))
                    mdbook.applyUpdate(mdmsg)
                    trader.onMDMsg(mdbook, mdmsg)
                except Exception:
                    print(traceback.format_exc())
                    log.warning(f"Could not apply md update, possibly missing previous chain {mdmsg.id}")

            elif header == COMMAND:
                prefs.onCMDMsg()
                cmdmsg = from_dict(data_class=CommandMsg, data=loads(payload[1:]))
                if cmdmsg.type == 'MarketOpen':
                    if trader.isRunning() == False:
                        log.info(f"Starting algo on Market Open event ...")
                        trader.start()
                    else:
                        log.info(f"Algo is already running, ignoring MarketOpen event ...")

                elif cmdmsg.type == 'MarketClosed':
                    if trader.isRunning() == True:
                        log.info(f"Stopping algo on Market Closed ...")
                        trader.stop()
                    else:
                        log.info(f"Algo is already stopped, ignoring Market Closed event ...")

                elif cmdmsg.type == 'Shutdown':
                    if (cmdmsg.containerGUID == None) or (cmdmsg.containerGUID == prefs.get_container_GUID()):
                        log.debug(f"CMD Msg Counter: {prefs.getCMDMsgCounter()}")
                        if prefs.getCMDMsgCounter() > 1:
                            lckfile = open("shutdowncmd.lck", "w")
                            lckfile.close()
                            log.info(f"Shutdown called on client, exiting ...")
                            os._exit(1)

                elif cmdmsg.type == 'ConfigRequest':
                    if (cmdmsg.containerGUID == prefs.get_container_GUID()) or (cmdmsg.containerGUID == None):
                        log.info(f"Received config request from engine for GUID: {cmdmsg.containerGUID}, sending ...")
                        resp: ConfigRespMsg = prefs.build_config_response(trader.isRunning())
                        msg = CONFIG_RESP + dumps(dataclasses.asdict(resp))
                        log.info(f'Sending client config: {msg}')
                        self.sendOrderMsg(msg)

            else:
                log.info(f"Unsupported header {header}, raw message {payload}")

    def process_orders(self, payload):
        if trader.isRunning() == True:
            pass

    def process_executions(self, payload):
        if trader.isRunning() == True:
            header = chr(payload[0])
            if header == EXECUTION:
                exe = from_dict(data_class=ExecutionMsg, data=loads(payload[1:]))

                if exe.account == prefs.get_account():
                    if exe.requestorID == None or (
                        exe.requestorID != None and exe.requestorID == prefs.get_kafka_clientid()):
                        orderbook.onExecution(exe)
                        trader.onExecutionMsg(orderbook, exe)

    def process_positions(self, payload):
        if trader.isRunning() == True:
            header = chr(payload[0])
            if header == PORTFOLIO:
                portfolio = from_dict(data_class=PortfolioMsg, data=loads(payload[1:]))
                if portfolio.account == prefs.get_account():
                    trader.onPortfolioMsg(portfolio)

    def sendOrderMsg(self, order):
        msg = order.encode()
        tlog.info(f"[{ORDERS_TOPIC_NAME}]: >> {msg}")
        ordPool.submit(ordtopic.send, ORDERS_TOPIC_NAME, msg)

    def bootstrapOrderStates(self):
        tlog.info("Bootstrapping Order states ...")
        req = OrderStatusMsg(prefs.get_account(), prefs.get_symbol(), prefs.get_kafka_clientid())
        reqmsg = ORDSTATUS + dumps(dataclasses.asdict(req))
        self.sendOrderMsg(reqmsg)

    def bootstrapMDBook(self):
        tlog.info("Bootstrapping Market Data book ...")
        req = MDBookReqMsg(prefs.get_account(), prefs.get_symbol(), prefs.get_kafka_clientid())
        reqmsg = MD_BOOT_REQ + dumps(dataclasses.asdict(req))
        self.sendOrderMsg(reqmsg)

    def bootstrapPortfolio(self):
        tlog.info("Bootstrapping Portfolio ...")
        accsForReset = []
        if prefs.get_reset_portfolio_on_start() == True:
            accsForReset.append(prefs.get_account())
            log.info(f"Requesting portfolio reset for accounts: {accsForReset}")
        req = PortfolioBootReqMsg(prefs.get_kafka_clientid(), accsForReset)
        reqmsg = PORTFOLIO_BOOT_REQ + dumps(dataclasses.asdict(req))
        self.sendOrderMsg(reqmsg)

    def newOrder(self, price, qty, isBuy):
        date_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ".000"
        self.orderID += 1
        clid = 'CL' + kafkaclientid + str(self.orderID)
        order = OrderMsg(prefs.get_account(), prefs.get_symbol(), self.orderID, clid, isBuy, date_time, price, qty)
        orderbook.createPendingOrder(order)
        ord = NEW_ORDER + dumps(dataclasses.asdict(order))
        tlog.debug(f"[ORD]: {ord}")
        self.sendOrderMsg(ord)

    def cancelOrder(self, orderID: int):
        date_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ".000"
        exec = orderbook.getOrderByOrderID(prefs.get_symbol(), orderID)

        if len(exec.index) == 1:
            self.orderID += 1
            clid = 'CL' + kafkaclientid + str(self.orderID)
            cancel = CXLMsg(prefs.get_account(), prefs.get_symbol(), orderID, clid, bool(exec.iloc[0][OB.ISBUY]), date_time, int(exec.iloc[0][OB.PRICE]))
            orderbook.createPendingCXRorCXL(cancel)
            cxl = ORDER_CXL + dumps(dataclasses.asdict(cancel))
            tlog.debug(f"[CXL]: {cxl}")
            self.sendOrderMsg(cxl)
        else:
            tlog.debug(f"Could not find order by OrderID: {orderID} in cache, potentially a pending order")

    def cancelAll(self):
        date_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ".000"
        openOrders = orderbook.getOpenOrders(prefs.get_symbol())

        for index, row in openOrders.iterrows():
            self.orderID += 1
            clid = 'CL' + kafkaclientid + str(self.orderID)
            cancel = CXLMsg(prefs.get_account(), prefs.get_symbol(), row[OB.ORDERID], clid, bool(row[OB.ISBUY]), date_time, int(row[OB.PRICE]))
            orderbook.createPendingCXRorCXL(cancel)
            cxl = ORDER_CXL + dumps(dataclasses.asdict(cancel))
            tlog.debug(f"[CXL]: {cxl}")
            self.sendOrderMsg(cxl)

    def modOrderPriceAndQtyCore(self, exec, orderID, price, qty):
        date_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ".000"

        self.orderID += 1
        clid = 'CL' + kafkaclientid + str(self.orderID)
        cxr = CXRMsg(prefs.get_account(), prefs.get_symbol(), orderID, clid, bool(exec.iloc[0][OB.ISBUY]), date_time, int(exec.iloc[0][OB.PRICE]))
        orderbook.createPendingCXRorCXL(cxr)
        cxr = ORDER_CXR + dumps(dataclasses.asdict(cxr))
        tlog.debug(f"[CXR]: {cxr}")
        self.sendOrderMsg(cxr)

    def modOrderPriceAndQty(self, orderID, price, qty):
        exec = orderbook.getOrderByOrderID(prefs.get_symbol(), orderID)

        if len(exec.index) == 1:
            self.modOrderPriceAndQtyCore(exec, orderID, price, qty)
        else:
            tlog.debug(f"Could not find order by OrderID: {orderID} in cache, potentially a pending order")

    def modOrderPrice(self, orderID: int, price):
        exec = orderbook.getOrderByOrderID(prefs.get_symbol(), orderID)
        if len(exec.index) == 1:
            qty = exec.iloc[0][OB.QTY]
            self.modOrderPriceAndQtyCore(exec, orderID, price, qty)
        else:
            tlog.debug(f"Could not find order by OrderID: {orderID} in cache, potentially a pending order")

    def modOrderQty(self, orderID: int, qty):
        exec = orderbook.getOrderByOrderID(prefs.get_symbol(), orderID)
        if len(exec.index) == 1:
            price = exec.iloc[0][OB.PRICE]
            self.modOrderPriceAndQtyCore(exec, orderID, price, qty)
        else:
            tlog.debug(f"Could not find order by OrderID: {orderID} in cache, potentially a pending order")

    # Stack orders on one or both sides of the book on multiple price levels based on arguments
    def stackOrders(self, bookCenter: int, bidOFFSET: int, askOFFSET: int, bidDEPTH: int, askDEPTH: int, useRandomQty: bool, RANDOM_QTY_SHIFT_RANGE: int, bidQtys: list[int], askQtys: list[int]):
        log.info(f"Stack Orders:")
        log.info(f"bookCenter: {bookCenter}; bidOFFSET: {bidOFFSET}; askOFFSET: {askOFFSET}; bidDEPTH: {bidDEPTH}; askDEPTH: {askDEPTH}; useRandomQty: {useRandomQty}; RANDOM_QTY_SHIFT_RANGE: {RANDOM_QTY_SHIFT_RANGE}; bidQtys: {bidQtys}; askQtys: {askQtys}")

        if bidOFFSET == 0 and askOFFSET == 0:
            if bidDEPTH != 0 and askDEPTH != 0:
                askOFFSET = 1

        bidMax = (bookCenter - bidOFFSET)
        bidMin = (bidMax - bidDEPTH) + 1
        askMin = (bookCenter + askOFFSET)
        askMax = (askMin + askDEPTH) - 1

        bidQMap = []
        askQMap = []
        for pr in range(bidMin, bidMax+1):
            if useRandomQty == False:
                bidQMap[pr] = bidQtys.pop(0)
            else:
                bidQMap[pr] = random.randint(1, RANDOM_QTY_SHIFT_RANGE)

        for pr in range(askMin, askMax+1):
            if useRandomQty == False:
                askQMap[pr] = askQtys.pop(0)
            else:
                askQMap[pr] = random.randint(1, RANDOM_QTY_SHIFT_RANGE)

        # Build Price to qty mapping as not all price levels may be filled in each iteration
        
        bidQMap = []
        askQMap = []
        for pr in range(bidMin, bidMax+1):
            if useRandomQty == False:
                bidQMap[pr] = bidQtys.pop(0)
            else:
                bidQMap[pr] = random.randint(1, RANDOM_QTY_SHIFT_RANGE)

        for pr in range(askMin, askMax+1):
            if useRandomQty == False:
                askQMap[pr] = askQtys.pop(0)
            else:
                askQMap[pr] = random.randint(1, RANDOM_QTY_SHIFT_RANGE)

        df = orderbook.getOpenOrdersBySide(prefs.get_symbol(), True)
        for index, row in df.iterrows():
            # we have a existing crossing buy order try to move to new spot or just cancel it
            if (row[OB.ISBUY] == True) & (row[OB.PRICE] >= askMin or row[OB.PRICE] > bidMax):
                # Find a new spot for this order if it is outside bid range
                if len(availBidPrices) > 0:
                    nwPrice = availBidPrices.pop(0)
                    tlog.debug(f"Modifying crossing bid @ {row[OB.PRICE]} to {nwPrice}")
                    self.modOrderPriceAndQty(row[OB.ORDERID], nwPrice, bidQMap[nwPrice])
                else:
                    tlog.debug(f"Cancelling crossing bid @ {row[OB.PRICE]}")
                    self.cancelOrder(row[OB.ORDERID])

            # We have a resting & crossing ask, move or cancel it
            if (row[OB.ISBUY] == False) & (row[OB.PRICE] <= bidMax or row[OB.PRICE] < askMin):
                # Find a new spot for this order if it is outside ask range
                if len(availAskPrices) > 0:
                    nwPrice = availAskPrices.pop(0)
                    tlog.debug(f"Modifying crossing ask @ {row[OB.PRICE]} to {nwPrice}")
                    self.modOrderPriceAndQty(row[OB.ORDERID], nwPrice, askQMap[nwPrice])
                else:
                    tlog.debug(f"Cancelling crossing ask @ {row[OB.PRICE]}")
                    self.cancelOrder(row[OB.ORDERID])

        tlog.debug(f"Available Outstanding Bids: {availBidPrices}")
        tlog.debug(f"Available Outstanding Asks: {availAskPrices}")
        # Finally if we still have some prices available we need to put in new orders as these
        # for price in availBidPrices:
        tlog.debug(f"Adding bid @ {price}")
        self.newOrder(price, bidQMap[price], True)

        # for price in availAskPrices:
        tlog.debug(f"Adding ask @ {price}")
        self.newOrder(price, askQMap[price], False)

    """
    Loads configuration set of prices in a set and removes the price that are already consumed with existing orders
    """
    def buildPriceStackSet(self, min, max, isBuy) -> list:
        ret = list()
        for i in range(min, max+1):
            # Add the price range we need to fill
            if isBuy == True:
                ret.append(i) # add to end, maintain order
            else:
                ret.insert(0, i) # add to head, reverse order

        # Now remove price levels that already have an order
        df = orderbook.getOpenOrdersBySide(prefs.get_symbol(), isBuy)

        for index, row in df.iterrows():
            try:
                ret.remove(row[OB.PRICE])
            except Exception:
                pass

        return ret

    def getOrderID(self):
        self.orderID += 1
        return 'CL'+prefs.get_kafka_clientid()+str(self.orderID)

    def getPreferences(self) -> Preferences:
        return prefs

def print_raw_msg(prefix, payload):
    #print(f"[LOGGING] {prefix} Header: {chr(payload[0])} Body: {payload[1:]}")
    tlog.debug(f"[{prefix}] Header: {chr(payload[0])} Body: {payload[1:]}")
    #capture log writelines?("[{prefix}] Header: {chr(payload[0])} Body: {payload[1:]}")

def blockForInputsOrExitCmd():
    while True:
        try:
            cmd = input("Input (m/o/p/a) for (MDBook/OrderBook/Portfolio/All) respectively or b:bootstrap, q to exit: ")
            log.info(f"Key entered: {cmd}")
            if cmd == 'm':
                log.info(f"MDBook: {mdbook.printBook()}")
            elif cmd == 'o':
                log.info(f"OrderBook: {orderbook.printBook()}")
            elif cmd == 'p':
                log.info(f"Portfolio: {portfolio}")
            elif cmd == 'a':
                log.info(f"MDBook: {mdbook.printBook()}")
                log.info(f"OrderBook: {orderbook.printBook()}")
                log.info(f"Portfolio: {portfolio}")
            elif cmd == 'b':
                log.info(f"Triggering bootstrap")
                container.bootstrap()
            elif cmd == 'q':
                mdtopic.commit()
                exectopic.commit()
                postopic.commit()

                ordPool.shutdown(wait=False, cancel_futures=True)
                execPool.shutdown(wait=False, cancel_futures=True)
                posPool.shutdown(wait=False, cancel_futures=True)
                mdPool.shutdown(wait=False, cancel_futures=True)
                bkTaskPool.shutdown(wait=False, cancel_futures=True)
                algoActionPool.shutdown(wait=False, cancel_futures=True)
                os._exit(0)

        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            traceback.print_exc()

# Command line to start this service
# python ClientContainer-faust.py worker -l info

if __name__ == '__main__':
    container = KafkaPyContainerService()
    container.on_start()
    blockForInputsOrExitCmd()
