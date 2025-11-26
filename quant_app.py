import streamlit as st
import datetime
import pandas as pd
import yfinance as yf
import akshare as ak
import numpy as np
import time
import quantstats as qs
import streamlit.components.v1 as components
import os
import requests 
from matplotlib import font_manager
import random

# ==========================================
# 0. 系统配置 (手机适配 & 字体修复)
# ==========================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import warnings
import matplotlib.dates

if not hasattr(matplotlib.dates, 'warnings'):
    matplotlib.dates.warnings = warnings

plt.style.use('seaborn-v0_8') 

# 字体初始化 (保持不变)
def init_chinese_font():
    font_name = "SimHei.ttf"
    font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
    if not os.path.exists(font_name):
        try:
            response = requests.get(font_url, timeout=20)
            with open(font_name, "wb") as f: f.write(response.content)
        except: pass
    if os.path.exists(font_name):
        font_manager.fontManager.addfont(font_name)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

init_chinese_font()

import backtrader as bt

# ==========================================
# 1. 组合策略类 (支持多只股票)
# ==========================================
class PortfolioStrategy(bt.Strategy):
    params = (
        ('strategy_type', 'SMA'), 
        ('use_risk_mgmt', False), 
        ('stop_loss', 0.05),      
        ('take_profit', 0.10),    
        ('pfast', 10), ('pslow', 30),
        ('rsi_period', 14), ('rsi_low', 30), ('rsi_high', 70),
        ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9),
        ('boll_period', 20), ('boll_dev', 2.0),
        ('mom_period', 10),
    )

    def __init__(self):
        self.inds = {} # 存储每只股票的指标
        self.orders = {} # 存储每只股票的订单状态

        # 遍历所有传入的股票数据 (self.datas)
        for d in self.datas:
            self.orders[d] = None
            
            # 为每一只股票初始化指标
            if self.params.strategy_type == 'SMA':
                sma1 = bt.indicators.SimpleMovingAverage(d, period=self.params.pfast)
                sma2 = bt.indicators.SimpleMovingAverage(d, period=self.params.pslow)
                self.inds[d] = bt.indicators.CrossOver(sma1, sma2)
            
            elif self.params.strategy_type == 'RSI':
                self.inds[d] = bt.indicators.RSI(d, period=self.params.rsi_period)
            
            elif self.params.strategy_type == 'MACD':
                macd = bt.indicators.MACD(d, period_me1=self.params.macd_fast, period_me2=self.params.macd_slow, period_signal=self.params.macd_signal)
                self.inds[d] = bt.indicators.CrossOver(macd.macd, macd.signal)
            
            elif self.params.strategy_type == 'Bollinger':
                self.inds[d] = bt.indicators.BollingerBands(d, period=self.params.boll_period, devfactor=self.params.boll_dev)
            
            elif self.params.strategy_type == 'Momentum':
                self.inds[d] = bt.indicators.Momentum(d, period=self.params.mom_period)

    def log(self, txt, dt=None):
        # 简化日志，只在控制台打印，避免手机端刷屏
        pass 

    def next(self):
        # 计算每只股票的目标仓位
        # 例如：有 5 只股票，每只股票最多占用 19% 的资金 (留 5% 现金)
        target_pct = 0.95 / len(self.datas)

        for d in self.datas:
            pos = self.getposition(d).size
            
            # --- 风控逻辑 (优先级最高) ---
            if pos != 0 and self.params.use_risk_mgmt:
                buy_price = self.getposition(d).price
                current_price = d.close[0]
                pnl_pct = (current_price - buy_price) / buy_price

                if pnl_pct <= -self.params.stop_loss:
                    self.close(data=d)
                    continue # 止损后跳过该股票的策略判断
                if pnl_pct >= self.params.take_profit:
                    self.close(data=d)
                    continue

            # --- 策略逻辑 ---
            # 1. SMA
            if self.params.strategy_type == 'SMA':
                if not pos and self.inds[d] > 0: # 金叉买入
                    self.order_target_percent(data=d, target=target_pct)
                elif pos and self.inds[d] < 0: # 死叉卖出
                    self.close(data=d)

            # 2. RSI
            elif self.params.strategy_type == 'RSI':
                if not pos and self.inds[d] < self.params.rsi_low:
                    self.order_target_percent(data=d, target=target_pct)
                elif pos and self.inds[d] > self.params.rsi_high:
                    self.close(data=d)

            # 3. MACD
            elif self.params.strategy_type == 'MACD':
                if not pos and self.inds[d] > 0:
                    self.order_target_percent(data=d, target=target_pct)
                elif pos and self.inds[d] < 0:
                    self.close(data=d)
            
            # 4. Bollinger
            elif self.params.strategy_type == 'Bollinger':
                if not pos and d.close[0] < self.inds[d].lines.bot[0]:
                    self.order_target_percent(data=d, target=target_pct)
                elif pos and d.close[0] > self.inds[d].lines.top[0]:
                    self.close(data=d)

            # 5. Momentum
            elif self.params.strategy_type == 'Momentum':
                if not pos and self.inds[d] > 0:
                    self.order_target_percent(data=d, target=target_pct)
                elif pos and self.inds[d] < 0:
                    self.close(data=d)

# ==========================================
# 2. 多股票数据获取
# ==========================================
@st.cache_data(ttl=3600)
def get_multiple_data(source, tickers_list, start_date, end_date):
    data_dict = {}
    bench_df = pd.DataFrame()
    
    # 1. 获取股票数据
    for ticker in tickers_list:
        ticker = ticker.strip() # 去除空格
        if not ticker: continue
        
        try:
            if source == "美股/港股":
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=10)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df.columns = df.columns.str.lower()
                if not df.empty: data_dict[ticker] = df
                
            elif source == "A股":
                s_str = start_date.strftime("%Y%m%d")
                e_str = end_date.strftime("%Y%m%d")
                df = ak.stock_zh_a_hist(symbol=ticker, start_date=s_str, end_date=e_str, adjust="qfq")
                df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
                df.index = pd.to_datetime(df['date'])
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.columns = df.columns.str.lower()
                if not df.empty: data_dict[ticker] = df
        except:
            pass # 单个失败不影响整体

    # 2. 获取基准数据
    try:
        if source == "美股/港股":
            bench_df = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
        else:
            bench_df = ak.stock_zh_index_daily(symbol="sh000300")
            bench_df.rename(columns={'date': 'date', 'close': 'close'}, inplace=True)
            bench_df.index = pd.to_datetime(bench_df['date'])
            bench_df = bench_df[(bench_df.index >= pd.to_datetime(start_date)) & (bench_df.index <= pd.to_datetime(end_date))]
        
        if isinstance(bench_df.columns, pd.MultiIndex): bench_df.columns = bench_df.columns.get_level_values(0)
        bench_df.columns = bench_df.columns.str.lower()
    except: pass

    return data_dict, bench_df

# ==========================================
# 3. 界面辅助函数
# ==========================================
def show_welcome_guide():
    st.info("👋 欢迎！这是一个可以在手机上玩的量化模拟器。")
    with st.expander("🐣 新手怎么玩？(点我展开)", expanded=False):
        st.markdown("""
        1.  **输入代码**: 支持同时测多只股票，用逗号隔开。
            *   A股示例: `600519, 000001, 300750`
            *   美股示例: `AAPL, TSLA, NVDA`
        2.  **系统会自动分配资金**: 如果你选了 3 只股票，系统会把钱分成 3 份，分别去买卖。
        3.  **点击“开始回测”**: 看你的策略组合能不能跑赢大盘！
        """)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    st.set_page_config(page_title="量化口袋版", layout="wide", page_icon="📱", initial_sidebar_state="collapsed")
    
    st.title("📱 量化口袋版 (V2.0)")
    
    # --- 顶部快速操作区 (手机友好) ---
    show_welcome_guide()

    col_input, col_action = st.columns([3, 1])
    
    with col_input:
        # 默认值逻辑
        if 'demo_mode' not in st.session_state: st.session_state.demo_mode = False
        
        default_tickers = "AAPL, MSFT, NVDA"
        default_source = "美股/港股"
        
        # 🎲 随机演示功能
        if st.button("🎲 随机演示 (点我试试)"):
            st.session_state.demo_mode = True
            demos = [
                ("美股/港股", "AAPL, TSLA, AMZN, GOOG"),
                ("A股", "600519, 000858, 600036"), # 茅台, 五粮液, 招行
                ("美股/港股", "BTC-USD, ETH-USD")
            ]
            choice = random.choice(demos)
            default_source = choice[0]
            default_tickers = choice[1]

        # 输入区域
        data_source = st.selectbox("市场", ["美股/港股", "A股"], index=0 if default_source=="美股/港股" else 1)
        tickers_input = st.text_area("股票代码 (用逗号隔开)", value=default_tickers, height=68, help="例如: AAPL, TSLA")

    # --- 折叠的高级设置 (手机端不占地) ---
    with st.expander("⚙️ 策略与风控设置 (点我修改)", expanded=False):
        c1, c2 = st.columns(2)
        start_date = c1.date_input("开始", datetime.date(2021, 1, 1))
        cash = c2.number_input("本金", 100000)
        
        strat_map = {"双均线 (趋势)": "SMA", "RSI (反转)": "RSI", "布林带 (通道)": "Bollinger"}
        s_name = st.selectbox("策略模型", list(strat_map.keys()))
        s_code = strat_map[s_name]
        
        st.caption("🛡️ 风控: 亏5%止损，赚15%止盈")
        use_risk = True
        stop_loss = 0.05
        take_profit = 0.15
        
        # 简单参数
        params = {}
        if s_code == "SMA":
            params['pfast'] = st.slider("快线", 5, 30, 10)
            params['pslow'] = st.slider("慢线", 20, 60, 30)
        elif s_code == "RSI":
            params['rsi_period'] = 14
            params['rsi_low'] = 30
            params['rsi_high'] = 70
        elif s_code == "Bollinger":
            params['boll_period'] = 20
            params['boll_dev'] = 2.0

    # --- 核心运行按钮 (大大的) ---
    run_btn = st.button("🚀 开始组合回测", type="primary", use_container_width=True)

    # --- 结果展示区 ---
    if run_btn or st.session_state.demo_mode:
        st.session_state.demo_mode = False # 重置演示状态
        
        ticker_list = [t.strip() for t in tickers_input.split(',') if t.strip()]
        
        if not ticker_list:
            st.error("请输入至少一个股票代码")
        else:
            with st.spinner(f"正在分析 {len(ticker_list)} 只股票的组合..."):
                # 1. 获取数据
                data_dict, df_bench = get_multiple_data(data_source, ticker_list, start_date, datetime.date.today())
                
                if not data_dict:
                    st.error("数据获取失败，请检查代码")
                else:
                    # 2. 回测引擎
                    cerebro = bt.Cerebro()
                    
                    # 添加所有股票数据
                    for t, df in data_dict.items():
                        data = bt.feeds.PandasData(dataname=df, name=t)
                        cerebro.adddata(data)
                    
                    cerebro.addstrategy(PortfolioStrategy, strategy_type=s_code, use_risk_mgmt=use_risk, stop_loss=stop_loss, take_profit=take_profit, **params)
                    cerebro.broker.setcash(cash)
                    
                    # 分析器
                    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
                    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                    
                    results = cerebro.run()
                    strat = results[0]
                    
                    # 3. 结果处理
                    strat_returns = pd.Series(strat.analyzers.returns.get_analysis())
                    strat_returns.index = pd.to_datetime(strat_returns.index)
                    
                    # 基准处理
                    bench_returns = None
                    if not df_bench.empty and 'close' in df_bench.columns:
                        bench_returns = df_bench['close'].pct_change().fillna(0)
                        bench_returns.index = pd.to_datetime(bench_returns.index)
                        bench_returns = bench_returns.reindex(strat_returns.index).fillna(0)

                    # 4. 手机端友好的展示
                    final_cash = cerebro.broker.getvalue()
                    ret_pct = (final_cash - cash) / cash
                    max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
                    
                    st.success("✅ 回测完成！")
                    
                    # 核心卡片
                    c1, c2, c3 = st.columns(3)
                    c1.metric("最终资产", f"${final_cash/1000:.1f}k", help="单位: 千")
                    c2.metric("总收益", f"{ret_pct*100:.1f}%", delta_color="normal" if ret_pct>0 else "inverse")
                    c3.metric("最大回撤", f"{max_dd:.1f}%")
                    
                    # 简单图表
                    st.caption("📈 资金曲线 (蓝色: 你的组合 | 灰色: 大盘)")
                    fig, ax = plt.subplots(figsize=(8, 4)) # 手机端图表做小一点
                    cum_strat = (1 + strat_returns).cumprod()
                    ax.plot(cum_strat.index, cum_strat, color='#2962FF', linewidth=2)
                    
                    if bench_returns is not None:
                        cum_bench = (1 + bench_returns).cumprod()
                        ax.plot(cum_bench.index, cum_bench, color='gray', linestyle='--', alpha=0.6)
                        # 填充颜色
                        ax.fill_between(cum_strat.index, cum_strat, cum_bench, where=(cum_strat>cum_bench), color='green', alpha=0.1)
                        ax.fill_between(cum_strat.index, cum_strat, cum_bench, where=(cum_strat<=cum_bench), color='red', alpha=0.1)
                    
                    st.pyplot(fig)
                    
                    # 组合详情
                    with st.expander("📊 查看组合详情 (QuantStats)"):
                        try:
                            report_file = "qs_mobile.html"
                            qs.reports.html(strat_returns, benchmark=bench_returns, output=report_file, title="Portfolio Report", download_filename=report_file)
                            with open(report_file, 'r', encoding='utf-8') as f:
                                components.html(f.read(), height=600, scrolling=True)
                        except: st.error("报告生成失败")

if __name__ == '__main__':
    main()
