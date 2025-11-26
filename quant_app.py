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
import textwrap # 用于处理代码缩进

# ==========================================
# 0. 系统配置
# ==========================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import warnings
import matplotlib.dates

if not hasattr(matplotlib.dates, 'warnings'):
    matplotlib.dates.warnings = warnings

plt.style.use('seaborn-v0_8') 

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
# 1. 增强版策略类 (支持自定义代码)
# ==========================================
class PortfolioStrategy(bt.Strategy):
    params = (
        ('strategy_type', 'SMA'), 
        ('use_risk_mgmt', False), 
        ('stop_loss', 0.05),      
        ('take_profit', 0.10),
        # 现有参数
        ('pfast', 10), ('pslow', 30),
        ('rsi_period', 14), ('rsi_low', 30), ('rsi_high', 70),
        ('boll_period', 20), ('boll_dev', 2.0),
        # 新增参数
        ('turtle_period', 20), # 海龟策略周期
        ('mean_period', 20),   # 均值回归周期
        ('custom_code', ''),   # 用户自定义代码字符串
    )

    def __init__(self):
        self.inds = {} 
        
        for d in self.datas:
            if self.params.strategy_type == 'SMA':
                sma1 = bt.indicators.SimpleMovingAverage(d, period=self.params.pfast)
                sma2 = bt.indicators.SimpleMovingAverage(d, period=self.params.pslow)
                self.inds[d] = bt.indicators.CrossOver(sma1, sma2)
            
            elif self.params.strategy_type == 'RSI':
                self.inds[d] = bt.indicators.RSI(d, period=self.params.rsi_period)
            
            elif self.params.strategy_type == 'Bollinger':
                self.inds[d] = bt.indicators.BollingerBands(d, period=self.params.boll_period, devfactor=self.params.boll_dev)
            
            # --- 新增策略指标 ---
            elif self.params.strategy_type == 'Turtle':
                # 唐奇安通道: 过去N天的最高价和最低价
                self.inds[d] = {
                    'high': bt.indicators.Highest(d.high(-1), period=self.params.turtle_period),
                    'low': bt.indicators.Lowest(d.low(-1), period=self.params.turtle_period)
                }
            
            elif self.params.strategy_type == 'MeanRev':
                # 均值回归: 价格偏离均线太远
                sma = bt.indicators.SMA(d, period=self.params.mean_period)
                self.inds[d] = {
                    'sma': sma,
                    'dist': (d.close - sma) / sma # 偏离率
                }

    def next(self):
        # 资金分配: 等权重
        target_pct = 0.95 / len(self.datas)

        for d in self.datas:
            pos = self.getposition(d).size
            
            # --- 0. 自定义策略 (最高优先级) ---
            if self.params.strategy_type == 'Custom':
                # 这是一个危险但强大的功能：动态执行用户代码
                # 我们把当前的上下文变量暴露给用户代码
                context = {
                    'self': self, 
                    'd': d, 
                    'pos': pos, 
                    'target_pct': target_pct,
                    'bt': bt
                }
                try:
                    exec(self.params.custom_code, {}, context)
                except Exception as e:
                    # 避免报错刷屏，只打印一次
                    pass
                continue # 执行完自定义代码后，跳过后续逻辑

            # --- 1. 风控逻辑 ---
            if pos != 0 and self.params.use_risk_mgmt:
                buy_price = self.getposition(d).price
                current_price = d.close[0]
                pnl_pct = (current_price - buy_price) / buy_price

                if pnl_pct <= -self.params.stop_loss:
                    self.close(data=d)
                    continue 
                if pnl_pct >= self.params.take_profit:
                    self.close(data=d)
                    continue

            # --- 2. 内置策略逻辑 ---
            
            # SMA (双均线)
            if self.params.strategy_type == 'SMA':
                if not pos and self.inds[d] > 0: self.order_target_percent(data=d, target=target_pct)
                elif pos and self.inds[d] < 0: self.close(data=d)

            # RSI
            elif self.params.strategy_type == 'RSI':
                if not pos and self.inds[d] < self.params.rsi_low: self.order_target_percent(data=d, target=target_pct)
                elif pos and self.inds[d] > self.params.rsi_high: self.close(data=d)
            
            # Bollinger
            elif self.params.strategy_type == 'Bollinger':
                if not pos and d.close[0] < self.inds[d].lines.bot[0]: self.order_target_percent(data=d, target=target_pct)
                elif pos and d.close[0] > self.inds[d].lines.top[0]: self.close(data=d)

            # Turtle (海龟/唐奇安通道)
            elif self.params.strategy_type == 'Turtle':
                # 突破过去20天最高价 -> 买入
                if not pos and d.close[0] > self.inds[d]['high'][0]:
                    self.order_target_percent(data=d, target=target_pct)
                # 跌破过去10天最低价 -> 卖出 (这里简化为同周期)
                elif pos and d.close[0] < self.inds[d]['low'][0]:
                    self.close(data=d)

            # Mean Reversion (均值回归)
            elif self.params.strategy_type == 'MeanRev':
                # 价格低于均线 5% -> 买入
                if not pos and self.inds[d]['dist'][0] < -0.05:
                    self.order_target_percent(data=d, target=target_pct)
                # 回归到均线 -> 卖出
                elif pos and d.close[0] >= self.inds[d]['sma'][0]:
                    self.close(data=d)

# ==========================================
# 2. 数据获取 (保持 V2.0)
# ==========================================
@st.cache_data(ttl=3600)
def get_multiple_data(source, tickers_list, start_date, end_date):
    data_dict = {}
    bench_df = pd.DataFrame()
    for ticker in tickers_list:
        ticker = ticker.strip()
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
        except: pass

    try:
        if source == "美股/港股": bench_df = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
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
# 3. 界面辅助
# ==========================================
def show_welcome_guide():
    st.info("👋 欢迎！V2.1 新增：海龟策略、均值回归、以及**自定义代码模式**。")

# ==========================================
# 4. 主程序
# ==========================================
def main():
    st.set_page_config(page_title="量化极客版", layout="wide", page_icon="👨‍💻", initial_sidebar_state="collapsed")
    st.title("👨‍💻 量化极客版 (V2.1)")
    
    show_welcome_guide()

    col_input, col_action = st.columns([3, 1])
    with col_input:
        if 'demo_mode' not in st.session_state: st.session_state.demo_mode = False
        default_tickers = "AAPL, MSFT, NVDA"
        default_source = "美股/港股"
        
        if st.button("🎲 随机演示"):
            st.session_state.demo_mode = True
            demos = [("美股/港股", "AAPL, TSLA, AMZN"), ("A股", "600519, 000858, 600036")]
            choice = random.choice(demos)
            default_source = choice[0]
            default_tickers = choice[1]

        data_source = st.selectbox("市场", ["美股/港股", "A股"], index=0 if default_source=="美股/港股" else 1)
        tickers_input = st.text_area("股票代码", value=default_tickers, height=68)

    # --- 策略配置区 ---
    with st.expander("⚙️ 策略配置 (点我展开)", expanded=True):
        c1, c2 = st.columns(2)
        start_date = c1.date_input("开始", datetime.date(2021, 1, 1))
        cash = c2.number_input("本金", 100000)
        
        # 策略映射
        strat_map = {
            "双均线 (趋势)": "SMA", 
            "RSI (反转)": "RSI", 
            "布林带 (通道)": "Bollinger",
            "海龟交易 (突破)": "Turtle",  # 新增
            "均值回归 (抄底)": "MeanRev", # 新增
            "🛠️ 自定义策略 (写代码)": "Custom" # 新增
        }
        s_name = st.selectbox("策略模型", list(strat_map.keys()))
        s_code = strat_map[s_name]
        
        # 动态参数显示
        params = {}
        custom_code_input = ""
        
        if s_code == "Custom":
            st.warning("⚠️ 高级功能：请直接编写 Python 代码逻辑。变量 `d` 代表当前股票数据，`pos` 代表当前持仓。")
            
            # 代码模板
            code_template = """# 示例：简单的价格突破策略
# 如果 收盘价 > 开盘价 * 1.02 (涨2%) -> 买入
# 如果 收盘价 < 开盘价 * 0.98 (跌2%) -> 卖出

if not pos and d.close[0] > d.open[0] * 1.02:
    self.order_target_percent(data=d, target=target_pct)
    
elif pos and d.close[0] < d.open[0] * 0.98:
    self.close(data=d)
"""
            custom_code_input = st.text_area("Python 代码编辑器", value=code_template, height=200)
            params['custom_code'] = custom_code_input
            
        elif s_code == "SMA":
            params['pfast'] = st.slider("快线", 5, 30, 10)
            params['pslow'] = st.slider("慢线", 20, 60, 30)
        elif s_code == "Turtle":
            st.caption("💡 海龟法则：突破过去 N 天最高价买入。")
            params['turtle_period'] = st.slider("突破周期 (天)", 10, 60, 20)
        elif s_code == "MeanRev":
            st.caption("💡 均值回归：价格偏离均线太远时反向操作。")
            params['mean_period'] = st.slider("均线周期", 10, 50, 20)
        # ... 其他略 ...

        # 风控 (自定义模式下通常由代码控制，但这里保留作为全局硬风控)
        if s_code != "Custom":
            st.caption("🛡️ 全局风控")
            use_risk = st.checkbox("开启止盈止损", value=True)
            stop_loss = st.slider("止损%", 1, 20, 5) / 100.0
            take_profit = st.slider("止盈%", 5, 50, 15) / 100.0
        else:
            use_risk = False # 自定义模式默认关闭硬风控，交给代码
            stop_loss = 0.05
            take_profit = 0.15

    run_btn = st.button("🚀 开始回测", type="primary", use_container_width=True)

    if run_btn or st.session_state.demo_mode:
        st.session_state.demo_mode = False
        ticker_list = [t.strip() for t in tickers_input.split(',') if t.strip()]
        
        if not ticker_list: st.error("请输入代码")
        else:
            with st.spinner("正在计算..."):
                data_dict, df_bench = get_multiple_data(data_source, ticker_list, start_date, datetime.date.today())
                
                if not data_dict: st.error("数据失败")
                else:
                    cerebro = bt.Cerebro()
                    for t, df in data_dict.items():
                        data = bt.feeds.PandasData(dataname=df, name=t)
                        cerebro.adddata(data)
                    
                    cerebro.addstrategy(PortfolioStrategy, strategy_type=s_code, use_risk_mgmt=use_risk, stop_loss=stop_loss, take_profit=take_profit, **params)
                    cerebro.broker.setcash(cash)
                    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
                    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                    
                    results = cerebro.run()
                    strat = results[0]
                    
                    strat_returns = pd.Series(strat.analyzers.returns.get_analysis())
                    strat_returns.index = pd.to_datetime(strat_returns.index)
                    
                    bench_returns = None
                    if not df_bench.empty and 'close' in df_bench.columns:
                        bench_returns = df_bench['close'].pct_change().fillna(0)
                        bench_returns.index = pd.to_datetime(bench_returns.index)
                        bench_returns = bench_returns.reindex(strat_returns.index).fillna(0)

                    final_cash = cerebro.broker.getvalue()
                    ret_pct = (final_cash - cash) / cash
                    max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("最终资产", f"${final_cash/1000:.1f}k")
                    c2.metric("总收益", f"{ret_pct*100:.1f}%", delta_color="normal" if ret_pct>0 else "inverse")
                    c3.metric("最大回撤", f"{max_dd:.1f}%")
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    cum_strat = (1 + strat_returns).cumprod()
                    ax.plot(cum_strat.index, cum_strat, color='#2962FF', linewidth=2)
                    if bench_returns is not None:
                        cum_bench = (1 + bench_returns).cumprod()
                        ax.plot(cum_bench.index, cum_bench, color='gray', linestyle='--', alpha=0.6)
                    st.pyplot(fig)
                    
                    with st.expander("📊 详细报告"):
                        try:
                            report_file = "qs_mobile.html"
                            qs.reports.html(strat_returns, benchmark=bench_returns, output=report_file, title="Report", download_filename=report_file)
                            with open(report_file, 'r', encoding='utf-8') as f: components.html(f.read(), height=600, scrolling=True)
                        except: pass

if __name__ == '__main__':
    main()
