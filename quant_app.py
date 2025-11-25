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

# 字体初始化
def init_chinese_font():
    font_name = "SimHei.ttf"
    font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
    if not os.path.exists(font_name):
        with st.spinner("正在初始化中文字体..."):
            try:
                response = requests.get(font_url, timeout=20)
                with open(font_name, "wb") as f:
                    f.write(response.content)
            except: pass
    if os.path.exists(font_name):
        font_manager.fontManager.addfont(font_name)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

init_chinese_font()

import backtrader as bt

# ==========================================
# 1. 策略类
# ==========================================
class MegaStrategy(bt.Strategy):
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
        self.dataclose = self.datas[0].close
        self.log_list = [] 
        self.order = None 

        self.sma1 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.pfast)
        self.sma2 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.pslow)
        self.sma_cross = bt.indicators.CrossOver(self.sma1, self.sma2)
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(self.datas[0], period_me1=self.params.macd_fast, period_me2=self.params.macd_slow, period_signal=self.params.macd_signal)
        self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.boll = bt.indicators.BollingerBands(self.datas[0], period=self.params.boll_period, devfactor=self.params.boll_dev)
        self.mom = bt.indicators.Momentum(self.datas[0], period=self.params.mom_period)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        self.log_list.append(f'{dt.isoformat()}: {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'🟢 买入成功: 价格 {order.executed.price:.2f} | 数量: {order.executed.size}')
            elif order.issell():
                self.log(f'🔴 卖出成功: 价格 {order.executed.price:.2f} | 盈亏: {order.executed.pnl:.2f}')
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('⚠️ 订单被拒绝 (可能是资金不足)')

    def next(self):
        if self.order: return

        if self.position and self.params.use_risk_mgmt:
            buy_price = self.position.price
            current_price = self.dataclose[0]
            pnl_pct = (current_price - buy_price) / buy_price

            if pnl_pct <= -self.params.stop_loss:
                self.log(f'🛡️ 触发止损 (亏损 {pnl_pct*100:.2f}%)')
                self.close()
                return
            if pnl_pct >= self.params.take_profit:
                self.log(f'💰 触发止盈 (盈利 {pnl_pct*100:.2f}%)')
                self.close()
                return

        if not self.position:
            if self.params.strategy_type == 'SMA' and self.sma_cross > 0: self.buy()
            elif self.params.strategy_type == 'RSI' and self.rsi < self.params.rsi_low: self.buy()
            elif self.params.strategy_type == 'MACD' and self.macd_cross > 0: self.buy()
            elif self.params.strategy_type == 'Bollinger' and self.dataclose < self.boll.lines.bot: self.buy()
            elif self.params.strategy_type == 'Momentum' and self.mom > 0: self.buy()
        else:
            if self.params.strategy_type == 'SMA' and self.sma_cross < 0: self.sell()
            elif self.params.strategy_type == 'RSI' and self.rsi > self.params.rsi_high: self.sell()
            elif self.params.strategy_type == 'MACD' and self.macd_cross < 0: self.sell()
            elif self.params.strategy_type == 'Bollinger' and self.dataclose > self.boll.lines.top: self.sell()
            elif self.params.strategy_type == 'Momentum' and self.mom < 0: self.sell()

# ==========================================
# 2. 数据获取
# ==========================================
@st.cache_data(ttl=3600)
def get_data_with_benchmark(source, ticker, start_date, end_date):
    stock_df = pd.DataFrame()
    bench_df = pd.DataFrame()
    max_retries = 3
    
    try:
        if source == "美股/港股 (Yahoo)":
            for i in range(max_retries):
                try:
                    if i > 0: time.sleep(1)
                    stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=10)
                    if not stock_df.empty: break
                except: pass
            if isinstance(stock_df.columns, pd.MultiIndex): stock_df.columns = stock_df.columns.get_level_values(0)
            stock_df.columns = stock_df.columns.str.lower()
            
            bench_ticker = "^GSPC" 
            for i in range(max_retries):
                try:
                    if i > 0: time.sleep(1)
                    bench_df = yf.download(bench_ticker, start=start_date, end=end_date, progress=False, timeout=10)
                    if not bench_df.empty: break
                except: pass
            if isinstance(bench_df.columns, pd.MultiIndex): bench_df.columns = bench_df.columns.get_level_values(0)
            bench_df.columns = bench_df.columns.str.lower()

        elif source == "A股 (AkShare)":
            s_str = start_date.strftime("%Y%m%d")
            e_str = end_date.strftime("%Y%m%d")
            try:
                stock_df = ak.stock_zh_a_hist(symbol=ticker, start_date=s_str, end_date=e_str, adjust="qfq")
                stock_df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
                stock_df.index = pd.to_datetime(stock_df['date'])
                stock_df = stock_df[['open', 'high', 'low', 'close', 'volume']]
                stock_df.columns = stock_df.columns.str.lower()

                bench_df = ak.stock_zh_index_daily(symbol="sh000300")
                bench_df.rename(columns={'date': 'date', 'close': 'close'}, inplace=True)
                bench_df.index = pd.to_datetime(bench_df['date'])
                bench_df = bench_df[(bench_df.index >= pd.to_datetime(start_date)) & (bench_df.index <= pd.to_datetime(end_date))]
                bench_df.columns = bench_df.columns.str.lower()
            except Exception as e:
                st.error(f"AkShare 接口报错: {e}")

        return stock_df, bench_df
    except Exception as e:
        st.error(f"严重数据错误: {e}")
        return None, None

# ==========================================
# 3. 文案内容
# ==========================================
def show_user_guide():
    st.markdown("""
    ### 🐣 新手第一课：怎么玩这个系统？
    **1. 选股票**: 左边选市场，输入代码（如 AAPL 或 600519）。
    **2. 选策略**: 推荐先试用“双均线”。
    **3. 设风控**: 止损建议 5%，止盈建议 15%。
    **4. 资金**: 建议本金设大一点（如 50万），防止买不起一手茅台。
    """)

def show_strategy_wiki():
    st.markdown("""
    ### 📖 策略大白话
    *   **双均线**: 追涨杀跌，牛市神器。
    *   **RSI**: 震荡市神器，高抛低吸。
    *   **布林带**: 跌破下轨买，突破上轨卖。
    """)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    st.set_page_config(page_title="量化回测新手版", layout="wide", page_icon="🌱")
    st.title("🌱 个人量化回测系统 (新手友好版)")

    with st.sidebar:
        st.header("🎛️ 操作面板")
        data_source = st.selectbox("数据来源", ["美股/港股 (Yahoo)", "A股 (AkShare)"])
        if data_source == "美股/港股 (Yahoo)":
            ticker = st.text_input("股票代码", "AAPL")
            benchmark_name = "标普500"
        else:
            ticker = st.text_input("股票代码", "600519")
            benchmark_name = "沪深300"
            
        col1, col2 = st.columns(2)
        start_date = col1.date_input("开始", datetime.date(2021, 1, 1))
        end_date = col2.date_input("结束", datetime.date.today())
        cash = st.number_input("初始本金", 500000, help="建议设大一点，防止买不起高价股")

        st.subheader("策略选择")
        strat_map = {"双均线 (趋势)": "SMA", "RSI (反转)": "RSI", "MACD (综合)": "MACD", "布林带 (通道)": "Bollinger", "动量 (惯性)": "Momentum"}
        s_name = st.selectbox("模型", list(strat_map.keys()))
        s_code = strat_map[s_name]

        with st.expander("🛡️ 风控设置", expanded=True):
            use_risk = st.checkbox("开启自动止盈止损", value=True)
            stop_loss = st.slider("止损线", 1, 20, 5) / 100.0
            take_profit = st.slider("止盈线", 5, 50, 15) / 100.0

        params = {}
        if s_code == "SMA":
            params['pfast'] = st.slider("快线周期", 5, 50, 10)
            params['pslow'] = st.slider("慢线周期", 20, 100, 30)
        elif s_code == "RSI":
            params['rsi_period'] = st.slider("周期", 5, 30, 14)
            params['rsi_low'] = st.slider("抄底分", 10, 40, 30)
            params['rsi_high'] = st.slider("逃顶分", 60, 90, 70)
        # ... 其他参数省略 ...

        run_btn = st.button("🚀 开始回测", type="primary")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 结果看板", "📑 专业体检报告", "📖 策略大白话", "🐣 新手教程"])

    with tab1:
        if run_btn:
            with st.spinner("正在模拟交易..."):
                df_stock, df_bench = get_data_with_benchmark(data_source, ticker, start_date, end_date)
                
                if df_stock is None or df_stock.empty:
                    st.error("数据获取失败")
                else:
                    cerebro = bt.Cerebro()
                    cerebro.adddata(bt.feeds.PandasData(dataname=df_stock))
                    cerebro.addstrategy(MegaStrategy, strategy_type=s_code, use_risk_mgmt=use_risk, stop_loss=stop_loss, take_profit=take_profit, **params)
                    cerebro.broker.setcash(cash)
                    
                    # 🔥 核心修复：满仓干 (95% 资金)
                    cerebro.addsizer(bt.sizers.AllInSizer, percents=95)
                    
                    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
                    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                    
                    results = cerebro.run()
                    strat = results[0]
                    
                    strat_returns = pd.Series(strat.analyzers.returns.get_analysis())
                    strat_returns.index = pd.to_datetime(strat_returns.index)
                    
                    if df_bench is not None and not df_bench.empty and 'close' in df_bench.columns:
                        bench_returns = df_bench['close'].pct_change().fillna(0)
                        bench_returns.index = pd.to_datetime(bench_returns.index)
                        bench_returns = bench_returns.reindex(strat_returns.index).fillna(0)
                    else:
                        bench_returns = None

                    final_cash = cerebro.broker.getvalue()
                    total_return = (final_cash - cash) / cash
                    max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
                    
                    st.subheader("核心成绩单")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("最终资产", f"${final_cash:,.0f}")
                    c2.metric("总收益率", f"{total_return*100:.2f}%", delta_color="normal" if total_return > 0 else "inverse")
                    
                    if bench_returns is not None:
                        bench_total = (1 + bench_returns).cumprod().iloc[-1] - 1
                        alpha = total_return - bench_total
                        c3.metric("跑赢大盘", f"{alpha*100:.2f}%", delta=f"{alpha*100:.2f}%")
                    else:
                        c3.metric("跑赢大盘", "无数据")
                    c4.metric("最惨亏损", f"{max_dd:.2f}%")

                    st.subheader("资金走势图")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    cum_strat = (1 + strat_returns).cumprod()
                    ax.plot(cum_strat.index, cum_strat, label="我的策略", color='#2962FF', linewidth=2)
                    if bench_returns is not None:
                        cum_bench = (1 + bench_returns).cumprod()
                        ax.plot(cum_bench.index, cum_bench, label=f"市场基准 ({benchmark_name})", color='#B0BEC5', linestyle='--', alpha=0.8)
                        ax.fill_between(cum_strat.index, cum_strat, cum_bench, where=(cum_strat > cum_bench), color='#00C853', alpha=0.1)
                        ax.fill_between(cum_strat.index, cum_strat, cum_bench, where=(cum_strat <= cum_bench), color='#D50000', alpha=0.1)
                    ax.legend()
                    st.pyplot(fig)

                    with st.expander("📝 查看每一笔买卖记录"):
                        if strat.log_list:
                            st.dataframe(pd.DataFrame(strat.log_list, columns=["交易详情"]), use_container_width=True)
                        else:
                            st.info("无交易记录")
                    
                    st.session_state['strat_returns'] = strat_returns
                    st.session_state['bench_returns'] = bench_returns
                    st.session_state['ticker'] = ticker

    with tab2:
        if 'strat_returns' in st.session_state:
            st.markdown("### 深度体检报告")
            try:
                report_file = "qs_report.html"
                qs.reports.html(st.session_state['strat_returns'], benchmark=st.session_state['bench_returns'], output=report_file, title=f"{st.session_state['ticker']} 策略分析", download_filename=report_file)
                with open(report_file, 'r', encoding='utf-8') as f: report_html = f.read()
                components.html(report_html, height=1000, scrolling=True)
            except: st.error("报告生成失败")

    with tab3: show_strategy_wiki()
    with tab4: show_user_guide()

if __name__ == '__main__':
    main()
