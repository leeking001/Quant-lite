import streamlit as st
import datetime
import pandas as pd
import yfinance as yf
import akshare as ak
import numpy as np
import time
import quantstats as qs
import streamlit.components.v1 as components

# ==========================================
# 0. 系统配置与 BigQuant 风格 CSS
# ==========================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import warnings
import matplotlib.dates

if not hasattr(matplotlib.dates, 'warnings'):
    matplotlib.dates.warnings = warnings

# 设置 Matplotlib 为深色模式，适配 BigQuant 风格
plt.style.use('dark_background') 

import backtrader as bt

def inject_custom_css():
    st.markdown("""
    <style>
        /* 全局背景色 - 深空灰 */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* 侧边栏背景 - 更深的灰 */
        [data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }

        /* 标题样式 */
        h1, h2, h3 {
            color: #58A6FF !important; /* BigQuant 蓝 */
            font-family: 'Helvetica Neue', sans-serif;
        }

        /* 按钮样式 - 极客蓝 */
        div.stButton > button {
            background-color: #238636;
            color: white;
            border: none;
            border-radius: 4px;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #2EA043;
        }

        /* 指标卡片样式 */
        [data-testid="stMetricValue"] {
            font-size: 24px;
            color: #3FB950 !important; /* 涨幅绿 */
        }
        [data-testid="stMetricLabel"] {
            color: #8B949E;
        }
        
        /* Tab 样式 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            color: #8B949E;
        }
        .stTabs [aria-selected="true"] {
            background-color: transparent;
            color: #58A6FF;
            border-bottom: 2px solid #58A6FF;
        }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 策略类 (逻辑保持不变)
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
                self.log(f'🟢 买入: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'🔴 卖出: {order.executed.price:.2f} | 盈亏: {order.executed.pnl:.2f}')
            self.order = None

    def next(self):
        if self.order: return

        # 风控逻辑
        if self.position and self.params.use_risk_mgmt:
            buy_price = self.position.price
            current_price = self.dataclose[0]
            pnl_pct = (current_price - buy_price) / buy_price

            if pnl_pct <= -self.params.stop_loss:
                self.log(f'🛡️ 止损触发: {pnl_pct*100:.2f}%')
                self.close()
                return
            if pnl_pct >= self.params.take_profit:
                self.log(f'💰 止盈触发: {pnl_pct*100:.2f}%')
                self.close()
                return

        # 策略逻辑
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
            
            if isinstance(stock_df.columns, pd.MultiIndex):
                stock_df.columns = stock_df.columns.get_level_values(0)
            stock_df.columns = stock_df.columns.str.lower()
            
            bench_ticker = "^GSPC" 
            for i in range(max_retries):
                try:
                    if i > 0: time.sleep(1)
                    bench_df = yf.download(bench_ticker, start=start_date, end=end_date, progress=False, timeout=10)
                    if not bench_df.empty: break
                except: pass
            
            if isinstance(bench_df.columns, pd.MultiIndex):
                bench_df.columns = bench_df.columns.get_level_values(0)
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
# 3. 文案内容 (新手指南 & 策略解释)
# ==========================================
def show_user_guide():
    st.markdown("""
    ### 🚀 新手快速入门指南
    
    欢迎来到 **Quant Pro**。这是一个模仿专业机构的量化回测沙箱。请按照以下步骤操作：

    #### 第一步：选择战场 (Market)
    *   在左侧侧边栏选择 **数据来源**。
    *   **美股**: 输入代码如 `AAPL` (苹果), `NVDA` (英伟达), `BTC-USD` (比特币)。
    *   **A股**: 输入6位数字代码，如 `600519` (茅台), `300750` (宁德时代)。

    #### 第二步：挑选武器 (Strategy)
    *   在 **模型** 下拉框中选择一个策略。
    *   *不知道选哪个？请点击上方的“🧠 策略百科”标签页学习。*

    #### 第三步：设置防线 (Risk Control)
    *   展开 **🛡️ 风控** 面板。
    *   **止损 (Stop Loss)**: 建议设置为 5%-10%。这是你的保命线。
    *   **止盈 (Take Profit)**: 建议设置为 10%-20%。这是你的落袋线。

    #### 第四步：解读战报 (Analysis)
    *   点击 **🚀 运行回测**。
    *   **Alpha**: 如果是正数，说明你跑赢了大盘（真本事）。
    *   **Sharpe**: 如果大于 1.0，说明策略性价比不错。
    *   **QuantStats 报告**: 在“📑 专业报告”中查看月度热力图，看看哪个月最赚钱。
    """)

def show_strategy_wiki():
    st.markdown("""
    ### 🧠 量化策略百科全书

    #### 1. SMA 双均线策略 (Trend Following)
    *   **原理**: "金叉买，死叉卖"。利用短期趋势和长期趋势的交叉来判断方向。
    *   **适用**: 趋势明显的牛市或熊市。
    *   **缺点**: 在震荡市（横盘）中会频繁打脸，导致不断止损。
    
    #### 2. RSI 相对强弱策略 (Mean Reversion)
    *   **原理**: "物极必反"。RSI < 30 认为超卖（太便宜了，买！），RSI > 70 认为超买（太贵了，卖！）。
    *   **适用**: 震荡市，箱体波动。
    *   **缺点**: 在单边暴涨行情中，RSI 会一直钝化在超买区，导致过早卖出踏空。

    #### 3. MACD 指数平滑策略 (Momentum)
    *   **原理**: "指标之王"。结合了动量和趋势。快线(DIF)上穿慢线(DEA)为买入信号。
    *   **适用**: 捕捉中长期的趋势反转点。
    *   **缺点**: 信号有滞后性，通常行情走了一段才发出信号。

    #### 4. Bollinger 布林带策略 (Volatility)
    *   **原理**: 价格总是围绕均线波动。跌破下轨认为被低估，突破上轨认为被高估。
    *   **适用**: 寻找价格的极端位置进行反向操作。
    
    #### 5. Momentum 动量策略 (Inertia)
    *   **原理**: "强者恒强"。如果过去N天涨得好，假设未来还会涨。
    *   **适用**: 热门股、妖股。
    """)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    st.set_page_config(page_title="Quant Pro AI", layout="wide", page_icon="⚡")
    
    # 注入 BigQuant 风格 CSS
    inject_custom_css()

    st.title("⚡ Quant Pro AI 量化平台")
    st.caption("Professional Quantitative Trading System | Powered by Backtrader & QuantStats")

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("🎛️ 控制台 (Control)")
        
        st.subheader("1. 市场数据")
        data_source = st.selectbox("数据源", ["美股/港股 (Yahoo)", "A股 (AkShare)"])
        if data_source == "美股/港股 (Yahoo)":
            ticker = st.text_input("代码 (Ticker)", "AAPL")
            benchmark_name = "S&P 500"
        else:
            ticker = st.text_input("代码 (Code)", "600519")
            benchmark_name = "CSI 300"
            
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start", datetime.date(2021, 1, 1))
        end_date = col2.date_input("End", datetime.date.today())
        cash = st.number_input("本金 (Cash)", 100000)

        st.subheader("2. 策略模型")
        strat_map = {"SMA (双均线)": "SMA", "RSI (相对强弱)": "RSI", "MACD (指数平滑)": "MACD", "Bollinger (布林带)": "Bollinger", "Momentum (动量)": "Momentum"}
        s_name = st.selectbox("选择模型", list(strat_map.keys()))
        s_code = strat_map[s_name]

        with st.expander("🛡️ 风控参数 (Risk)", expanded=False):
            use_risk = st.checkbox("开启止盈止损", value=True)
            stop_loss = st.slider("止损 (Stop Loss)", 1, 20, 5) / 100.0
            take_profit = st.slider("止盈 (Take Profit)", 5, 50, 15) / 100.0

        # 动态参数
        params = {}
        if s_code == "SMA":
            params['pfast'] = st.slider("Fast MA", 5, 50, 10)
            params['pslow'] = st.slider("Slow MA", 20, 100, 30)
        elif s_code == "RSI":
            params['rsi_period'] = st.slider("Period", 5, 30, 14)
            params['rsi_low'] = st.slider("Low", 10, 40, 30)
            params['rsi_high'] = st.slider("High", 60, 90, 70)
        # ... 其他参数省略，逻辑同前 ...

        run_btn = st.button("🚀 开始回测 (Run Backtest)", type="primary")

    # --- 主界面 Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 回测看板", "📑 专业报告", "🧠 策略百科", "📖 新手指南"])

    # --- Tab 1: 回测看板 ---
    with tab1:
        if run_btn:
            with st.spinner("正在连接量化引擎..."):
                df_stock, df_bench = get_data_with_benchmark(data_source, ticker, start_date, end_date)
                
                if df_stock is None or df_stock.empty:
                    st.error("❌ 数据获取失败，请检查代码或网络。")
                else:
                    # 运行回测
                    cerebro = bt.Cerebro()
                    cerebro.adddata(bt.feeds.PandasData(dataname=df_stock))
                    cerebro.addstrategy(MegaStrategy, strategy_type=s_code, use_risk_mgmt=use_risk, stop_loss=stop_loss, take_profit=take_profit, **params)
                    cerebro.broker.setcash(cash)
                    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
                    
                    results = cerebro.run()
                    strat = results[0]
                    
                    # 数据处理
                    strat_returns = pd.Series(strat.analyzers.returns.get_analysis())
                    strat_returns.index = pd.to_datetime(strat_returns.index)
                    
                    if df_bench is not None and not df_bench.empty and 'close' in df_bench.columns:
                        bench_returns = df_bench['close'].pct_change().fillna(0)
                        bench_returns.index = pd.to_datetime(bench_returns.index)
                        bench_returns = bench_returns.reindex(strat_returns.index).fillna(0)
                    else:
                        bench_returns = None

                    # 计算指标
                    final_cash = cerebro.broker.getvalue()
                    total_return = (final_cash - cash) / cash
                    
                    # 1. 核心指标卡片 (BigQuant 风格)
                    st.markdown("### 核心绩效 (Key Metrics)")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("最终资产", f"${final_cash:,.0f}")
                    c2.metric("策略收益", f"{total_return*100:.2f}%", delta_color="normal" if total_return > 0 else "inverse")
                    
                    if bench_returns is not None:
                        bench_total = (1 + bench_returns).cumprod().iloc[-1] - 1
                        alpha = total_return - bench_total
                        c3.metric(f"基准收益 ({benchmark_name})", f"{bench_total*100:.2f}%")
                        c4.metric("Alpha (超额)", f"{alpha*100:.2f}%", delta=f"{alpha*100:.2f}%")
                    else:
                        c3.metric("基准收益", "N/A")
                        c4.metric("Alpha", "N/A")

                    # 2. 净值曲线 (深色模式)
                    st.markdown("### 净值走势 (Equity Curve)")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    # 设置背景色
                    fig.patch.set_facecolor('#0E1117')
                    ax.set_facecolor('#0E1117')
                    
                    cum_strat = (1 + strat_returns).cumprod()
                    ax.plot(cum_strat.index, cum_strat, label="Strategy", color='#00E676', linewidth=2)
                    
                    if bench_returns is not None:
                        cum_bench = (1 + bench_returns).cumprod()
                        ax.plot(cum_bench.index, cum_bench, label="Benchmark", color='#58A6FF', linestyle='--', alpha=0.6)
                        ax.fill_between(cum_strat.index, cum_strat, cum_bench, where=(cum_strat > cum_bench), color='#00E676', alpha=0.1)
                        ax.fill_between(cum_strat.index, cum_strat, cum_bench, where=(cum_strat <= cum_bench), color='#FF5252', alpha=0.1)
                    
                    ax.grid(True, color='#30363D', linestyle='--', alpha=0.5)
                    ax.tick_params(colors='#8B949E')
                    ax.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='#FAFAFA')
                    st.pyplot(fig)

                    # 3. 交易记录
                    with st.expander("📝 查看详细交易日志"):
                        if strat.log_list:
                            st.dataframe(pd.DataFrame(strat.log_list, columns=["Log"]), use_container_width=True)
                        else:
                            st.info("无交易记录")
                    
                    # 保存数据给 Tab 2 使用
                    st.session_state['strat_returns'] = strat_returns
                    st.session_state['bench_returns'] = bench_returns
                    st.session_state['ticker'] = ticker

        else:
            st.info("👈 请在左侧设置参数并点击 '开始回测'")

    # --- Tab 2: 专业报告 ---
    with tab2:
        if 'strat_returns' in st.session_state:
            st.markdown("### 深度量化分析报告 (QuantStats)")
            st.caption("Generating Wall Street grade report...")
            try:
                report_file = "qs_report.html"
                qs.reports.html(
                    st.session_state['strat_returns'], 
                    benchmark=st.session_state['bench_returns'], 
                    output=report_file, 
                    title=f"{st.session_state['ticker']} Analysis",
                    download_filename=report_file
                )
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_html = f.read()
                components.html(report_html, height=1000, scrolling=True)
            except Exception as e:
                st.error(f"报告生成失败: {e}")
        else:
            st.warning("请先在 '回测看板' 运行回测。")

    # --- Tab 3: 策略百科 ---
    with tab3:
        show_strategy_wiki()

    # --- Tab 4: 新手指南 ---
    with tab4:
        show_user_guide()

if __name__ == '__main__':
    main()
