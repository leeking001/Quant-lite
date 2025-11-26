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
# 1. 策略引擎 (支持积木式逻辑)
# ==========================================
class PortfolioStrategy(bt.Strategy):
    params = (
        ('strategy_type', 'SMA'), 
        ('use_risk_mgmt', False), 
        ('stop_loss', 0.05),      
        ('take_profit', 0.10),
        # 标准参数
        ('pfast', 10), ('pslow', 30),
        ('rsi_period', 14), ('rsi_low', 30), ('rsi_high', 70),
        ('boll_period', 20), ('boll_dev', 2.0),
        ('turtle_period', 20),
        ('mean_period', 20),
        # --- 积木策略参数 ---
        ('builder_indicator', 'Close'), # 指标: 收盘价/RSI/均线
        ('builder_operator', '>'),      # 符号: > / <
        ('builder_threshold', 'SMA'),   # 阈值: 数值/均线
        ('builder_param', 20),          # 阈值参数 (如均线周期)
    )

    def __init__(self):
        self.inds = {} 
        
        for d in self.datas:
            # --- 初始化标准策略指标 ---
            if self.params.strategy_type == 'SMA':
                sma1 = bt.indicators.SimpleMovingAverage(d, period=self.params.pfast)
                sma2 = bt.indicators.SimpleMovingAverage(d, period=self.params.pslow)
                self.inds[d] = bt.indicators.CrossOver(sma1, sma2)
            elif self.params.strategy_type == 'RSI':
                self.inds[d] = bt.indicators.RSI(d, period=self.params.rsi_period)
            elif self.params.strategy_type == 'Bollinger':
                self.inds[d] = bt.indicators.BollingerBands(d, period=self.params.boll_period, devfactor=self.params.boll_dev)
            elif self.params.strategy_type == 'Turtle':
                self.inds[d] = {'high': bt.indicators.Highest(d.high(-1), period=self.params.turtle_period), 'low': bt.indicators.Lowest(d.low(-1), period=self.params.turtle_period)}
            elif self.params.strategy_type == 'MeanRev':
                sma = bt.indicators.SMA(d, period=self.params.mean_period)
                self.inds[d] = {'sma': sma, 'dist': (d.close - sma) / sma}
            
            # --- 初始化积木策略指标 (Builder) ---
            elif self.params.strategy_type == 'Builder':
                # 1. 准备左边 (Indicator)
                if self.params.builder_indicator == 'RSI':
                    self.inds[d] = {'left': bt.indicators.RSI(d, period=14)}
                else: # Close
                    self.inds[d] = {'left': d.close}
                
                # 2. 准备右边 (Threshold)
                if self.params.builder_threshold == 'SMA':
                    self.inds[d]['right'] = bt.indicators.SMA(d, period=self.params.builder_param)
                else: # 固定数值
                    self.inds[d]['right'] = float(self.params.builder_param)

    def next(self):
        target_pct = 0.95 / len(self.datas)

        for d in self.datas:
            pos = self.getposition(d).size
            
            # 风控 (最高优先级)
            if pos != 0 and self.params.use_risk_mgmt:
                buy_price = self.getposition(d).price
                pnl_pct = (d.close[0] - buy_price) / buy_price
                if pnl_pct <= -self.params.stop_loss:
                    self.close(data=d); continue 
                if pnl_pct >= self.params.take_profit:
                    self.close(data=d); continue

            # --- 策略逻辑 ---
            signal_buy = False
            signal_sell = False

            # 1. 积木策略 (Builder)
            if self.params.strategy_type == 'Builder':
                left_val = self.inds[d]['left'][0]
                # 如果右边是指标，取值；如果是数字，直接用
                right_val = self.inds[d]['right'][0] if hasattr(self.inds[d]['right'], '__getitem__') else self.inds[d]['right']
                
                op = self.params.builder_operator
                
                # 判断逻辑
                condition = False
                if op == '>': condition = left_val > right_val
                elif op == '<': condition = left_val < right_val
                
                if condition: signal_buy = True
                else: signal_sell = True # 简单的反向逻辑：不满足买入就卖出(简化版)

            # 2. 标准策略
            elif self.params.strategy_type == 'SMA':
                if self.inds[d] > 0: signal_buy = True
                elif self.inds[d] < 0: signal_sell = True
            elif self.params.strategy_type == 'RSI':
                if self.inds[d] < self.params.rsi_low: signal_buy = True
                elif self.inds[d] > self.params.rsi_high: signal_sell = True
            elif self.params.strategy_type == 'Bollinger':
                if d.close[0] < self.inds[d].lines.bot[0]: signal_buy = True
                elif d.close[0] > self.inds[d].lines.top[0]: signal_sell = True
            elif self.params.strategy_type == 'Turtle':
                if d.close[0] > self.inds[d]['high'][0]: signal_buy = True
                elif d.close[0] < self.inds[d]['low'][0]: signal_sell = True
            elif self.params.strategy_type == 'MeanRev':
                if self.inds[d]['dist'][0] < -0.05: signal_buy = True
                elif d.close[0] >= self.inds[d]['sma'][0]: signal_sell = True

            # 执行交易
            if not pos and signal_buy:
                self.order_target_percent(data=d, target=target_pct)
            elif pos and signal_sell:
                self.close(data=d)

# ==========================================
# 2. 数据获取 (保持 V2.1)
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
# 3. 文案与教程
# ==========================================
def show_manual():
    st.markdown("""
    ## 📘 新手保姆级手册
    
    ### 第一阶段：准备工作
    1.  **选市场**: 
        *   如果你想玩 **茅台、宁德时代**，请选 **A股**。
        *   如果你想玩 **苹果、特斯拉、比特币**，请选 **美股/港股**。
    2.  **输代码**: 
        *   支持一次测多只！用逗号隔开。
        *   例如: `600519, 000858` (茅台+五粮液)。
    
    ### 第二阶段：选择武器 (策略)
    *   **小白推荐**: 先用 **双均线 (SMA)**。这是最经典的策略，容易理解。
    *   **进阶玩家**: 试试 **海龟交易**，这是捕捉大牛股的神器。
    *   **高玩**: 使用 **🛠️ 策略工厂**，自己定义买卖逻辑！

    ### 第三阶段：风控 (最重要!)
    *   **止损 (Stop Loss)**: 类似于“保险丝”。比如设 5%，亏了 5% 自动断电（卖出），防止房子烧光（本金亏光）。
    *   **止盈 (Take Profit)**: 类似于“收网”。赚够了就跑，防止煮熟的鸭子飞了。

    ### 第四阶段：看懂结果
    *   **资金曲线**: 蓝线是你，灰线是大盘。蓝线在灰线上面，说明你牛；在下面，说明你菜。
    *   **Alpha**: 正数=牛，负数=菜。
    *   **最大回撤**: 越小越好。如果回撤 -50%，说明你资产腰斩过，心脏受得了吗？
    """)

def show_wiki():
    st.markdown("""
    ## 🧠 策略百科全书

    ### 1. 双均线 (SMA Cross)
    *   **原理**: 两根线，一快一慢。快线上穿慢线叫“金叉”（买），下穿叫“死叉”（卖）。
    *   **适用**: **大牛市、大熊市**。
    *   **缺点**: **震荡市**。股价横盘时，两根线会反复缠绕，导致你频繁买卖，亏手续费。
    *   **参数**: 
        *   *快线周期*: 灵敏度。越小越灵敏，但也越容易被骗。
        *   *慢线周期*: 稳定性。越大越稳，但信号来得越晚。

    ### 2. RSI (相对强弱)
    *   **原理**: 测量市场的情绪。0-100分。低于30分大家恐慌（抄底），高于70分大家狂热（逃顶）。
    *   **适用**: **震荡市**。股价在一个箱体里来回跳。
    *   **缺点**: **大牛市**。牛市里 RSI 会一直高于 70，如果你卖了，就踏空了后面的大涨。

    ### 3. 海龟交易 (Turtle)
    *   **原理**: 价格突破了过去 N 天的最高价，说明新趋势来了，无脑追涨！
    *   **适用**: **趋势行情**。
    *   **缺点**: 假突破。看着突破了，买进去立马跌回来。

    ### 4. 均值回归 (Mean Reversion)
    *   **原理**: 橡皮筋理论。价格拉得离均线太远，总会弹回来。
    *   **适用**: **急涨急跌**后的修复行情。
    """)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    st.set_page_config(page_title="量化工厂 V3.0", layout="wide", page_icon="🏭", initial_sidebar_state="collapsed")
    st.title("🏭 量化策略工厂 (V3.0)")
    
    # --- 顶部输入区 ---
    col_input, col_action = st.columns([3, 1])
    with col_input:
        if 'demo_mode' not in st.session_state: st.session_state.demo_mode = False
        default_tickers = "AAPL, MSFT, NVDA"
        default_source = "美股/港股"
        
        if st.button("🎲 随机演示 (Demo)"):
            st.session_state.demo_mode = True
            demos = [("美股/港股", "AAPL, TSLA, AMZN"), ("A股", "600519, 000858, 600036")]
            choice = random.choice(demos)
            default_source = choice[0]
            default_tickers = choice[1]

        data_source = st.selectbox("市场", ["美股/港股", "A股"], index=0 if default_source=="美股/港股" else 1)
        tickers_input = st.text_area("股票代码", value=default_tickers, height=68, help="输入代码，用逗号隔开")

    # --- 策略配置区 (核心升级) ---
    with st.expander("⚙️ 策略配置 (点击展开)", expanded=True):
        c1, c2 = st.columns(2)
        start_date = c1.date_input("开始日期", datetime.date(2021, 1, 1))
        cash = c2.number_input("初始本金", 100000, help="建议 10万 以上")
        
        # 策略选择
        strat_map = {
            "🛠️ 零代码策略工厂 (自定义)": "Builder",
            "双均线 (趋势)": "SMA", 
            "RSI (反转)": "RSI", 
            "布林带 (通道)": "Bollinger",
            "海龟交易 (突破)": "Turtle",
            "均值回归 (抄底)": "MeanRev"
        }
        s_name = st.selectbox("选择策略模型", list(strat_map.keys()))
        s_code = strat_map[s_name]
        
        # --- 动态参数区 ---
        params = {}
        
        if s_code == "Builder":
            st.info("🏗️ **策略工厂**：用自然语言搭建你的策略！")
            bc1, bc2, bc3, bc4 = st.columns([2, 1, 2, 2])
            
            with bc1:
                b_ind = st.selectbox("当...", ["收盘价 (Price)", "RSI指标"], help="选择作为判断依据的指标")
                params['builder_indicator'] = 'RSI' if 'RSI' in b_ind else 'Close'
            
            with bc2:
                params['builder_operator'] = st.selectbox("比较", [">", "<"], help="大于还是小于")
            
            with bc3:
                b_thres = st.selectbox("目标...", ["均线 (SMA)", "固定数值"], help="和什么比较？")
                params['builder_threshold'] = 'SMA' if 'SMA' in b_thres else 'Value'
            
            with bc4:
                if params['builder_threshold'] == 'SMA':
                    params['builder_param'] = st.number_input("均线周期", 5, 200, 20, help="例如 20日均线")
                    desc = f"当 **{b_ind}** {params['builder_operator']} **{params['builder_param']}日均线** 时 -> **买入**"
                else:
                    def_val = 30 if params['builder_indicator'] == 'RSI' else 100
                    params['builder_param'] = st.number_input("数值", 0, 10000, def_val)
                    desc = f"当 **{b_ind}** {params['builder_operator']} **{params['builder_param']}** 时 -> **买入**"
            
            st.success(f"📝 当前策略逻辑：{desc} (反之卖出)")

        elif s_code == "SMA":
            st.caption("经典趋势策略：快线上穿慢线买入。")
            params['pfast'] = st.slider("快线周期", 5, 30, 10, help="灵敏度高")
            params['pslow'] = st.slider("慢线周期", 20, 60, 30, help="稳定性高")
        elif s_code == "RSI":
            st.caption("经典反转策略：低买高卖。")
            params['rsi_period'] = 14
            params['rsi_low'] = st.slider("超卖阈值 (买)", 10, 40, 30)
            params['rsi_high'] = st.slider("超买阈值 (卖)", 60, 90, 70)
        elif s_code == "Turtle":
            st.caption("海龟法则：突破过去 N 天最高价。")
            params['turtle_period'] = st.slider("突破周期", 10, 60, 20)
        elif s_code == "MeanRev":
            st.caption("均值回归：价格偏离均线过大。")
            params['mean_period'] = st.slider("均线周期", 10, 50, 20)

        # 风控
        st.divider()
        st.caption("🛡️ **风控设置** (建议开启)")
        use_risk = st.checkbox("开启自动止盈止损", value=True)
        stop_loss = st.slider("止损 (Stop Loss)", 1, 20, 5, help="亏损达到此比例自动卖出") / 100.0
        take_profit = st.slider("止盈 (Take Profit)", 5, 50, 15, help="盈利达到此比例自动卖出") / 100.0

    run_btn = st.button("🚀 开始回测", type="primary", use_container_width=True)

    # --- 结果展示 ---
    tab1, tab2, tab3 = st.tabs(["📊 回测结果", "📘 新手手册", "🧠 策略百科"])

    with tab1:
        if run_btn or st.session_state.demo_mode:
            st.session_state.demo_mode = False
            ticker_list = [t.strip() for t in tickers_input.split(',') if t.strip()]
            
            if not ticker_list: st.error("请输入代码")
            else:
                with st.spinner("正在构建策略工厂..."):
                    data_dict, df_bench = get_multiple_data(data_source, ticker_list, start_date, datetime.date.today())
                    
                    if not data_dict: st.error("数据获取失败")
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
                        ax.plot(cum_strat.index, cum_strat, color='#2962FF', linewidth=2, label='策略')
                        if bench_returns is not None:
                            cum_bench = (1 + bench_returns).cumprod()
                            ax.plot(cum_bench.index, cum_bench, color='gray', linestyle='--', alpha=0.6, label='基准')
                        ax.legend()
                        st.pyplot(fig)
                        
                        with st.expander("📊 详细报告 (QuantStats)"):
                            try:
                                report_file = "qs_mobile.html"
                                qs.reports.html(strat_returns, benchmark=bench_returns, output=report_file, title="Report", download_filename=report_file)
                                with open(report_file, 'r', encoding='utf-8') as f: components.html(f.read(), height=600, scrolling=True)
                            except: pass

    with tab2: show_manual()
    with tab3: show_wiki()

if __name__ == '__main__':
    main()
