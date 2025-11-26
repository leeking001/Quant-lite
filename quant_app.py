import streamlit as st
import datetime
import pandas as pd
import yfinance as yf
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
# 1. 策略引擎 (保持不变)
# ==========================================
class PortfolioStrategy(bt.Strategy):
    params = (
        ('strategy_type', 'SMA'), 
        ('use_risk_mgmt', False), 
        ('stop_loss', 0.05),      
        ('take_profit', 0.10),
        ('pfast', 10), ('pslow', 30),
        ('rsi_period', 14), ('rsi_low', 30), ('rsi_high', 70),
        ('boll_period', 20), ('boll_dev', 2.0),
        ('turtle_period', 20),
        ('mean_period', 20),
        ('builder_indicator', 'Close'),
        ('builder_operator', '>'),
        ('builder_threshold', 'SMA'),
        ('builder_param', 20),
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
            elif self.params.strategy_type == 'Turtle':
                self.inds[d] = {'high': bt.indicators.Highest(d.high(-1), period=self.params.turtle_period), 'low': bt.indicators.Lowest(d.low(-1), period=self.params.turtle_period)}
            elif self.params.strategy_type == 'MeanRev':
                sma = bt.indicators.SMA(d, period=self.params.mean_period)
                self.inds[d] = {'sma': sma, 'dist': (d.close - sma) / sma}
            elif self.params.strategy_type == 'Builder':
                if self.params.builder_indicator == 'RSI': self.inds[d] = {'left': bt.indicators.RSI(d, period=14)}
                else: self.inds[d] = {'left': d.close}
                if self.params.builder_threshold == 'SMA': self.inds[d]['right'] = bt.indicators.SMA(d, period=self.params.builder_param)
                else: self.inds[d]['right'] = float(self.params.builder_param)

    def next(self):
        target_pct = 0.95 / len(self.datas)
        for d in self.datas:
            pos = self.getposition(d).size
            # 风控
            if pos != 0 and self.params.use_risk_mgmt:
                buy_price = self.getposition(d).price
                pnl_pct = (d.close[0] - buy_price) / buy_price
                if pnl_pct <= -self.params.stop_loss: self.close(data=d); continue 
                if pnl_pct >= self.params.take_profit: self.close(data=d); continue

            # 策略逻辑
            signal_buy = False
            signal_sell = False

            if self.params.strategy_type == 'Builder':
                left_val = self.inds[d]['left'][0]
                right_val = self.inds[d]['right'][0] if hasattr(self.inds[d]['right'], '__getitem__') else self.inds[d]['right']
                op = self.params.builder_operator
                condition = (left_val > right_val) if op == '>' else (left_val < right_val)
                if condition: signal_buy = True
                else: signal_sell = True
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

            if not pos and signal_buy: self.order_target_percent(data=d, target=target_pct)
            elif pos and signal_sell: self.close(data=d)

# ==========================================
# 2. 数据获取 (V3.3 终极修复版：Yahoo 智能后缀)
# ==========================================
@st.cache_data(ttl=3600)
def get_multiple_data(source, tickers_list, start_date, end_date):
    data_dict = {}
    bench_df = pd.DataFrame()
    
    # 1. 获取股票数据
    for ticker in tickers_list:
        ticker = ticker.strip()
        if not ticker: continue
        
        # --- A股智能后缀处理 ---
        # 如果用户选了A股，且输入的是纯数字，自动加后缀
        search_ticker = ticker
        if source == "A股" and ticker.isdigit():
            if ticker.startswith('6'):
                search_ticker = f"{ticker}.SS" # 上海
            elif ticker.startswith('0') or ticker.startswith('3'):
                search_ticker = f"{ticker}.SZ" # 深圳
            elif ticker.startswith('4') or ticker.startswith('8'):
                search_ticker = f"{ticker}.BJ" # 北京
        # ---------------------

        try:
            # 统一使用 Yahoo Finance，因为它在云端最稳定
            df = yf.download(search_ticker, start=start_date, end=end_date, progress=False, timeout=10)
            
            # 数据清洗
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.lower()
            
            # 修复时区问题 (Yahoo 返回的是带时区的，Backtrader 不喜欢)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            if not df.empty: 
                data_dict[ticker] = df # 存回字典时用原始代码(如600519)做key，方便显示
        except: 
            pass

    # 2. 获取基准数据
    try:
        bench_ticker = "^GSPC" # 默认标普500
        if source == "A股":
            bench_ticker = "000300.SS" # 沪深300 (Yahoo代码)
            
        bench_df = yf.download(bench_ticker, start=start_date, end=end_date, progress=False)
        
        if isinstance(bench_df.columns, pd.MultiIndex): 
            bench_df.columns = bench_df.columns.get_level_values(0)
        bench_df.columns = bench_df.columns.str.lower()
        
        if bench_df.index.tz is not None:
            bench_df.index = bench_df.index.tz_localize(None)
            
    except: pass

    return data_dict, bench_df

# ==========================================
# 3. 文案内容
# ==========================================
def show_manual():
    st.markdown("""
    ### 📘 新手保姆级手册
    
    **第一步：准备工作**
    1.  **选市场**: 玩茅台选 **A股**，玩苹果/特斯拉选 **美股/港股**。
    2.  **输代码**: 
        *   A股直接输数字，如 `600519`。
        *   美股输字母，如 `AAPL`。
        *   支持多只！用逗号隔开，例如 `600519, 000858`。
    3.  **本金**: 建议填 **100,000** 以上，否则可能买不起一手高价股。

    **第二步：选择策略**
    *   **稳健型**: 推荐 **双均线 (SMA)** 或 **布林带**。
    *   **激进型**: 推荐 **海龟交易** 或 **RSI**。
    *   **DIY型**: 使用 **策略工厂** 自己拼逻辑。

    **第三步：风控 (必看!)**
    *   **止损**: 亏了多少比例强制卖出。建议 **5%**。
    *   **止盈**: 赚了多少比例强制卖出。建议 **15%**。
    """)

def show_wiki():
    st.markdown("""
    ### 🧠 策略百科全书 (共5种)

    #### 1. 双均线 (SMA Cross)
    *   **原理**: "金叉买，死叉卖"。快线（如10日）上穿慢线（如30日）买入。
    *   **适用**: **大趋势行情**。
    *   **缺点**: 震荡市会频繁打脸亏损。

    #### 2. RSI (相对强弱)
    *   **原理**: "物极必反"。分数低（<30）说明超卖，买入；分数高（>70）说明超买，卖出。
    *   **适用**: **震荡市**（箱体波动）。
    *   **缺点**: 大牛市中会过早卖出，踏空后续涨幅。

    #### 3. 布林带 (Bollinger Bands)
    *   **原理**: "回归中枢"。股价通常在通道内运行。跌破下轨买入，突破上轨卖出。
    *   **适用**: **震荡修复行情**。
    *   **缺点**: 在单边暴跌中，股价会沿着下轨一直跌，导致过早抄底被套。

    #### 4. 海龟交易 (Turtle)
    *   **原理**: "追涨杀跌"。突破过去 N 天的最高价，说明新一轮趋势开始了，果断追涨。
    *   **适用**: **大牛市、大熊市**。
    *   **缺点**: 假突破。看着突破了，买进去立马回调。

    #### 5. 均值回归 (Mean Reversion)
    *   **原理**: "橡皮筋理论"。价格偏离均线太远（如跌了5%），总会弹回来。
    *   **适用**: **急涨急跌**后的反弹。
    """)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    st.set_page_config(page_title="量化交易模拟器", layout="wide", page_icon="📈", initial_sidebar_state="collapsed")
    st.title("📈 量化交易模拟器")
    
    tab_sim, tab_manual, tab_wiki = st.tabs(["🚀 开始模拟", "📘 新手手册", "🧠 策略百科"])

    with tab_manual: show_manual()
    with tab_wiki: show_wiki()

    with tab_sim:
        col_input, col_action = st.columns([3, 1])
        with col_input:
            default_tickers = "AAPL, MSFT, NVDA"
            data_source = st.selectbox("市场", ["美股/港股", "A股"])
            tickers_input = st.text_area("股票代码 (用逗号隔开)", value=default_tickers, height=68)

        with st.expander("⚙️ 策略与风控配置", expanded=True):
            c1, c2 = st.columns(2)
            start_date = c1.date_input("开始日期", datetime.date(2021, 1, 1))
            cash = c2.number_input("初始本金", 100000, help="建议 10万 以上")
            
            strat_map = {
                "🛠️ 零代码策略工厂": "Builder",
                "双均线 (趋势)": "SMA", 
                "RSI (反转)": "RSI", 
                "布林带 (通道)": "Bollinger",
                "海龟交易 (突破)": "Turtle",
                "均值回归 (抄底)": "MeanRev"
            }
            s_name = st.selectbox("选择策略模型", list(strat_map.keys()))
            s_code = strat_map[s_name]
            
            params = {}
            if s_code == "Builder":
                st.info("🏗️ **策略工厂**：当 [指标] [比较] [阈值] 时买入")
                bc1, bc2, bc3, bc4 = st.columns([2, 1, 2, 2])
                with bc1:
                    b_ind = st.selectbox("指标", ["收盘价", "RSI"])
                    params['builder_indicator'] = 'RSI' if 'RSI' in b_ind else 'Close'
                with bc2: params['builder_operator'] = st.selectbox("比较", [">", "<"])
                with bc3:
                    b_thres = st.selectbox("阈值类型", ["均线 (SMA)", "固定数值"])
                    params['builder_threshold'] = 'SMA' if 'SMA' in b_thres else 'Value'
                with bc4:
                    def_val = 20 if params['builder_threshold'] == 'SMA' else (30 if params['builder_indicator'] == 'RSI' else 100)
                    params['builder_param'] = st.number_input("参数值", 0, 10000, def_val)

            elif s_code == "SMA":
                params['pfast'] = st.slider("快线周期", 5, 30, 10)
                params['pslow'] = st.slider("慢线周期", 20, 60, 30)
            elif s_code == "RSI":
                params['rsi_period'] = 14
                params['rsi_low'] = st.slider("超卖 (买)", 10, 40, 30)
                params['rsi_high'] = st.slider("超买 (卖)", 60, 90, 70)
            elif s_code == "Bollinger":
                params['boll_period'] = st.slider("周期", 10, 50, 20)
                params['boll_dev'] = st.slider("标准差倍数", 1.0, 3.0, 2.0)
            elif s_code == "Turtle":
                params['turtle_period'] = st.slider("突破周期", 10, 60, 20)
            elif s_code == "MeanRev":
                params['mean_period'] = st.slider("均线周期", 10, 50, 20)

            st.divider()
            use_risk = st.checkbox("开启自动止盈止损", value=True)
            stop_loss = st.slider("止损 (Stop Loss)", 1, 20, 5) / 100.0
            take_profit = st.slider("止盈 (Take Profit)", 5, 50, 15) / 100.0

        run_btn = st.button("🚀 开始回测", type="primary", use_container_width=True)

        if run_btn:
            ticker_list = [t.strip() for t in tickers_input.split(',') if t.strip()]
            if not ticker_list: st.error("请输入代码")
            else:
                with st.spinner("正在计算..."):
                    data_dict, df_bench = get_multiple_data(data_source, ticker_list, start_date, datetime.date.today())
                    
                    if not data_dict: st.error("数据获取失败。如果是A股，请确认代码正确（如 600519）。")
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
                        
                        with st.expander("📊 详细数据报告 (手机友好版)"):
                            try:
                                metrics = qs.reports.metrics(strat_returns, benchmark=bench_returns, mode='basic', display=False)
                                st.dataframe(metrics, use_container_width=True)
                                
                                report_file = "qs_report.html"
                                qs.reports.html(strat_returns, benchmark=bench_returns, output=report_file, title="Report", download_filename=report_file)
                                with open(report_file, 'r', encoding='utf-8') as f:
                                    st.download_button("📥 下载完整图表报告 (电脑端查看)", f, file_name="report.html")
                            except Exception as e:
                                st.error(f"指标计算失败: {e}")

if __name__ == '__main__':
    main()
