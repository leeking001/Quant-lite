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
# 1. 策略引擎 (回测用)
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
            if len(d) < self.params.pslow: continue
            pos = self.getposition(d).size
            
            if pos != 0 and self.params.use_risk_mgmt:
                buy_price = self.getposition(d).price
                if buy_price > 0:
                    pnl_pct = (d.close[0] - buy_price) / buy_price
                    if pnl_pct <= -self.params.stop_loss: self.close(data=d); continue 
                    if pnl_pct >= self.params.take_profit: self.close(data=d); continue

            signal_buy = False
            signal_sell = False
            try:
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
            except: continue

            if not pos and signal_buy: self.order_target_percent(data=d, target=target_pct)
            elif pos and signal_sell: self.close(data=d)

# ==========================================
# 2. 数据获取
# ==========================================
@st.cache_data(ttl=3600)
def get_multiple_data(source, tickers_list, start_date, end_date):
    data_dict = {}
    bench_df = pd.DataFrame()
    
    for ticker in tickers_list:
        ticker = ticker.strip()
        if not ticker: continue
        search_ticker = ticker
        if source == "A股" and ticker.isdigit():
            if ticker.startswith('6'): search_ticker = f"{ticker}.SS"
            elif ticker.startswith('0') or ticker.startswith('3'): search_ticker = f"{ticker}.SZ"
            elif ticker.startswith('4') or ticker.startswith('8'): search_ticker = f"{ticker}.BJ"

        try:
            df = yf.download(search_ticker, start=start_date, end=end_date, progress=False, timeout=10)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.lower()
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            if not df.empty: data_dict[ticker] = df 
        except: pass

    try:
        bench_ticker = "^GSPC" 
        if source == "A股": bench_ticker = "000300.SS"
        bench_df = yf.download(bench_ticker, start=start_date, end=end_date, progress=False)
        if isinstance(bench_df.columns, pd.MultiIndex): bench_df.columns = bench_df.columns.get_level_values(0)
        bench_df.columns = bench_df.columns.str.lower()
        if bench_df.index.tz is not None: bench_df.index = bench_df.index.tz_localize(None)
    except: pass

    return data_dict, bench_df

# ==========================================
# 3. 智能顾问逻辑
# ==========================================
def generate_strategy_report(ret_pct, max_dd, alpha, sharpe):
    score = 0
    if ret_pct > 0: score += 20
    if ret_pct > 0.2: score += 10
    if alpha > 0: score += 20
    if max_dd > -20: score += 20 
    if sharpe and sharpe > 1.0: score += 30
    
    if score >= 80: stars = "⭐⭐⭐⭐⭐ (卓越)"
    elif score >= 60: stars = "⭐⭐⭐⭐ (优秀)"
    elif score >= 40: stars = "⭐⭐⭐ (良好)"
    elif score >= 20: stars = "⭐⭐ (及格)"
    else: stars = "⭐ (失败)"

    verdict = ""
    advice = []
    
    if ret_pct <= 0:
        verdict = "❌ **策略失效**"
        advice.append("策略在测试期内是亏损的，切勿实盘。")
        advice.append("尝试更换策略模型（如从趋势改为反转）。")
    elif ret_pct < 0.1: 
        verdict = "⚠️ **收益微薄**"
        advice.append("收益率甚至不如买理财，性价比低。")
        advice.append("可能是交易太频繁导致手续费过高，尝试调大均线周期。")
    else:
        verdict = "✅ **策略有效**"
        
    if max_dd < -30:
        verdict = "⚠️ **风险过高**"
        advice.append(f"最大回撤达到了 {max_dd:.1f}%，这意味着你的资产可能腰斩。")
        advice.append("强烈建议开启【自动止损】，或降低止损阈值（如设为 5%）。")
    
    if alpha < 0:
        advice.append("虽然赚钱了，但没跑赢大盘。不如直接买指数基金（ETF）省心。")
    else:
        advice.append("恭喜！你凭本事跑赢了市场基准。")

    next_step = ""
    if score >= 60:
        next_step = "🚀 **建议**：该策略表现稳健，可以尝试小资金实盘，或换一个时间段（如熊市）再测一次验证稳定性。"
    elif score >= 40:
        next_step = "🔧 **建议**：策略有潜力，但需要优化。试着调整参数（如 RSI 阈值、均线周期）再测几次。"
    else:
        next_step = "🛑 **建议**：该策略逻辑在当前市场行不通。请彻底更换思路或股票。"

    return stars, verdict, advice, next_step

# ==========================================
# 4. 文案内容
# ==========================================
def show_manual():
    st.markdown("""
    ### 📘 新手保姆级手册 (完整版)

    #### 1. 什么是量化回测？(时光机原理)
    想象你有一台**时光机**。你带着 **10万块钱** 回到了 **2021年**。你发誓严格执行一个死板的规则，绝不手软。这个模拟器就是帮你算出：**如果你真的这么做了，今天你手里会有多少钱？**

    #### 2. 回测 vs 实战演练：有什么区别？
    *   **🚀 策略回测 (看录像)**：
        *   **模式**：一键跑完过去3年的所有行情。
        *   **作用**：快速验证策略是否靠谱，筛选出赚钱的参数。
        *   **比喻**：像是在看一场已经踢完的球赛录像，分析哪里踢得好。
    *   **🤖 实战演练 (打模拟赛)**：
        *   **模式**：把未来的K线遮住，一天一天地走。
        *   **作用**：让你看着策略在“当下”是如何买卖的，验证你是否能忍受中间的波动。
        *   **比喻**：像是亲自上场踢球，你不知道下一秒对方会怎么传球。

    #### 3. 快速上手三步走
    *   **第一步：选战场 (市场)**: A股(茅台) / 美股(苹果)。
    *   **第二步：选武器 (策略)**: 推荐双均线(稳健)或海龟(激进)。
    *   **第三步：设防线 (风控)**: 止损(保命) / 止盈(落袋)。

    #### 4. 量化黑话词典
    *   **Alpha**: 比大盘多赚的钱。
    *   **夏普比率**: 性价比。>1.0 算好。
    *   **最大回撤**: 历史上最惨的一次亏损。
    """)

def show_wiki():
    st.markdown("""
    ### 🧠 策略百科全书
    #### 1. 双均线 (SMA Cross)
    *   **原理**: 快线上穿慢线买入。**适用**: 大趋势。**缺点**: 震荡市亏损。
    #### 2. RSI (相对强弱)
    *   **原理**: 低分抄底，高分逃顶。**适用**: 震荡市。**缺点**: 牛市踏空。
    #### 3. 布林带 (Bollinger)
    *   **原理**: 跌破下轨买，突破上轨卖。**适用**: 震荡修复。
    #### 4. 海龟交易 (Turtle)
    *   **原理**: 突破新高追涨。**适用**: 大牛市。**缺点**: 假突破。
    #### 5. 均值回归 (Mean Reversion)
    *   **原理**: 偏离均线太远会回调。**适用**: 急涨急跌。
    #### 6. 🛠️ 自定义策略
    *   **玩法**: 像造句一样组合你的交易逻辑。
    *   **例子**: 当 `[收盘价]` `[>]` `[均线]` `[20]` 时买入。
    """)

# ==========================================
# 5. 模拟实战逻辑 (V4.2 全策略支持版)
# ==========================================
def init_sim_session():
    if 'sim_step' not in st.session_state:
        st.session_state.sim_step = 50 
        st.session_state.sim_cash = 100000
        st.session_state.sim_shares = 0
        st.session_state.sim_history = []
        st.session_state.sim_data = None
        st.session_state.sim_indicators = None

def calculate_sim_indicators(df, strategy_name, params):
    """为模拟盘预计算指标 (Pandas版)"""
    inds = pd.DataFrame(index=df.index)
    inds['signal'] = 0 # 1=买, -1=卖
    
    try:
        if strategy_name == "双均线 (趋势策略)":
            inds['fast'] = df['close'].rolling(params['pfast']).mean()
            inds['slow'] = df['close'].rolling(params['pslow']).mean()
            buy_cond = (inds['fast'] > inds['slow']) & (inds['fast'].shift(1) <= inds['slow'].shift(1))
            sell_cond = (inds['fast'] < inds['slow']) & (inds['fast'].shift(1) >= inds['slow'].shift(1))
            inds.loc[buy_cond, 'signal'] = 1
            inds.loc[sell_cond, 'signal'] = -1
            
        elif strategy_name == "RSI (反转策略)":
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(params['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(params['rsi_period']).mean()
            rs = gain / loss
            inds['rsi'] = 100 - (100 / (1 + rs))
            inds.loc[inds['rsi'] < params['rsi_low'], 'signal'] = 1
            inds.loc[inds['rsi'] > params['rsi_high'], 'signal'] = -1
            
        elif strategy_name == "布林带 (通道策略)":
            inds['mid'] = df['close'].rolling(params['boll_period']).mean()
            std = df['close'].rolling(params['boll_period']).std()
            inds['top'] = inds['mid'] + params['boll_dev'] * std
            inds['bot'] = inds['mid'] - params['boll_dev'] * std
            inds.loc[df['close'] < inds['bot'], 'signal'] = 1
            inds.loc[df['close'] > inds['top'], 'signal'] = -1
            
        elif strategy_name == "海龟交易 (突破策略)":
            # 过去N天的最高/最低 (不含今天，防止未来函数)
            inds['high'] = df['high'].shift(1).rolling(params['turtle_period']).max()
            inds['low'] = df['low'].shift(1).rolling(params['turtle_period']).min()
            inds.loc[df['close'] > inds['high'], 'signal'] = 1
            inds.loc[df['close'] < inds['low'], 'signal'] = -1
            
        elif strategy_name == "均值回归 (抄底策略)":
            inds['sma'] = df['close'].rolling(params['mean_period']).mean()
            inds['dist'] = (df['close'] - inds['sma']) / inds['sma']
            inds.loc[inds['dist'] < -0.05, 'signal'] = 1 # 偏离-5%买入
            inds.loc[df['close'] >= inds['sma'], 'signal'] = -1 # 回归均线卖出
            
        elif strategy_name == "🛠️ 自定义策略":
            # 解析自定义参数
            p_ind = params['builder_indicator']
            p_op = params['builder_operator']
            p_thres = params['builder_threshold']
            p_val = params['builder_param']
            
            # 1. 左边
            if p_ind == 'RSI':
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                left_series = 100 - (100 / (1 + rs))
            else:
                left_series = df['close']
            
            # 2. 右边
            if p_thres == 'SMA':
                right_series = df['close'].rolling(int(p_val)).mean()
            else:
                right_series = float(p_val)
            
            # 3. 比较
            if p_op == '>':
                cond = left_series > right_series
            else:
                cond = left_series < right_series
                
            # 简单的信号生成：满足条件买，不满足卖
            inds.loc[cond, 'signal'] = 1
            inds.loc[~cond, 'signal'] = -1
            
    except Exception as e:
        st.error(f"指标计算错误: {e}")
        
    return inds

def run_simulation_tab():
    st.info("🤖 **策略实战演练**：看着策略自动交易，验证它的靠谱程度！")
    
    c1, c2 = st.columns([1, 1])
    with c1: sim_ticker = st.text_input("演练代码", "601360")
    with c2: 
        sim_strat = st.selectbox("演练策略", [
            "双均线 (趋势策略)", "RSI (反转策略)", "布林带 (通道策略)", 
            "海龟交易 (突破策略)", "均值回归 (抄底策略)", "🛠️ 自定义策略"
        ])
    
    # --- 演练参数配置 (与回测保持一致) ---
    sim_params = {}
    with st.expander("⚙️ 演练参数设置 (影响买卖点)", expanded=False):
        if sim_strat == "双均线 (趋势策略)":
            sim_params['pfast'] = st.slider("快线", 5, 30, 10, key='s_pfast')
            sim_params['pslow'] = st.slider("慢线", 20, 60, 30, key='s_pslow')
        elif sim_strat == "RSI (反转策略)":
            sim_params['rsi_period'] = 14
            sim_params['rsi_low'] = st.slider("超卖", 10, 40, 30, key='s_rlow')
            sim_params['rsi_high'] = st.slider("超买", 60, 90, 70, key='s_rhigh')
        elif sim_strat == "布林带 (通道策略)":
            sim_params['boll_period'] = st.slider("周期", 10, 50, 20, key='s_bper')
            sim_params['boll_dev'] = st.slider("倍数", 1.0, 3.0, 2.0, key='s_bdev')
        elif sim_strat == "海龟交易 (突破策略)":
            sim_params['turtle_period'] = st.slider("突破周期", 10, 60, 20, key='s_tper')
        elif sim_strat == "均值回归 (抄底策略)":
            sim_params['mean_period'] = st.slider("均线周期", 10, 50, 20, key='s_mper')
        elif sim_strat == "🛠️ 自定义策略":
            bc1, bc2 = st.columns(2)
            with bc1:
                b_ind = st.selectbox("指标", ["收盘价", "RSI"], key='s_bind')
                sim_params['builder_indicator'] = 'RSI' if 'RSI' in b_ind else 'Close'
                sim_params['builder_operator'] = st.selectbox("比较", [">", "<"], key='s_bop')
            with bc2:
                b_thres = st.selectbox("阈值", ["均线 (SMA)", "固定数值"], key='s_bthres')
                sim_params['builder_threshold'] = 'SMA' if 'SMA' in b_thres else 'Value'
                sim_params['builder_param'] = st.number_input("参数值", 0, 10000, 20, key='s_bval')
            sim_params['builder_operator'] = '>' if '>' in sim_params['builder_operator'] else '<'

    if st.button("🔄 重置/开始演练"):
        with st.spinner("准备数据..."):
            data_dict, _ = get_multiple_data("A股", [sim_ticker], datetime.date(2022, 1, 1), datetime.date.today())
            if sim_ticker in data_dict:
                st.session_state.sim_data = data_dict[sim_ticker]
                st.session_state.sim_indicators = calculate_sim_indicators(st.session_state.sim_data, sim_strat, sim_params)
                st.session_state.sim_step = 50
                st.session_state.sim_cash = 100000
                st.session_state.sim_shares = 0
                st.session_state.sim_history = []
                st.rerun()

    if st.session_state.get('sim_data') is not None:
        df = st.session_state.sim_data
        inds = st.session_state.sim_indicators
        step = st.session_state.sim_step
        
        if step >= len(df):
            st.success("演练结束！")
            return

        curr_date = df.index[step]
        curr_price = df['close'].iloc[step]
        signal = inds['signal'].iloc[step]
        action_msg = ""
        
        if signal == 1 and st.session_state.sim_shares == 0:
            cost = st.session_state.sim_cash * 0.95
            shares = int(cost / curr_price)
            st.session_state.sim_shares = shares
            st.session_state.sim_cash -= shares * curr_price
            st.session_state.sim_history.append(f"🟢 {curr_date.date()} [策略买入] {shares}股 @ {curr_price:.2f}")
            action_msg = "🟢 策略触发买入信号，已自动执行！"
        elif signal == -1 and st.session_state.sim_shares > 0:
            st.session_state.sim_cash += st.session_state.sim_shares * curr_price
            st.session_state.sim_history.append(f"🔴 {curr_date.date()} [策略卖出] {st.session_state.sim_shares}股 @ {curr_price:.2f}")
            st.session_state.sim_shares = 0
            action_msg = "🔴 策略触发卖出信号，已自动执行！"

        total_asset = st.session_state.sim_cash + st.session_state.sim_shares * curr_price
        pnl = (total_asset - 100000) / 100000
        
        if action_msg: st.toast(action_msg, icon="🤖")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("当前日期", curr_date.strftime("%Y-%m-%d"))
        m2.metric("当前价格", f"{curr_price:.2f}")
        m3.metric("总资产", f"{total_asset:.0f}", f"{pnl*100:.2f}%")

        show_df = df.iloc[step-50:step+1]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(show_df.index, show_df['close'], label='Price')
        
        # 简单的可视化辅助
        if "fast" in inds.columns:
            ax.plot(show_df.index, inds['fast'].iloc[step-50:step+1], label='Fast', linestyle='--')
            ax.plot(show_df.index, inds['slow'].iloc[step-50:step+1], label='Slow', linestyle='--')
        elif "top" in inds.columns:
            ax.plot(show_df.index, inds['top'].iloc[step-50:step+1], label='Top', linestyle=':', alpha=0.5)
            ax.plot(show_df.index, inds['bot'].iloc[step-50:step+1], label='Bot', linestyle=':', alpha=0.5)
            
        ax.legend()
        st.pyplot(fig)

        c_next, c_auto = st.columns(2)
        with c_next:
            if st.button("⏭️ 下一天 (单步)"):
                st.session_state.sim_step += 1
                st.rerun()
        with c_auto:
            if st.button("⏩ 快进 7 天"):
                st.session_state.sim_step += 7
                st.rerun()
        
        with st.expander("📜 自动交易记录", expanded=True):
            for h in reversed(st.session_state.sim_history):
                st.text(h)

# ==========================================
# 6. 主程序
# ==========================================
def main():
    st.set_page_config(page_title="量化交易模拟器", layout="wide", page_icon="📈", initial_sidebar_state="collapsed")
    st.title("📈 量化交易模拟器")
    
    init_sim_session()

    tab_sim, tab_game, tab_manual, tab_wiki = st.tabs(["🚀 策略回测", "🤖 策略实战演练", "📘 新手手册", "🧠 策略百科"])

    with tab_manual: show_manual()
    with tab_wiki: show_wiki()
    with tab_game: run_simulation_tab()

    with tab_sim:
        col_input, col_action = st.columns([3, 1])
        with col_input:
            default_tickers = "601360"
            data_source = st.selectbox("选择市场", ["美股/港股", "A股"], index=1)
            tickers_input = st.text_area("股票代码 (支持多只，用逗号隔开)", value=default_tickers, height=68, help="输入股票代码。\nA股示例：600519, 000858\n美股示例：AAPL, TSLA")

        with st.expander("⚙️ 策略与风控配置 (点击展开)", expanded=True):
            c1, c2 = st.columns(2)
            start_date = c1.date_input("回测开始日期", datetime.date(2021, 1, 1), help="模拟交易开始的时间。建议至少选1年前，数据太短算不出指标。")
            cash = c2.number_input("初始本金 (元/美元)", 100000, help="你模拟账户里的初始资金。建议设大一点（如10万），防止买不起一手高价股（如茅台一手要15万）。")
            
            strat_map = {
                "🛠️ 自定义策略": "Builder",
                "双均线 (趋势策略)": "SMA", 
                "RSI (反转策略)": "RSI", 
                "布林带 (通道策略)": "Bollinger",
                "海龟交易 (突破策略)": "Turtle",
                "均值回归 (抄底策略)": "MeanRev"
            }
            s_name = st.selectbox("选择策略模型", list(strat_map.keys()), help="决定什么时候买、什么时候卖的规则。不知道选哪个？去顶部的【策略百科】看看。")
            s_code = strat_map[s_name]
            
            params = {}
            if s_code == "Builder":
                st.info("🏗️ **自定义策略**：当 [指标] [比较] [阈值] 时买入")
                bc1, bc2, bc3, bc4 = st.columns([2, 1, 2, 2])
                with bc1:
                    b_ind = st.selectbox("指标", ["收盘价", "RSI"], help="作为判断依据的数据。")
                    params['builder_indicator'] = 'RSI' if 'RSI' in b_ind else 'Close'
                with bc2: params['builder_operator'] = st.selectbox("比较符", ["> (大于)", "< (小于)"], help="判断条件。")
                with bc3:
                    b_thres = st.selectbox("阈值类型", ["均线 (SMA)", "固定数值"], help="和什么进行比较？")
                    params['builder_threshold'] = 'SMA' if 'SMA' in b_thres else 'Value'
                with bc4:
                    def_val = 20 if params['builder_threshold'] == 'SMA' else (30 if params['builder_indicator'] == 'RSI' else 100)
                    params['builder_param'] = st.number_input("参数值", 0, 10000, def_val, help="如果是均线，这里填天数；如果是数值，这里填具体数字。")
                params['builder_operator'] = '>' if '>' in params['builder_operator'] else '<'

            elif s_code == "SMA":
                params['pfast'] = st.slider("快线周期 (灵敏)", 5, 30, 10, help="【灵敏度】数值越小，反应越快，但容易被假动作骗。")
                params['pslow'] = st.slider("慢线周期 (稳定)", 20, 60, 30, help="【趋势判断】数值越大，代表长期趋势，反应较慢。")
            elif s_code == "RSI":
                params['rsi_period'] = 14
                params['rsi_low'] = st.slider("超卖阈值 (买入线)", 10, 40, 30, help="【抄底线】低于这个分，说明跌过头了，建议买入。")
                params['rsi_high'] = st.slider("超买阈值 (卖出线)", 60, 90, 70, help="【逃顶线】高于这个分，说明涨过头了，建议卖出。")
            elif s_code == "Bollinger":
                params['boll_period'] = st.slider("计算周期", 10, 50, 20, help="计算中轨均线的天数，一般设20。")
                params['boll_dev'] = st.slider("标准差倍数 (通道宽度)", 1.0, 3.0, 2.0, help="【通道宽度】数值越大，通道越宽，不容易触发买卖；数值越小，交易越频繁。")
            elif s_code == "Turtle":
                params['turtle_period'] = st.slider("突破周期 (天)", 10, 60, 20, help="【追涨标准】如果今天的价格超过了过去 N 天的最高价，就买入。")
            elif s_code == "MeanRev":
                params['mean_period'] = st.slider("均线周期", 10, 50, 20, help="作为基准的均线天数。")

            st.divider()
            use_risk = st.checkbox("开启自动止盈止损 (推荐)", value=True, help="【新手必选】相当于给你的账户买了保险。")
            stop_loss = st.slider("止损比例 (跌多少卖)", 1, 20, 5, help="【保命线】如果亏损达到这个比例（如5%），系统强制卖出，防止亏光本金。") / 100.0
            take_profit = st.slider("止盈比例 (涨多少卖)", 5, 50, 15, help="【落袋线】如果盈利达到这个比例（如15%），系统强制卖出，锁定利润。") / 100.0

        run_btn = st.button("🚀 开始回测", type="primary", use_container_width=True)

        if run_btn:
            ticker_list = [t.strip() for t in tickers_input.split(',') if t.strip()]
            if not ticker_list: st.error("请输入至少一个股票代码")
            else:
                with st.spinner("正在连接数据源并计算..."):
                    data_dict, df_bench = get_multiple_data(data_source, ticker_list, start_date, datetime.date.today())
                    
                    if not data_dict: st.error("数据获取失败。请检查代码是否正确。")
                    else:
                        cerebro = bt.Cerebro()
                        valid_cnt = 0
                        min_bars = 60
                        for t, df in data_dict.items():
                            if len(df) < min_bars:
                                st.warning(f"⚠️ {t} 数据不足 {min_bars} 天，已跳过。")
                                continue
                            data = bt.feeds.PandasData(dataname=df, name=t)
                            cerebro.adddata(data)
                            valid_cnt += 1
                        
                        if valid_cnt == 0: st.error("所有股票数据都不足，无法回测。")
                        else:
                            cerebro.addstrategy(PortfolioStrategy, strategy_type=s_code, use_risk_mgmt=use_risk, stop_loss=stop_loss, take_profit=take_profit, **params)
                            cerebro.broker.setcash(cash)
                            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
                            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                            
                            try:
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
                                sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio')
                                
                                alpha = 0
                                if bench_returns is not None:
                                    bench_total = (1 + bench_returns).cumprod().iloc[-1] - 1
                                    alpha = ret_pct - bench_total

                                stars, verdict, advice, next_step = generate_strategy_report(ret_pct, max_dd, alpha, sharpe)
                                
                                st.divider()
                                st.subheader("🤖 智能回测点评")
                                
                                col_score, col_verdict = st.columns([1, 3])
                                with col_score:
                                    st.metric("综合评分", stars)
                                with col_verdict:
                                    if "失效" in verdict: st.error(verdict)
                                    elif "微薄" in verdict or "风险" in verdict: st.warning(verdict)
                                    else: st.success(verdict)
                                
                                with st.container(border=True):
                                    st.markdown("**🧐 详细分析：**")
                                    for item in advice:
                                        st.markdown(f"- {item}")
                                    st.markdown("---")
                                    st.markdown(next_step)
                                st.divider()

                                c1, c2, c3 = st.columns(3)
                                c1.metric("最终资产", f"${final_cash/1000:.1f}k", help="回测结束时，你账户里的总钱数（本金+盈亏）。")
                                c2.metric("总收益率", f"{ret_pct*100:.1f}%", delta_color="normal" if ret_pct>0 else "inverse", help="（最终资产 - 初始本金）/ 初始本金。")
                                c3.metric("最大回撤", f"{max_dd:.1f}%", help="【风险指标】历史上最倒霉的时候，从最高点跌下来跌了多少。数值越小越安全。")
                                
                                fig, ax = plt.subplots(figsize=(8, 4))
                                cum_strat = (1 + strat_returns).cumprod()
                                ax.plot(cum_strat.index, cum_strat, color='#2962FF', linewidth=2, label='我的策略')
                                if bench_returns is not None:
                                    cum_bench = (1 + bench_returns).cumprod()
                                    ax.plot(cum_bench.index, cum_bench, color='gray', linestyle='--', alpha=0.6, label='市场基准')
                                ax.legend()
                                st.pyplot(fig)
                                
                                with st.expander("📊 详细数据报告 (中文版)"):
                                    try:
                                        metrics = qs.reports.metrics(strat_returns, benchmark=bench_returns, mode='basic', display=False)
                                        trans_map = {
                                            'Start Period': '开始日期', 'End Period': '结束日期',
                                            'Risk-Free Rate': '无风险利率', 'Time in Market': '持仓时间占比',
                                            'Cumulative Return': '累计收益率', 'CAGR﹪': '年化收益率',
                                            'Sharpe': '夏普比率', 'Prob. Sharpe Ratio': '概率夏普比',
                                            'Sortino': '索提诺比率', 'Sortino/√2': '索提诺/√2',
                                            'Omega': '欧米伽比率', 'Max Drawdown': '最大回撤',
                                            'Longest DD Days': '最长回撤天数', 'Volatility (ann.)': '年化波动率',
                                            'R^2': 'R平方 (拟合度)', 'Information Ratio': '信息比率',
                                            'Calmar': '卡玛比率', 'Skew': '偏度', 'Kurtosis': '峰度',
                                            'Expected Daily %%': '日均预期收益', 'Expected Monthly %%': '月均预期收益',
                                            'Expected Yearly %%': '年均预期收益', 'Kelly Criterion': '凯利公式仓位',
                                            'Risk of Ruin': '破产概率', 'Daily Value-at-Risk': '日风险价值(VaR)',
                                            'Expected Shortfall (cVaR)': '预期亏损(cVaR)',
                                            'Gain/Pain Ratio': '收益痛苦比', 'Gain/Pain (1M)': '收益痛苦比(1月)',
                                            'Payoff Ratio': '盈亏比', 'Profit Factor': '获利因子',
                                            'Common Sense Ratio': '常识比率', 'CPC Index': 'CPC指数',
                                            'Tail Ratio': '尾部比率', 'Outlier Win Ratio': '异常盈利比',
                                            'Outlier Loss Ratio': '异常亏损比', 'MTD': '本月收益',
                                            '3M': '近3月收益', '6M': '近6月收益', 'YTD': '今年以来收益',
                                            '1Y': '近1年收益', '3Y (ann.)': '近3年年化',
                                            '5Y (ann.)': '近5年年化', '10Y (ann.)': '近10年年化',
                                            'All-time (ann.)': '全时段年化', 'Best Day': '最好的一天',
                                            'Worst Day': '最惨的一天', 'Best Month': '最好的月',
                                            'Worst Month': '最惨的月', 'Best Year': '最好的年',
                                            'Worst Year': '最惨的年', 'Avg. Drawdown': '平均回撤',
                                            'Avg. Drawdown Days': '平均回撤天数', 'Recovery Factor': '恢复因子',
                                            'Ulcer Index': '溃疡指数', 'Serenity Index': '宁静指数',
                                            'Avg. Up Month': '平均上涨月收益', 'Avg. Down Month': '平均下跌月收益',
                                            'Win Days %%': '盈利天数占比', 'Win Month %%': '盈利月数占比',
                                            'Win Quarter %%': '盈利季度占比', 'Win Year %%': '盈利年份占比'
                                        }
                                        metrics_cn = metrics.rename(index=trans_map)
                                        st.dataframe(metrics_cn, use_container_width=True)
                                        
                                        report_file = "qs_report.html"
                                        qs.reports.html(strat_returns, benchmark=bench_returns, output=report_file, title="Quant Report", download_filename=report_file)
                                        with open(report_file, 'r', encoding='utf-8') as f:
                                            html_content = f.read()
                                        
                                        html_content = html_content.replace('Strategy', '我的策略')
                                        html_content = html_content.replace('Benchmark', '市场基准')
                                        html_content = html_content.replace('Cumulative Return', '累计收益率')
                                        html_content = html_content.replace('Max Drawdown', '最大回撤')
                                        html_content = html_content.replace('Sharpe', '夏普比率')
                                        html_content = html_content.replace('Volatility', '波动率')
                                        html_content = html_content.replace('EOY Returns', '年度收益热力图')
                                        html_content = html_content.replace('Monthly Returns', '月度收益')
                                        html_content = html_content.replace('Distribution of Returns', '收益分布')
                                        html_content = html_content.replace('Daily Returns', '日收益率')
                                        html_content = html_content.replace('Rolling Volatility', '滚动波动率')
                                        html_content = html_content.replace('Rolling Sharpe', '滚动夏普比')
                                        html_content = html_content.replace('Underwater Plot', '潜水图 (回撤)')
                                        
                                        st.download_button("📥 下载完整HTML报告 (已汉化)", html_content, file_name="report_cn.html", mime="text/html")
                                        
                                    except Exception as e:
                                        st.error(f"指标计算失败: {e}")
                            except Exception as e:
                                st.error(f"回测运行出错: {e}")

if __name__ == '__main__':
    main()
