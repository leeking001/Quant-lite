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
import requests # 用于下载字体
from matplotlib import font_manager

# ==========================================
# 0. 系统配置 (中文乱码修复版)
# ==========================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import warnings
import matplotlib.dates

if not hasattr(matplotlib.dates, 'warnings'):
    matplotlib.dates.warnings = warnings

# 1. 先设置绘图风格
plt.style.use('seaborn-v0_8') 

# 2. 核心修复：自动下载并加载中文字体
def init_chinese_font():
    # 字体文件名
    font_name = "SimHei.ttf"
    # 字体下载地址 (使用 GitHub 镜像或稳定源)
    font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
    
    # 如果字体文件不存在，则下载
    if not os.path.exists(font_name):
        with st.spinner("正在初始化中文字体，初次运行可能需要几秒钟..."):
            try:
                response = requests.get(font_url, timeout=20)
                with open(font_name, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                st.error(f"字体下载失败，图表中文可能无法显示: {e}")
    
    # 注册字体
    if os.path.exists(font_name):
        font_manager.fontManager.addfont(font_name)
        # 设置全局字体
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定黑体
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块的问题

# 执行字体初始化
init_chinese_font()

import backtrader as bt

# ==========================================
# 1. 策略类 (逻辑核心)
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
                self.log(f'🟢 买入成功: 价格 {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'🔴 卖出成功: 价格 {order.executed.price:.2f} | 盈亏: {order.executed.pnl:.2f}')
            self.order = None

    def next(self):
        if self.order: return

        # 风控逻辑
        if self.position and self.params.use_risk_mgmt:
            buy_price = self.position.price
            current_price = self.dataclose[0]
            pnl_pct = (current_price - buy_price) / buy_price

            if pnl_pct <= -self.params.stop_loss:
                self.log(f'🛡️ 触发止损 (亏损 {pnl_pct*100:.2f}%) - 卖出保命')
                self.close()
                return
            if pnl_pct >= self.params.take_profit:
                self.log(f'💰 触发止盈 (盈利 {pnl_pct*100:.2f}%) - 落袋为安')
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
# 3. 文案内容
# ==========================================
def show_user_guide():
    st.markdown("""
    ### 🐣 新手第一课：怎么玩这个系统？
    
    **1. 选股票**
    *   在左边选择是玩 **美股** 还是 **A股**。
    *   输入代码。比如茅台是 `600519`，苹果是 `AAPL`。

    **2. 选策略 (怎么买卖)**
    *   **双均线 (SMA)**: 最简单。金叉买，死叉卖。适合大牛市。
    *   **RSI**: 适合震荡市。跌多了买，涨多了卖。
    *   **布林带**: 价格跌破下轨买，突破上轨卖。

    **3. 设风控 (保命符)**
    *   **止损**: 比如设 5%。买入后如果跌了 5%，系统强制卖出。**新手一定要开！**
    *   **止盈**: 比如设 15%。买入后赚了 15%，系统强制卖出落袋为安。

    **4. 看结果**
    *   **跑赢大盘**: 如果是红色的正数，说明你比直接买指数基金强。
    *   **性价比**: 越高越好。
    *   **最惨亏损**: 历史上最倒霉的时候亏了多少。
    """)

def show_strategy_wiki():
    st.markdown("""
    ### 📖 策略大白话解释

    #### 1. 双均线 (SMA) —— "追涨杀跌"
    *   **人话**: 有两根线，一根快线（反应快），一根慢线（反应慢）。
    *   **买入**: 当快线从下往上穿过慢线，说明最近涨得比以前快，趋势来了，买！
    *   **卖出**: 当快线从上往下穿过慢线，说明涨不动了，要跌，卖！
    *   **适合**: 这种策略在**大涨大跌**的行情里最赚钱。
    *   **坑点**: 如果股价横盘震荡，你会频繁买卖，把手续费亏光。

    #### 2. RSI —— "物极必反"
    *   **人话**: 这是一个 0 到 100 的分数。
    *   **买入**: 分数低于 30，说明大家都在恐慌抛售，价格可能被低估了，抄底！
    *   **卖出**: 分数高于 70，说明大家都在疯狂买入，价格可能太贵了，逃顶！
    *   **适合**: 股价上蹿下跳的时候。

    #### 3. 布林带 (Bollinger) —— "出轨回归"
    *   **人话**: 股价通常在一条通道里运行。
    *   **买入**: 股价跌破了通道下边缘（出轨了），通常会弹回来，买！
    *   **卖出**: 股价突破了通道上边缘，通常会跌回来，卖！
    """)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    st.set_page_config(page_title="量化回测新手版", layout="wide", page_icon="🌱")
    
    st.title("🌱 个人量化回测系统 (新手友好版)")
    st.caption("不用懂代码，像玩游戏一样测试你的炒股策略")

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("🎛️ 操作面板")
        
        st.subheader("1. 选市场")
        data_source = st.selectbox("数据来源", ["美股/港股 (Yahoo)", "A股 (AkShare)"], help="A股数据由 AkShare 提供，美股由 Yahoo 提供")
        
        if data_source == "美股/港股 (Yahoo)":
            ticker = st.text_input("股票代码", "AAPL", help="例如: AAPL (苹果), TSLA (特斯拉), BTC-USD (比特币)")
            benchmark_name = "标普500指数"
        else:
            ticker = st.text_input("股票代码", "600519", help="请输入6位数字代码，例如: 600519 (茅台)")
            benchmark_name = "沪深300指数"
            
        col1, col2 = st.columns(2)
        start_date = col1.date_input("开始日期", datetime.date(2021, 1, 1))
        end_date = col2.date_input("结束日期", datetime.date.today())
        cash = st.number_input("初始本金", 100000, help="你模拟账户里的初始资金，建议填 10万")

        st.subheader("2. 选策略")
        strat_map = {
            "双均线 (趋势策略)": "SMA", 
            "RSI (反转策略)": "RSI", 
            "MACD (综合策略)": "MACD", 
            "布林带 (通道策略)": "Bollinger", 
            "动量 (惯性策略)": "Momentum"
        }
        s_name = st.selectbox("策略模型", list(strat_map.keys()), help="不知道选哪个？去右边的'策略大白话'看看")
        s_code = strat_map[s_name]

        with st.expander("🛡️ 风控设置 (保命必看)", expanded=True):
            use_risk = st.checkbox("开启自动止盈止损", value=True, help="强烈建议开启！这是新手不亏光本金的关键。")
            stop_loss = st.slider("止损线 (跌多少卖)", 1, 20, 5, help="如果亏了 5%，系统自动帮你卖出，防止亏更多。") / 100.0
            take_profit = st.slider("止盈线 (涨多少卖)", 5, 50, 15, help="如果赚了 15%，系统自动帮你卖出，落袋为安。") / 100.0

        # 动态参数
        params = {}
        if s_code == "SMA":
            st.info("💡 双均线：短线穿长线就买。")
            params['pfast'] = st.slider("快线周期", 5, 50, 10, help="反应比较快的均线，一般设 5 或 10")
            params['pslow'] = st.slider("慢线周期", 20, 100, 30, help="反应比较慢的均线，一般设 20 或 30")
        elif s_code == "RSI":
            st.info("💡 RSI：分数太低买，分数太高卖。")
            params['rsi_period'] = st.slider("计算周期", 5, 30, 14)
            params['rsi_low'] = st.slider("抄底分数 (买)", 10, 40, 30, help="低于这个分，说明超卖了，买入")
            params['rsi_high'] = st.slider("逃顶分数 (卖)", 60, 90, 70, help="高于这个分，说明超买了，卖出")
        # ... 其他参数省略 ...

        run_btn = st.button("🚀 开始回测", type="primary", help="点击开始模拟交易")

    # --- 主界面 ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 结果看板", "📑 专业体检报告", "📖 策略大白话", "🐣 新手教程"])

    # --- Tab 1: 结果看板 ---
    with tab1:
        if run_btn:
            with st.spinner("正在模拟交易中，请稍等..."):
                df_stock, df_bench = get_data_with_benchmark(data_source, ticker, start_date, end_date)
                
                if df_stock is None or df_stock.empty:
                    st.error("❌ 数据获取失败。如果是美股，可能是网络问题；如果是A股，请检查代码是否正确。")
                else:
                    # 运行回测
                    cerebro = bt.Cerebro()
                    cerebro.adddata(bt.feeds.PandasData(dataname=df_stock))
                    cerebro.addstrategy(MegaStrategy, strategy_type=s_code, use_risk_mgmt=use_risk, stop_loss=stop_loss, take_profit=take_profit, **params)
                    cerebro.broker.setcash(cash)
                    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
                    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                    
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
                    max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
                    
                    # 1. 核心指标卡片
                    st.subheader("核心成绩单")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("最终资产", f"${final_cash:,.0f}", help="你的本金 + 赚的钱")
                    c2.metric("总收益率", f"{total_return*100:.2f}%", delta_color="normal" if total_return > 0 else "inverse", help="一共赚了百分之多少")
                    
                    if bench_returns is not None:
                        bench_total = (1 + bench_returns).cumprod().iloc[-1] - 1
                        alpha = total_return - bench_total
                        c3.metric("跑赢大盘 (Alpha)", f"{alpha*100:.2f}%", delta=f"{alpha*100:.2f}%", help="正数说明你比买指数基金强，负数说明你瞎折腾还不如躺平")
                    else:
                        c3.metric("跑赢大盘", "无数据")

                    c4.metric("最惨亏损", f"{max_dd:.2f}%", help="历史上从最高点跌下来最惨的一次跌了多少。越小越好。")

                    # 2. 净值曲线 (浅色清晰版)
                    st.subheader("资金走势图")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    cum_strat = (1 + strat_returns).cumprod()
                    ax.plot(cum_strat.index, cum_strat, label="我的策略", color='#2962FF', linewidth=2)
                    
                    if bench_returns is not None:
                        cum_bench = (1 + bench_returns).cumprod()
                        ax.plot(cum_bench.index, cum_bench, label=f"市场基准 ({benchmark_name})", color='#B0BEC5', linestyle='--', alpha=0.8)
                        ax.fill_between(cum_strat.index, cum_strat, cum_bench, where=(cum_strat > cum_bench), color='#00C853', alpha=0.1, label="跑赢区域")
                        ax.fill_between(cum_strat.index, cum_strat, cum_bench, where=(cum_strat <= cum_bench), color='#D50000', alpha=0.1, label="跑输区域")
                    
                    ax.legend()
                    ax.set_ylabel("累计收益 (1.0 = 本金)")
                    st.pyplot(fig)

                    # 3. 交易记录
                    with st.expander("📝 查看每一笔买卖记录"):
                        if strat.log_list:
                            st.dataframe(pd.DataFrame(strat.log_list, columns=["交易详情"]), use_container_width=True)
                        else:
                            st.info("这段时间没有触发任何交易。")
                    
                    # 保存数据给 Tab 2 使用
                    st.session_state['strat_returns'] = strat_returns
                    st.session_state['bench_returns'] = bench_returns
                    st.session_state['ticker'] = ticker

        else:
            st.info("👈 请在左侧设置好参数，然后点击 '开始回测' 按钮")

    # --- Tab 2: 专业报告 ---
    with tab2:
        if 'strat_returns' in st.session_state:
            st.markdown("### 深度体检报告")
            st.caption("这份报告由 QuantStats 生成，包含了华尔街基金经理看的所有指标。")
            try:
                report_file = "qs_report.html"
                qs.reports.html(
                    st.session_state['strat_returns'], 
                    benchmark=st.session_state['bench_returns'], 
                    output=report_file, 
                    title=f"{st.session_state['ticker']} 策略分析",
                    download_filename=report_file
                )
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_html = f.read()
                components.html(report_html, height=1000, scrolling=True)
            except Exception as e:
                st.error(f"报告生成失败: {e}")
        else:
            st.warning("请先在 '结果看板' 运行一次回测。")

    # --- Tab 3: 策略百科 ---
    with tab3:
        show_strategy_wiki()

    # --- Tab 4: 新手指南 ---
    with tab4:
        show_user_guide()

if __name__ == '__main__':
    main()
