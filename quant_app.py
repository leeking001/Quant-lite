import streamlit as st
import datetime
import pandas as pd
import yfinance as yf
import akshare as ak
import numpy as np
import time
import quantstats as qs # V1.3 新增核心库
import streamlit.components.v1 as components # 用于渲染 HTML

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

plt.style.use('bmh') 

import backtrader as bt

# ==========================================
# 1. 策略类 (保持 V1.1 风控版逻辑)
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
# 2. 数据获取 (增强版：重试 + 小写修复)
# ==========================================
@st.cache_data(ttl=3600)
def get_data_with_benchmark(source, ticker, start_date, end_date):
    stock_df = pd.DataFrame()
    bench_df = pd.DataFrame()
    max_retries = 3
    
    try:
        if source == "美股/港股 (Yahoo)":
            # 1. 个股
            for i in range(max_retries):
                try:
                    if i > 0: time.sleep(1)
                    stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=10)
                    if not stock_df.empty: break
                except: pass
            
            if isinstance(stock_df.columns, pd.MultiIndex):
                stock_df.columns = stock_df.columns.get_level_values(0)
            stock_df.columns = stock_df.columns.str.lower()
            
            # 2. 基准
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
# 3. 主程序
# ==========================================
def main():
    st.set_page_config(page_title="Quant Pro 专业版", layout="wide", page_icon="📈")
    st.title("📈 Quant Pro 量化系统 (V1.3 专业版)")

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("🎛️ 设置")
        data_source = st.selectbox("市场", ["美股/港股 (Yahoo)", "A股 (AkShare)"])
        
        if data_source == "美股/港股 (Yahoo)":
            ticker = st.text_input("代码", "AAPL")
            benchmark_name = "标普500"
        else:
            ticker = st.text_input("代码", "600519")
            benchmark_name = "沪深300"
            
        col1, col2 = st.columns(2)
        start_date = col1.date_input("开始", datetime.date(2021, 1, 1))
        end_date = col2.date_input("结束", datetime.date.today())
        cash = st.number_input("本金", 100000)

        st.subheader("策略")
        strat_map = {"SMA": "SMA", "RSI": "RSI", "MACD": "MACD", "Bollinger": "Bollinger", "Momentum": "Momentum"}
        s_name = st.selectbox("模型", list(strat_map.keys()))
        s_code = strat_map[s_name]

        with st.expander("🛡️ 风控", expanded=False):
            use_risk = st.checkbox("开启止盈止损", value=True)
            stop_loss = st.slider("止损%", 1, 20, 5) / 100.0
            take_profit = st.slider("止盈%", 5, 50, 15) / 100.0

        params = {}
        if s_code == "SMA":
            params['pfast'] = st.slider("快线", 5, 50, 10)
            params['pslow'] = st.slider("慢线", 20, 100, 30)
        # ... 其他参数省略 ...

        run_btn = st.button("🚀 运行回测", type="primary")

    # --- 主界面 ---
    # 使用 Tabs 分离基础结果和专业报告
    tab1, tab2, tab3 = st.tabs(["📊 基础回测", "📑 专业报告 (QuantStats)", "📝 交易日志"])

    if run_btn:
        with st.spinner("正在进行量化计算与报告生成..."):
            # 1. 获取数据
            df_stock, df_bench = get_data_with_benchmark(data_source, ticker, start_date, end_date)
            
            if df_stock is None or df_stock.empty:
                st.error("数据获取失败")
                return

            # 2. 运行 Backtrader
            cerebro = bt.Cerebro()
            cerebro.adddata(bt.feeds.PandasData(dataname=df_stock))
            cerebro.addstrategy(MegaStrategy, strategy_type=s_code, use_risk_mgmt=use_risk, stop_loss=stop_loss, take_profit=take_profit, **params)
            cerebro.broker.setcash(cash)
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
            
            results = cerebro.run()
            strat = results[0]
            
            # 3. 数据处理
            # 提取策略收益率 (Pandas Series)
            strat_returns = pd.Series(strat.analyzers.returns.get_analysis())
            strat_returns.index = pd.to_datetime(strat_returns.index) # 确保索引是时间格式
            
            # 提取基准收益率
            if df_bench is not None and not df_bench.empty and 'close' in df_bench.columns:
                bench_returns = df_bench['close'].pct_change().fillna(0)
                bench_returns.index = pd.to_datetime(bench_returns.index)
                # 对齐时间轴
                bench_returns = bench_returns.reindex(strat_returns.index).fillna(0)
            else:
                bench_returns = None

            # --- Tab 1: 基础回测结果 ---
            with tab1:
                final_cash = cerebro.broker.getvalue()
                total_return = (final_cash - cash) / cash
                
                c1, c2 = st.columns(2)
                c1.metric("最终资产", f"${final_cash:,.0f}")
                c2.metric("总收益率", f"{total_return*100:.2f}%", delta_color="normal" if total_return > 0 else "inverse")

                st.subheader("净值曲线")
                fig, ax = plt.subplots(figsize=(10, 5))
                # 计算累计收益
                cum_strat = (1 + strat_returns).cumprod()
                ax.plot(cum_strat.index, cum_strat, label="策略")
                if bench_returns is not None:
                    cum_bench = (1 + bench_returns).cumprod()
                    ax.plot(cum_bench.index, cum_bench, label=f"基准 ({benchmark_name})", alpha=0.6, linestyle="--")
                ax.legend()
                st.pyplot(fig)

            # --- Tab 2: QuantStats 专业报告 (V1.3 核心) ---
            with tab2:
                st.info("💡 提示: 下方报告由 QuantStats 生成，包含华尔街级别的详细指标。")
                
                try:
                    # 使用 QuantStats 生成 HTML 报告
                    # 注意：Streamlit Cloud 文件系统是临时的，我们生成后读取字符串
                    report_file = "quantstats_report.html"
                    
                    # 生成报告 (suppress_warnings=True 防止 Matplotlib 冲突)
                    qs.reports.html(
                        strat_returns, 
                        benchmark=bench_returns, 
                        output=report_file, 
                        title=f"{ticker} 策略分析报告",
                        download_filename=report_file
                    )
                    
                    # 读取 HTML 内容并渲染
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_html = f.read()
                    
                    # 使用 iframe 嵌入显示 (height 设置高一点以便滚动)
                    components.html(report_html, height=800, scrolling=True)
                    
                    # 提供下载按钮
                    st.download_button(
                        label="📥 下载完整 HTML 报告",
                        data=report_html,
                        file_name=f"quant_report_{ticker}.html",
                        mime="text/html"
                    )
                    
                except Exception as e:
                    st.error(f"报告生成失败: {e}")
                    st.caption("可能是数据太短，或者收益率序列包含太多 NaN。")

            # --- Tab 3: 交易日志 ---
            with tab3:
                if strat.log_list:
                    st.dataframe(pd.DataFrame(strat.log_list, columns=["详情"]), use_container_width=True)
                else:
                    st.info("无交易记录")

if __name__ == '__main__':
    main()
