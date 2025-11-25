import streamlit as st
import datetime
import pandas as pd
import yfinance as yf
import akshare as ak
import numpy as np

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
# 2. 数据获取 (新增基准指数获取)
# ==========================================
@st.cache_data(ttl=3600)
def get_data_with_benchmark(source, ticker, start_date, end_date):
    stock_df = pd.DataFrame()
    bench_df = pd.DataFrame()
    
    try:
        if source == "美股/港股 (Yahoo)":
            # 1. 获取个股
            stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(stock_df.columns, pd.MultiIndex):
                stock_df.columns = stock_df.columns.get_level_values(0)
            
            # 2. 获取基准 (标普500)
            bench_ticker = "^GSPC" 
            bench_df = yf.download(bench_ticker, start=start_date, end=end_date, progress=False)
            if isinstance(bench_df.columns, pd.MultiIndex):
                bench_df.columns = bench_df.columns.get_level_values(0)

        elif source == "A股 (AkShare)":
            s_str = start_date.strftime("%Y%m%d")
            e_str = end_date.strftime("%Y%m%d")
            
            # 1. 获取个股
            stock_df = ak.stock_zh_a_hist(symbol=ticker, start_date=s_str, end_date=e_str, adjust="qfq")
            stock_df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
            stock_df.index = pd.to_datetime(stock_df['date'])
            stock_df = stock_df[['open', 'high', 'low', 'close', 'volume']]

            # 2. 获取基准 (沪深300)
            # AkShare 指数接口
            bench_df = ak.stock_zh_index_daily(symbol="sh000300")
            bench_df.rename(columns={'date': 'date', 'close': 'close'}, inplace=True)
            bench_df.index = pd.to_datetime(bench_df['date'])
            # 截取对应时间段
            bench_df = bench_df[(bench_df.index >= pd.to_datetime(start_date)) & (bench_df.index <= pd.to_datetime(end_date))]

        return stock_df, bench_df

    except Exception as e:
        st.error(f"数据获取异常: {e}")
        return None, None

# ==========================================
# 3. 主程序
# ==========================================
def main():
    st.set_page_config(page_title="Quant Pro 现实版", layout="wide", page_icon="⚖️")
    st.title("⚖️ Quant Pro 量化系统 (V1.2 现实版)")

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("🎛️ 设置")
        data_source = st.selectbox("市场", ["美股/港股 (Yahoo)", "A股 (AkShare)"])
        
        if data_source == "美股/港股 (Yahoo)":
            ticker = st.text_input("代码", "AAPL")
            benchmark_name = "标普500指数"
        else:
            ticker = st.text_input("代码", "600519")
            benchmark_name = "沪深300指数"
            
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

        # 简化的参数输入
        params = {}
        if s_code == "SMA":
            params['pfast'] = st.slider("快线", 5, 50, 10)
            params['pslow'] = st.slider("慢线", 20, 100, 30)
        # ... (其他策略参数省略，使用默认值以简化代码长度，实际使用可补全) ...

        run_btn = st.button("🚀 运行回测", type="primary")

    # --- 主界面 ---
    if run_btn:
        with st.spinner("正在下载数据并进行双轨对比..."):
            # 1. 获取双份数据
            df_stock, df_bench = get_data_with_benchmark(data_source, ticker, start_date, end_date)
            
            if df_stock is None or df_stock.empty:
                st.error("股票数据获取失败")
                return

            # 2. 运行 Backtrader
            cerebro = bt.Cerebro()
            cerebro.adddata(bt.feeds.PandasData(dataname=df_stock))
            cerebro.addstrategy(MegaStrategy, strategy_type=s_code, use_risk_mgmt=use_risk, stop_loss=stop_loss, take_profit=take_profit, **params)
            cerebro.broker.setcash(cash)
            
            # 添加 TimeReturn 分析器，用于提取每日收益率
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
            
            results = cerebro.run()
            strat = results[0]
            
            # 3. 数据处理：策略 vs 基准
            # 提取策略收益率序列
            strat_returns = pd.Series(strat.analyzers.returns.get_analysis())
            # 计算策略累计收益率 (Cumulative Return)
            strat_cum = (1 + strat_returns).cumprod()
            strat_cum.name = "Strategy"

            # 处理基准数据
            if df_bench is not None and not df_bench.empty:
                # 计算基准每日收益率
                bench_returns = df_bench['close'].pct_change().fillna(0)
                # 截取与策略相同的时间段
                bench_returns = bench_returns.reindex(strat_returns.index).fillna(0)
                # 计算基准累计收益率
                bench_cum = (1 + bench_returns).cumprod()
                bench_cum.name = "Benchmark"
            else:
                bench_cum = pd.Series(1, index=strat_returns.index) # 如果获取失败，画一条平线

            # 4. 计算核心指标
            total_return = strat_cum.iloc[-1] - 1
            bench_return = bench_cum.iloc[-1] - 1
            alpha = total_return - bench_return # 超额收益

            # 5. 展示结果
            st.subheader("📊 现实检验 (Reality Check)")
            
            # 指标卡片
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("策略总收益", f"{total_return*100:.2f}%")
            c2.metric(f"基准收益 ({benchmark_name})", f"{bench_return*100:.2f}%", help="如果你什么都不做，直接买指数基金的收益")
            
            # Alpha 颜色逻辑
            alpha_color = "normal" if alpha > 0 else "inverse"
            c3.metric("Alpha (超额收益)", f"{alpha*100:.2f}%", delta=f"{alpha*100:.2f}%", delta_color=alpha_color, help="策略收益 - 基准收益。如果是负数，说明你跑输了大盘。")
            
            final_cash = cerebro.broker.getvalue()
            c4.metric("最终资产", f"${final_cash:,.0f}")

            # 6. 绘制对比图 (Matplotlib)
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制策略线 (蓝色)
            ax.plot(strat_cum.index, strat_cum, label='我的策略 (Strategy)', color='#1f77b4', linewidth=2)
            
            # 绘制基准线 (灰色)
            ax.plot(bench_cum.index, bench_cum, label=f'市场基准 ({benchmark_name})', color='gray', linestyle='--', alpha=0.7)
            
            # 填充 Alpha 区域
            ax.fill_between(strat_cum.index, strat_cum, bench_cum, where=(strat_cum > bench_cum), color='green', alpha=0.1, label='跑赢大盘')
            ax.fill_between(strat_cum.index, strat_cum, bench_cum, where=(strat_cum <= bench_cum), color='red', alpha=0.1, label='跑输大盘')

            ax.set_title("策略净值 vs 市场基准")
            ax.legend()
            st.pyplot(fig)

            # 7. 原始交易图表
            with st.expander("查看详细买卖点 (K线图)"):
                try:
                    figs = cerebro.plot(style='candlestick', volume=False)
                    st.pyplot(figs[0][0])
                except: pass

            # 8. 交易日志
            with st.expander("查看交易日志"):
                if strat.log_list:
                    st.dataframe(pd.DataFrame(strat.log_list, columns=["详情"]), use_container_width=True)

if __name__ == '__main__':
    main()
