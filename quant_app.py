import streamlit as st
import datetime
import pandas as pd
import yfinance as yf
import akshare as ak  # 新增 A股数据库

# ==========================================
# 0. 兼容性补丁 (必须保留)
# ==========================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import warnings
import matplotlib.dates
if not hasattr(matplotlib.dates, 'warnings'):
    matplotlib.dates.warnings = warnings

import backtrader as bt

# ==========================================
# 1. 综合策略类 (支持 SMA, RSI, MACD)
# ==========================================
class CompositeStrategy(bt.Strategy):
    params = (
        ('strategy_type', 'SMA'), # 策略类型: SMA, RSI, MACD
        # SMA 参数
        ('pfast', 10), 
        ('pslow', 30),
        # RSI 参数
        ('rsi_period', 14),
        ('rsi_low', 30),
        ('rsi_high', 70),
        # MACD 参数
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.log_list = [] 

        # --- 初始化所有指标 (无论是否使用，为了绘图好看) ---
        
        # 1. SMA 双均线
        self.sma1 = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.pfast)
        self.sma2 = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.pslow)
        self.sma_cross = bt.indicators.CrossOver(self.sma1, self.sma2)

        # 2. RSI 相对强弱指标
        self.rsi = bt.indicators.RSI(
            self.datas[0], period=self.params.rsi_period)

        # 3. MACD 指标
        self.macd = bt.indicators.MACD(
            self.datas[0], 
            period_me1=self.params.macd_fast, 
            period_me2=self.params.macd_slow, 
            period_signal=self.params.macd_signal
        )
        # MACD 金叉/死叉 (MACD线 上穿 信号线)
        self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        self.log_list.append(f'{dt.isoformat()}: {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'卖出: {order.executed.price:.2f}')

    def next(self):
        if not self.position:
            # --- 买入逻辑 ---
            if self.params.strategy_type == 'SMA':
                if self.sma_cross > 0: # 金叉
                    self.buy()
            
            elif self.params.strategy_type == 'RSI':
                if self.rsi < self.params.rsi_low: # 超卖，买入
                    self.buy()
            
            elif self.params.strategy_type == 'MACD':
                if self.macd_cross > 0: # 金叉
                    self.buy()

        else:
            # --- 卖出逻辑 ---
            if self.params.strategy_type == 'SMA':
                if self.sma_cross < 0: # 死叉
                    self.sell()
            
            elif self.params.strategy_type == 'RSI':
                if self.rsi > self.params.rsi_high: # 超买，卖出
                    self.sell()
            
            elif self.params.strategy_type == 'MACD':
                if self.macd_cross < 0: # 死叉
                    self.sell()

# ==========================================
# 2. 数据获取函数 (封装 Yahoo 和 AkShare)
# ==========================================
def get_data(source, ticker, start_date, end_date):
    if source == "美股/港股 (Yahoo)":
        # Yahoo Finance 逻辑
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    
    elif source == "A股 (AkShare)":
        # AkShare 逻辑
        # 格式转换: 2020-01-01 -> 20200101
        s_str = start_date.strftime("%Y%m%d")
        e_str = end_date.strftime("%Y%m%d")
        
        try:
            # 获取 A 股日线数据
            stock_df = ak.stock_zh_a_hist(symbol=ticker, start_date=s_str, end_date=e_str, adjust="qfq")
            
            # 清洗数据以符合 Backtrader 格式
            # AkShare 返回列: 日期, 开盘, 收盘, 最高, 最低, 成交量...
            stock_df.rename(columns={
                '日期': 'date', '开盘': 'open', '收盘': 'close', 
                '最高': 'high', '最低': 'low', '成交量': 'volume'
            }, inplace=True)
            
            stock_df.index = pd.to_datetime(stock_df['date'])
            stock_df = stock_df[['open', 'high', 'low', 'close', 'volume']]
            return stock_df
        except Exception as e:
            st.error(f"AkShare 数据获取失败: {e}")
            return pd.DataFrame()

# ==========================================
# 3. Streamlit 主界面
# ==========================================
def main():
    st.set_page_config(page_title="全能量化系统", layout="wide")
    st.title("📈 全能量化交易系统 (A股+美股+多策略)")

    # --- 侧边栏配置 ---
    st.sidebar.header("1. 数据设置")
    data_source = st.sidebar.selectbox("数据源", ["美股/港股 (Yahoo)", "A股 (AkShare)"])
    
    if data_source == "美股/港股 (Yahoo)":
        default_ticker = "AAPL"
        st.sidebar.caption("示例: AAPL, TSLA, 0700.HK")
    else:
        default_ticker = "600519"
        st.sidebar.caption("示例: 600519 (茅台), 000001 (平安)")
        
    ticker = st.sidebar.text_input("股票代码", default_ticker)
    start_date = st.sidebar.date_input("开始日期", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("结束日期", datetime.date.today())
    start_cash = st.sidebar.number_input("初始资金", value=100000)

    st.sidebar.header("2. 策略选择")
    strategy_type = st.sidebar.selectbox("选择策略", ["SMA (双均线)", "RSI (相对强弱)", "MACD (指数平滑)"])
    
    # 根据选择显示不同参数
    pfast, pslow = 10, 30
    rsi_period, rsi_low, rsi_high = 14, 30, 70
    macd_fast, macd_slow, macd_signal = 12, 26, 9

    if strategy_type == "SMA (双均线)":
        pfast = st.sidebar.slider("快速均线", 5, 50, 10)
        pslow = st.sidebar.slider("慢速均线", 20, 200, 30)
        strat_code = 'SMA'
    elif strategy_type == "RSI (相对强弱)":
        rsi_period = st.sidebar.slider("RSI 周期", 5, 30, 14)
        rsi_low = st.sidebar.slider("超卖阈值 (买入)", 10, 40, 30)
        rsi_high = st.sidebar.slider("超买阈值 (卖出)", 60, 90, 70)
        strat_code = 'RSI'
    else:
        macd_fast = st.sidebar.slider("快线周期", 5, 20, 12)
        macd_slow = st.sidebar.slider("慢线周期", 20, 40, 26)
        macd_signal = st.sidebar.slider("信号线周期", 5, 20, 9)
        strat_code = 'MACD'

    # --- 运行逻辑 ---
    if st.sidebar.button("🚀 开始回测"):
        st.info(f"正在从 {data_source} 获取 {ticker} 数据...")
        
        # 1. 获取数据
        df = get_data(data_source, ticker, start_date, end_date)
        
        if df is None or df.empty:
            st.error("数据为空，请检查代码或日期。注意：A股代码通常为6位数字。")
            return

        # 2. 初始化 Cerebro
        cerebro = bt.Cerebro()
        
        # 3. 添加数据
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)

        # 4. 添加策略 (传入所有参数)
        cerebro.addstrategy(CompositeStrategy, 
                            strategy_type=strat_code,
                            pfast=pfast, pslow=pslow,
                            rsi_period=rsi_period, rsi_low=rsi_low, rsi_high=rsi_high,
                            macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal)

        cerebro.broker.setcash(start_cash)
        
        # 5. 运行
        initial_value = cerebro.broker.getvalue()
        results = cerebro.run()
        strat = results[0]
        final_value = cerebro.broker.getvalue()
        
        # 6. 展示结果
        pnl = final_value - initial_value
        col1, col2, col3 = st.columns(3)
        col1.metric("初始资金", f"{initial_value:,.0f}")
        col2.metric("最终资金", f"{final_value:,.0f}")
        col3.metric("净收益", f"{pnl:,.0f}", delta=f"{(pnl/initial_value)*100:.2f}%")

        # 7. 绘图
        st.subheader(f"策略可视化 ({strategy_type})")
        try:
            # volume=False 避免某些数据源缺失成交量导致报错
            figs = cerebro.plot(style='candlestick', volume=False)
            if figs:
                st.pyplot(figs[0][0])
        except Exception as e:
            st.warning(f"绘图部分出现问题: {e}")

        # 8. 日志
        if strat.log_list:
            with st.expander("查看详细交易日志"):
                st.dataframe(pd.DataFrame(strat.log_list, columns=["交易详情"]), use_container_width=True)
        else:
            st.info("该策略在选定时间段内未触发任何交易。")

if __name__ == '__main__':
    main()
