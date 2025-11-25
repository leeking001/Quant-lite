import streamlit as st
import datetime
import pandas as pd
import yfinance as yf

# ==========================================
# 0. 兼容性补丁 (必须保留)
# ==========================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import warnings
import matplotlib.dates
# 修复 Backtrader 报错 AttributeError: module 'matplotlib.dates' has no attribute 'warnings'
if not hasattr(matplotlib.dates, 'warnings'):
    matplotlib.dates.warnings = warnings

import backtrader as bt

# ==========================================
# 1. 策略类定义
# ==========================================
class SmaCross(bt.Strategy):
    params = (
        ('pfast', 10), 
        ('pslow', 30), 
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.sma1 = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.pfast)
        self.sma2 = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.pslow)
        self.crossover = bt.indicators.CrossOver(self.sma1, self.sma2)
        self.log_list = [] 

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        self.log_list.append(f'{dt.isoformat()}: {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'卖出: {order.executed.price:.2f}')

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.sell()

# ==========================================
# 2. Streamlit 界面与主逻辑
# ==========================================
def main():
    st.set_page_config(page_title="个人量化回测系统", layout="wide")
    st.title("📈 个人量化交易系统 (最终修复版)")

    st.sidebar.header("⚙️ 参数设置")
    ticker = st.sidebar.text_input("股票代码", "AAPL")
    start_date = st.sidebar.date_input("开始日期", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("结束日期", datetime.date.today())
    start_cash = st.sidebar.number_input("初始资金", value=10000)
    
    pfast = st.sidebar.slider("快速均线", 5, 50, 10)
    pslow = st.sidebar.slider("慢速均线", 20, 200, 30)

    if st.sidebar.button("🚀 运行回测"):
        st.info(f"正在获取 {ticker} 数据...")
        
        cerebro = bt.Cerebro()
        cerebro.addstrategy(SmaCross, pfast=pfast, pslow=pslow)

        try:
            # --- 核心修复：数据下载与清洗 ---
            # 1. 下载数据
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # 2. 检查数据是否为空
            if df.empty:
                st.error("未获取到数据，请检查股票代码或日期范围。")
                return

            # 3. 扁平化列名 (解决 AttributeError 关键步骤)
            # 如果 yfinance 返回的是 MultiIndex (例如: ('Close', 'AAPL')), 我们只取第一层 'Close'
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 4. 确保数据包含 Backtrader 需要的列
            # 有时候 yfinance 的列名首字母可能不一致，这里不做强制转换，Backtrader 通常能识别 'Close', 'Open' 等
            
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data)
            # -----------------------------

        except Exception as e:
            st.error(f"数据处理错误: {e}")
            # 打印详细错误以便调试
            st.write(e)
            return

        cerebro.broker.setcash(start_cash)
        
        # 运行回测
        initial_value = cerebro.broker.getvalue()
        results = cerebro.run()
        strat = results[0]
        final_value = cerebro.broker.getvalue()
        
        # 显示结果
        col1, col2 = st.columns(2)
        col1.metric("初始资金", f"${initial_value:,.2f}")
        col2.metric("最终资金", f"${final_value:,.2f}", delta=f"{final_value-initial_value:.2f}")

        # 绘图
        st.subheader("策略图表")
        try:
            figs = cerebro.plot(style='candlestick', volume=False)
            if figs and len(figs) > 0:
                fig = figs[0][0]
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"绘图引擎出现兼容性警告 (不影响计算结果): {e}")

        # 日志
        if strat.log_list:
            st.subheader("交易记录")
            st.dataframe(pd.DataFrame(strat.log_list, columns=["日志"]), use_container_width=True)

if __name__ == '__main__':
    main()
