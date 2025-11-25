import streamlit as st
import backtrader as bt
import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# 设置 Matplotlib 后端，防止在某些环境下报错
import matplotlib
matplotlib.use('Agg')

# ==========================================
# 1. 策略类定义 (Strategy)
# ==========================================
class SmaCross(bt.Strategy):
    # 定义策略参数，可以在运行时通过 UI 修改
    params = (
        ('pfast', 10),  # 快速均线周期
        ('pslow', 30),  # 慢速均线周期
    )

    def __init__(self):
        # 初始化数据引用
        self.dataclose = self.datas[0].close
        
        # 初始化指标
        self.sma1 = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.pfast)
        self.sma2 = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.pslow)
        
        # 交叉信号：1为金叉（快线上穿慢线），-1为死叉
        self.crossover = bt.indicators.CrossOver(self.sma1, self.sma2)
        
        # 用于UI显示的日志列表
        self.log_list = [] 

    def log(self, txt, dt=None):
        ''' 日志记录函数 '''
        dt = dt or self.datas[0].datetime.date(0)
        self.log_list.append(f'{dt.isoformat()}: {txt}')

    def notify_order(self, order):
        ''' 订单状态通知 '''
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行, 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'卖出执行, 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')

    def next(self):
        ''' 每一个新K线数据的执行逻辑 '''
        # 如果没有持仓
        if not self.position:
            # 金叉买入
            if self.crossover > 0:
                self.log(f'产生买入信号 (金叉): {self.dataclose[0]:.2f}')
                self.buy()
        # 如果持有持仓
        else:
            # 死叉卖出
            if self.crossover < 0:
                self.log(f'产生卖出信号 (死叉): {self.dataclose[0]:.2f}')
                self.sell()

# ==========================================
# 2. Streamlit 界面与主逻辑
# ==========================================
def main():
    st.set_page_config(page_title="个人量化回测系统", layout="wide")
    st.title("📈 个人量化交易系统 (模拟器)")

    # --- 侧边栏：参数设置 ---
    st.sidebar.header("⚙️ 策略参数设置")
    
    # 股票代码输入
    ticker = st.sidebar.text_input("股票代码 (Yahoo格式)", "AAPL")
    
    # 日期范围
    start_date = st.sidebar.date_input("开始日期", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("结束日期", datetime.date.today())
    
    # 资金设置
    start_cash = st.sidebar.number_input("初始资金 ($)", value=10000)
    commission = st.sidebar.number_input("交易手续费 (例如 0.001 为千分之一)", value=0.001, format="%.4f")

    # 策略参数
    st.sidebar.subheader("均线策略参数")
    pfast = st.sidebar.slider("快速均线周期 (Fast SMA)", 5, 50, 10)
    pslow = st.sidebar.slider("慢速均线周期 (Slow SMA)", 20, 200, 30)

    run_btn = st.sidebar.button("🚀 运行回测")

    # --- 主界面逻辑 ---
    if run_btn:
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期")
            return

        st.info(f"正在下载 {ticker} 数据并运行策略...")

        # 1. 初始化 Cerebro 引擎
        cerebro = bt.Cerebro()

        # 2. 添加策略
        # 将UI获取的参数传递给策略
        cerebro.addstrategy(SmaCross, pfast=pfast, pslow=pslow)

        # 3. 获取数据
        try:
            data = bt.feeds.PandasData(
                dataname=yf.download(ticker, start=start_date, end=end_date, progress=False)
            )
            cerebro.adddata(data)
        except Exception as e:
            st.error(f"数据下载失败: {e}")
            return

        # 4. 设置资金与手续费
        cerebro.broker.setcash(start_cash)
        cerebro.broker.setcommission(commission=commission)

        # 5. 添加分析器 (Analyzer) 以获取详细结果
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        # 6. 运行回测
        initial_value = cerebro.broker.getvalue()
        results = cerebro.run()
        strat = results[0] # 获取策略实例
        final_value = cerebro.broker.getvalue()
        
        # --- 结果展示 ---
        
        # 1. 关键指标卡片
        pnl = final_value - initial_value
        pnl_pct = (pnl / initial_value) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("初始资金", f"${initial_value:,.2f}")
        col2.metric("最终资金", f"${final_value:,.2f}")
        col3.metric("净收益 (PnL)", f"${pnl:,.2f}", delta=f"{pnl_pct:.2f}%")
        
        # 获取夏普比率
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio')
        sharpe_text = f"{sharpe:.2f}" if sharpe else "无"
        col4.metric("夏普比率", sharpe_text)

        # 2. 绘制图表
        st.subheader("📊 策略图表 (买卖点标记)")
        # Backtrader 的绘图通常比较复杂，这里我们捕获 matplotlib 的 figure
        fig = cerebro.plot(style='candlestick', barup='green', bardown='red', volume=False)[0][0]
        st.pyplot(fig)

        # 3. 交易日志
        st.subheader("📝 交易日志")
        if strat.log_list:
            log_df = pd.DataFrame(strat.log_list, columns=["日志详情"])
            st.dataframe(log_df, use_container_width=True)
        else:
            st.write("在此期间无交易记录。")

        # 4. 交易统计详情
        st.subheader("📈 交易统计")
        trade_analysis = strat.analyzers.trades.get_analysis()
        
        if trade_analysis.get('total', {}).get('total', 0) > 0:
            total_trades = trade_analysis['total']['total']
            won_trades = trade_analysis['won']['total']
            lost_trades = trade_analysis['lost']['total']
            win_rate = (won_trades / total_trades) * 100
            
            st.write(f"**总交易次数:** {total_trades}")
            st.write(f"**胜率:** {win_rate:.2f}% (胜: {won_trades} / 负: {lost_trades})")
        else:
            st.write("无已完成的交易。")

if __name__ == '__main__':
    main()
