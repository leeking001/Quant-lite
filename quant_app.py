import streamlit as st
import datetime
import pandas as pd
import yfinance as yf
import akshare as ak

# ==========================================
# 0. 系统配置与兼容性补丁
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
# 1. 超级策略类 (含风控逻辑)
# ==========================================
class MegaStrategy(bt.Strategy):
    params = (
        ('strategy_type', 'SMA'), 
        # 风控参数 (新增)
        ('use_risk_mgmt', False), # 是否开启风控
        ('stop_loss', 0.05),      # 止损百分比 (0.05 = 5%)
        ('take_profit', 0.10),    # 止盈百分比 (0.10 = 10%)
        
        # 策略参数
        ('pfast', 10), ('pslow', 30),
        ('rsi_period', 14), ('rsi_low', 30), ('rsi_high', 70),
        ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9),
        ('boll_period', 20), ('boll_dev', 2.0),
        ('mom_period', 10),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.log_list = [] 
        self.order = None # 记录当前未完成的订单

        # --- 初始化指标 ---
        self.sma1 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.pfast)
        self.sma2 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.pslow)
        self.sma_cross = bt.indicators.CrossOver(self.sma1, self.sma2)

        self.rsi = bt.indicators.RSI(self.datas[0], period=self.params.rsi_period)

        self.macd = bt.indicators.MACD(self.datas[0], 
            period_me1=self.params.macd_fast, period_me2=self.params.macd_slow, period_signal=self.params.macd_signal)
        self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        self.boll = bt.indicators.BollingerBands(self.datas[0], period=self.params.boll_period, devfactor=self.params.boll_dev)

        self.mom = bt.indicators.Momentum(self.datas[0], period=self.params.mom_period)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        self.log_list.append(f'{dt.isoformat()}: {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'🟢 买入执行: 价格 {order.executed.price:.2f}')
            elif order.issell():
                # 计算这笔交易的盈亏
                pnl = order.executed.pnl
                self.log(f'🔴 卖出执行: 价格 {order.executed.price:.2f} | 盈亏: {pnl:.2f}')
            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('⚠️ 订单被拒绝/取消/资金不足')
            self.order = None

    def next(self):
        # 如果有订单正在执行中，不进行下一步操作
        if self.order:
            return

        # ===========================
        # 🛡️ 风控逻辑 (优先于策略)
        # ===========================
        if self.position and self.params.use_risk_mgmt:
            # 获取持仓成本价
            buy_price = self.position.price
            current_price = self.dataclose[0]
            
            # 计算收益率
            pnl_pct = (current_price - buy_price) / buy_price

            # 1. 止损检查 (Stop Loss)
            if pnl_pct <= -self.params.stop_loss:
                self.log(f'🛡️ 触发止损! 当前亏损: {pnl_pct*100:.2f}%')
                self.close() # 清仓
                return # 强制结束本次循环，不再执行策略卖出逻辑

            # 2. 止盈检查 (Take Profit)
            if pnl_pct >= self.params.take_profit:
                self.log(f'💰 触发止盈! 当前收益: {pnl_pct*100:.2f}%')
                self.close() # 清仓
                return

        # ===========================
        # 📈 策略开平仓逻辑
        # ===========================
        if not self.position:
            # --- 买入逻辑 ---
            if self.params.strategy_type == 'SMA' and self.sma_cross > 0:
                self.buy()
            elif self.params.strategy_type == 'RSI' and self.rsi < self.params.rsi_low:
                self.buy()
            elif self.params.strategy_type == 'MACD' and self.macd_cross > 0:
                self.buy()
            elif self.params.strategy_type == 'Bollinger' and self.dataclose < self.boll.lines.bot:
                self.buy()
            elif self.params.strategy_type == 'Momentum' and self.mom > 0:
                self.buy()

        else:
            # --- 卖出逻辑 (技术指标卖出) ---
            # 只有在没触发风控的情况下，才检查技术指标卖出信号
            if self.params.strategy_type == 'SMA' and self.sma_cross < 0:
                self.sell()
            elif self.params.strategy_type == 'RSI' and self.rsi > self.params.rsi_high:
                self.sell()
            elif self.params.strategy_type == 'MACD' and self.macd_cross < 0:
                self.sell()
            elif self.params.strategy_type == 'Bollinger' and self.dataclose > self.boll.lines.top:
                self.sell()
            elif self.params.strategy_type == 'Momentum' and self.mom < 0:
                self.sell()

# ==========================================
# 2. 数据获取模块
# ==========================================
@st.cache_data(ttl=3600)
def get_data(source, ticker, start_date, end_date):
    try:
        if source == "美股/港股 (Yahoo)":
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        elif source == "A股 (AkShare)":
            s_str = start_date.strftime("%Y%m%d")
            e_str = end_date.strftime("%Y%m%d")
            stock_df = ak.stock_zh_a_hist(symbol=ticker, start_date=s_str, end_date=e_str, adjust="qfq")
            stock_df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
            stock_df.index = pd.to_datetime(stock_df['date'])
            return stock_df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        return None

# ==========================================
# 3. 辅助显示函数
# ==========================================
def show_strategy_guide():
    st.markdown("""
    ### 🛡️ 为什么需要风控？
    
    在量化交易中，**活下来**比**赚大钱**更重要。
    
    *   **止损 (Stop Loss)**: 类似于汽车的安全气囊。当亏损达到一定程度（如5%）时，无条件卖出。防止单次错误判断导致本金归零。
    *   **止盈 (Take Profit)**: 类似于落袋为安。当盈利达到预期（如10%）时，主动卖出。防止利润回撤，把赚到的钱变成亏损。
    
    **本系统逻辑**:
    风控逻辑的优先级 **高于** 技术指标。即使 MACD 还是金叉状态，但如果亏损触及止损线，系统也会强制卖出。
    """)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    st.set_page_config(page_title="Quant Pro 风控版", layout="wide", page_icon="🛡️")
    
    st.title("🛡️ Quant Pro 量化系统 (V1.1 风控版)")

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("🎛️ 控制面板")
        
        # 1. 数据
        st.subheader("1. 数据设置")
        data_source = st.selectbox("数据来源", ["美股/港股 (Yahoo)", "A股 (AkShare)"])
        if data_source == "美股/港股 (Yahoo)":
            ticker = st.text_input("代码", "AAPL")
        else:
            ticker = st.text_input("代码", "600519")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("开始", datetime.date(2021, 1, 1))
        end_date = col2.date_input("结束", datetime.date.today())
        cash = st.number_input("初始资金", 100000, step=10000)

        # 2. 策略
        st.subheader("2. 策略模型")
        strat_map = {"SMA (双均线)": "SMA", "RSI (相对强弱)": "RSI", "MACD (指数平滑)": "MACD", "Bollinger (布林带)": "Bollinger", "Momentum (动量)": "Momentum"}
        selected_strat_name = st.selectbox("选择策略", list(strat_map.keys()))
        selected_strat_code = strat_map[selected_strat_name]

        # 3. 风控 (新增核心功能)
        with st.expander("🛡️ 风控设置 (Stop Loss/Take Profit)", expanded=True):
            use_risk = st.checkbox("开启止盈止损", value=True)
            stop_loss_pct = st.slider("止损阈值 (跌幅)", 1, 20, 5) / 100.0
            take_profit_pct = st.slider("止盈阈值 (涨幅)", 5, 50, 15) / 100.0
            
            if use_risk:
                st.caption(f"逻辑: 亏损 > {stop_loss_pct*100:.0f}% 割肉，盈利 > {take_profit_pct*100:.0f}% 止盈")

        # 策略参数
        params = {}
        if selected_strat_code == "SMA":
            params['pfast'] = st.slider("快速均线", 5, 60, 10)
            params['pslow'] = st.slider("慢速均线", 20, 200, 30)
        elif selected_strat_code == "RSI":
            params['rsi_period'] = st.slider("RSI周期", 5, 30, 14)
            params['rsi_low'] = st.slider("超卖", 10, 40, 30)
            params['rsi_high'] = st.slider("超买", 60, 90, 70)
        elif selected_strat_code == "MACD":
            params['macd_fast'] = st.slider("快线", 5, 20, 12)
            params['macd_slow'] = st.slider("慢线", 20, 40, 26)
            params['macd_signal'] = st.slider("信号线", 5, 20, 9)
        elif selected_strat_code == "Bollinger":
            params['boll_period'] = st.slider("周期", 10, 50, 20)
            params['boll_dev'] = st.slider("倍数", 1.0, 3.0, 2.0)
        elif selected_strat_code == "Momentum":
            params['mom_period'] = st.slider("动量周期", 5, 30, 10)

        run_btn = st.button("🚀 开始回测", type="primary", use_container_width=True)

    # --- 主界面 ---
    tab1, tab2 = st.tabs(["📈 回测结果", "🧠 策略与风控说明"])

    with tab1:
        if run_btn:
            with st.spinner("正在回测中..."):
                df = get_data(data_source, ticker, start_date, end_date)
                if df is None or df.empty:
                    st.error("数据获取失败")
                else:
                    cerebro = bt.Cerebro()
                    cerebro.adddata(bt.feeds.PandasData(dataname=df))
                    
                    # 注入策略和风控参数
                    cerebro.addstrategy(MegaStrategy, 
                                        strategy_type=selected_strat_code,
                                        use_risk_mgmt=use_risk,
                                        stop_loss=stop_loss_pct,
                                        take_profit=take_profit_pct,
                                        **params)
                    
                    cerebro.broker.setcash(cash)
                    cerebro.broker.setcommission(commission=0.001)
                    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

                    initial_val = cerebro.broker.getvalue()
                    results = cerebro.run()
                    strat = results[0]
                    final_val = cerebro.broker.getvalue()

                    pnl = final_val - initial_val
                    roi = (pnl / initial_val) * 100
                    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio')
                    max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("最终资产", f"${final_val:,.0f}")
                    m2.metric("净收益", f"${pnl:,.0f}", delta=f"{roi:.2f}%")
                    m3.metric("最大回撤", f"{max_dd:.2f}%", delta_color="inverse")
                    m4.metric("夏普比率", f"{sharpe:.2f}" if sharpe else "无")

                    st.subheader("📊 策略可视化")
                    try:
                        plt.rcParams['figure.figsize'] = [15, 8] 
                        figs = cerebro.plot(style='candlestick', volume=False, barup='green', bardown='red')
                        if figs: st.pyplot(figs[0][0])
                    except: pass

                    with st.expander("📄 交易日志 (含风控触发记录)", expanded=True):
                        if strat.log_list:
                            log_df = pd.DataFrame(strat.log_list, columns=["详情"])
                            # 高亮显示止盈止损记录
                            def highlight_risk(val):
                                color = 'red' if '止损' in val else 'green' if '止盈' in val else 'black'
                                return f'color: {color}'
                            st.dataframe(log_df.style.map(highlight_risk), use_container_width=True)
                        else:
                            st.info("无交易记录")

    with tab2:
        show_strategy_guide()

if __name__ == '__main__':
    main()
