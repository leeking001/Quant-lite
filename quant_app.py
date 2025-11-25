import streamlit as st
import datetime
import pandas as pd
import yfinance as yf
import akshare as ak

# ==========================================
# 0. 系统配置与兼容性补丁
# ==========================================
import matplotlib
matplotlib.use('Agg') # 云端渲染必须
import matplotlib.pyplot as plt
import warnings
import matplotlib.dates

# 修复 Backtrader 在新版 Matplotlib 下的报错
if not hasattr(matplotlib.dates, 'warnings'):
    matplotlib.dates.warnings = warnings

# 设置绘图风格 (更现代的深色网格风格)
plt.style.use('bmh') 

import backtrader as bt

# ==========================================
# 1. 超级策略类 (包含5种策略)
# ==========================================
class MegaStrategy(bt.Strategy):
    params = (
        ('strategy_type', 'SMA'), 
        # SMA 参数
        ('pfast', 10), ('pslow', 30),
        # RSI 参数
        ('rsi_period', 14), ('rsi_low', 30), ('rsi_high', 70),
        # MACD 参数
        ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9),
        # Bollinger 参数
        ('boll_period', 20), ('boll_dev', 2.0),
        # Momentum 参数
        ('mom_period', 10),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.log_list = [] 

        # --- 初始化所有指标 ---
        # 1. SMA
        self.sma1 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.pfast)
        self.sma2 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.pslow)
        self.sma_cross = bt.indicators.CrossOver(self.sma1, self.sma2)

        # 2. RSI
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.params.rsi_period)

        # 3. MACD
        self.macd = bt.indicators.MACD(self.datas[0], 
            period_me1=self.params.macd_fast, period_me2=self.params.macd_slow, period_signal=self.params.macd_signal)
        self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        # 4. Bollinger Bands (布林带)
        self.boll = bt.indicators.BollingerBands(self.datas[0], period=self.params.boll_period, devfactor=self.params.boll_dev)

        # 5. Momentum (动量)
        self.mom = bt.indicators.Momentum(self.datas[0], period=self.params.mom_period)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        self.log_list.append(f'{dt.isoformat()}: {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'🟢 买入执行: 价格 {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'🔴 卖出执行: 价格 {order.executed.price:.2f}')

    def next(self):
        if not self.position:
            # --- 买入逻辑 ---
            if self.params.strategy_type == 'SMA' and self.sma_cross > 0:
                self.buy()
            elif self.params.strategy_type == 'RSI' and self.rsi < self.params.rsi_low:
                self.buy()
            elif self.params.strategy_type == 'MACD' and self.macd_cross > 0:
                self.buy()
            elif self.params.strategy_type == 'Bollinger' and self.dataclose < self.boll.lines.bot:
                # 价格跌破下轨，博反弹
                self.buy()
            elif self.params.strategy_type == 'Momentum' and self.mom > 0:
                # 动量转正
                self.buy()

        else:
            # --- 卖出逻辑 ---
            if self.params.strategy_type == 'SMA' and self.sma_cross < 0:
                self.sell()
            elif self.params.strategy_type == 'RSI' and self.rsi > self.params.rsi_high:
                self.sell()
            elif self.params.strategy_type == 'MACD' and self.macd_cross < 0:
                self.sell()
            elif self.params.strategy_type == 'Bollinger' and self.dataclose > self.boll.lines.top:
                # 价格突破上轨，止盈
                self.sell()
            elif self.params.strategy_type == 'Momentum' and self.mom < 0:
                self.sell()

# ==========================================
# 2. 数据获取模块
# ==========================================
@st.cache_data(ttl=3600) # 缓存数据1小时，避免重复下载
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
# 3. 界面内容辅助函数
# ==========================================
def show_strategy_guide():
    st.markdown("""
    ### 🧠 策略原理科普
    
    **1. SMA 双均线策略 (Trend Following)**
    *   **原理**: 利用短期均线和长期均线的交叉来判断趋势。
    *   **买入**: 短线(如10日) 上穿 长线(如30日) -> 金叉。
    *   **卖出**: 短线 下穿 长线 -> 死叉。
    *   **适用**: 趋势明显的单边行情。
    
    **2. RSI 相对强弱策略 (Mean Reversion)**
    *   **原理**: 测量价格变动的速度和变化，判断超买或超卖。
    *   **买入**: RSI < 30 (超卖，价格可能过低，准备反弹)。
    *   **卖出**: RSI > 70 (超买，价格可能过高，准备回调)。
    *   **适用**: 震荡市，箱体波动。
    
    **3. MACD 指数平滑策略 (Momentum)**
    *   **原理**: 结合了动量和趋势的指标。
    *   **买入**: 快线(DIF) 上穿 慢线(DEA)。
    *   **卖出**: 快线 下穿 慢线。
    *   **适用**: 捕捉中长期的趋势反转。

    **4. Bollinger Bands 布林带策略 (Volatility)**
    *   **原理**: 利用统计学原理，价格通常在上下轨之间波动。
    *   **买入**: 价格跌破下轨 (认为被低估，回归均值)。
    *   **卖出**: 价格突破上轨 (认为被高估)。
    *   **适用**: 震荡行情。

    **5. Momentum 动量策略 (Speed)**
    *   **原理**: 物理学惯性定律在股市的应用，强者恒强。
    *   **买入**: 当前价格高于N天前价格 (动量 > 0)。
    *   **卖出**: 当前价格低于N天前价格 (动量 < 0)。
    *   **适用**: 快速爆发的行情。
    """)

def show_user_manual():
    st.markdown("""
    ### 📖 新手使用说明书
    
    **第一步：选择数据源**
    *   **美股/港股**: 使用 Yahoo 数据。
        *   苹果: `AAPL`
        *   特斯拉: `TSLA`
        *   腾讯(港股): `0700.HK`
        *   比特币: `BTC-USD`
    *   **A股**: 使用 AkShare 数据。
        *   输入 **6位数字代码**。
        *   茅台: `600519`
        *   平安银行: `000001`
        *   宁德时代: `300750`

    **第二步：设置参数**
    *   **初始资金**: 模拟账户里的钱，建议设为 100,000。
    *   **策略参数**: 每一项策略都有对应的滑块，你可以拖动滑块来优化策略表现。

    **第三步：看懂结果**
    *   **净收益**: 赚了多少钱。
    *   **图表**: 
        *   **绿色箭头**: 模拟买入点。
        *   **红色箭头**: 模拟卖出点。
        *   **K线**: 股票走势。
    """)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    st.set_page_config(page_title="Quant Pro 量化系统", layout="wide", page_icon="📊")
    
    st.title("📊 Quant Pro 个人量化回测系统")
    st.caption("基于 Backtrader & Streamlit | 支持 A股/美股/港股/加密货币")

    # --- 侧边栏：控制面板 ---
    with st.sidebar:
        st.header("🎛️ 控制面板")
        
        # 1. 数据源
        st.subheader("1. 数据设置")
        data_source = st.selectbox("数据来源", ["美股/港股 (Yahoo)", "A股 (AkShare)"])
        
        if data_source == "美股/港股 (Yahoo)":
            ticker = st.text_input("代码 (Ticker)", "AAPL", help="例如: AAPL, TSLA, 0700.HK")
        else:
            ticker = st.text_input("代码 (Code)", "600519", help="请输入6位A股代码")
            
        col1, col2 = st.columns(2)
        start_date = col1.date_input("开始", datetime.date(2021, 1, 1))
        end_date = col2.date_input("结束", datetime.date.today())
        cash = st.number_input("初始资金", 100000, step=10000)

        # 2. 策略选择
        st.subheader("2. 策略配置")
        strat_map = {
            "SMA (双均线)": "SMA",
            "RSI (相对强弱)": "RSI",
            "MACD (指数平滑)": "MACD",
            "Bollinger (布林带)": "Bollinger",
            "Momentum (动量)": "Momentum"
        }
        selected_strat_name = st.selectbox("选择策略模型", list(strat_map.keys()))
        selected_strat_code = strat_map[selected_strat_name]

        # 动态参数
        params = {}
        if selected_strat_code == "SMA":
            params['pfast'] = st.slider("快速均线周期", 5, 60, 10)
            params['pslow'] = st.slider("慢速均线周期", 20, 200, 30)
        elif selected_strat_code == "RSI":
            params['rsi_period'] = st.slider("RSI周期", 5, 30, 14)
            params['rsi_low'] = st.slider("超卖阈值 (买)", 10, 40, 30)
            params['rsi_high'] = st.slider("超买阈值 (卖)", 60, 90, 70)
        elif selected_strat_code == "MACD":
            params['macd_fast'] = st.slider("快线", 5, 20, 12)
            params['macd_slow'] = st.slider("慢线", 20, 40, 26)
            params['macd_signal'] = st.slider("信号线", 5, 20, 9)
        elif selected_strat_code == "Bollinger":
            params['boll_period'] = st.slider("周期", 10, 50, 20)
            params['boll_dev'] = st.slider("标准差倍数", 1.0, 3.0, 2.0, 0.1)
        elif selected_strat_code == "Momentum":
            params['mom_period'] = st.slider("动量周期", 5, 30, 10)

        run_btn = st.button("🚀 开始回测", type="primary", use_container_width=True)

    # --- 主界面：标签页布局 ---
    tab1, tab2, tab3 = st.tabs(["📈 回测结果", "🧠 策略详解", "💡 新手指南"])

    # --- Tab 1: 回测逻辑 ---
    with tab1:
        if run_btn:
            with st.spinner(f"正在获取 {ticker} 数据并计算..."):
                # 1. 获取数据
                df = get_data(data_source, ticker, start_date, end_date)
                
                if df is None or df.empty:
                    st.error("❌ 数据获取失败！请检查股票代码是否正确，或日期范围是否有数据。")
                else:
                    # 2. 运行回测
                    cerebro = bt.Cerebro()
                    data = bt.feeds.PandasData(dataname=df)
                    cerebro.adddata(data)
                    
                    # 注入参数
                    cerebro.addstrategy(MegaStrategy, strategy_type=selected_strat_code, **params)
                    cerebro.broker.setcash(cash)
                    cerebro.broker.setcommission(commission=0.001) # 默认千一手续费

                    # 分析器
                    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

                    initial_val = cerebro.broker.getvalue()
                    results = cerebro.run()
                    strat = results[0]
                    final_val = cerebro.broker.getvalue()

                    # 3. 显示核心指标
                    pnl = final_val - initial_val
                    roi = (pnl / initial_val) * 100
                    
                    # 获取夏普和回撤
                    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio')
                    max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)

                    # 指标卡片
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("最终资产", f"${final_val:,.0f}")
                    m2.metric("净收益 (PnL)", f"${pnl:,.0f}", delta=f"{roi:.2f}%")
                    m3.metric("最大回撤", f"{max_dd:.2f}%", delta_color="inverse")
                    m4.metric("夏普比率", f"{sharpe:.2f}" if sharpe else "无")

                    # 4. 绘图 (优化版)
                    st.subheader("📊 策略可视化")
                    try:
                        # 调整绘图尺寸和风格
                        plt.rcParams['figure.figsize'] = [15, 8] 
                        figs = cerebro.plot(style='candlestick', volume=False, barup='green', bardown='red')
                        if figs:
                            st.pyplot(figs[0][0])
                    except Exception as e:
                        st.warning(f"绘图引擎提示: {e}")

                    # 5. 交易日志
                    with st.expander("📄 查看详细交易记录"):
                        if strat.log_list:
                            log_df = pd.DataFrame(strat.log_list, columns=["交易详情"])
                            st.dataframe(log_df, use_container_width=True)
                        else:
                            st.info("在此期间无交易触发。")
        else:
            st.info("👈 请在左侧设置参数并点击“开始回测”")

    # --- Tab 2: 策略详解 ---
    with tab2:
        show_strategy_guide()

    # --- Tab 3: 新手指南 ---
    with tab3:
        show_user_manual()

if __name__ == '__main__':
    main()
