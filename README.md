# 📈 量化交易模拟器 (Quant Trading Simulator) V5.0

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)](https://streamlit.io/)
[![Backtrader](https://img.shields.io/badge/Backtrader-Powered-green)](https://www.backtrader.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**量化交易模拟器** 是一个基于 Python 的全功能量化回测与实战演练平台。它专为新手和进阶交易者设计，支持 **A股/美股/港股**，拥有 **零代码策略工厂**、**AI 智能回测点评** 以及独创的 **未来沙盘推演** 功能。

无需编程基础，像搭积木一样构建你的交易策略，并在平行宇宙中验证它的有效性！

---

## ✨ 核心功能 (Key Features)

### 1. 🚀 多市场策略回测
*   **全市场支持**：完美支持 **A股**（自动识别后缀，如 `600519` -> `600519.SS`）、**美股**（如 `AAPL`）、**港股**。
*   **经典策略库**：内置 **双均线 (SMA)**、**RSI 反转**、**布林带 (Bollinger)**、**海龟交易 (Turtle)**、**均值回归** 等经典模型。
*   **🛠️ 自定义策略工厂**：**零代码**！通过下拉菜单组合逻辑（例如：`当 [收盘价] > [20日均线] 时买入`）。

### 2. 🔮 未来沙盘推演 (V5.0 独家)
*   **蒙特卡洛模拟**：基于股票的历史波动率，利用随机漫步算法生成 **未来 180 天的虚拟行情**。
*   **平行宇宙测试**：点击“重置”，生成完全不同的未来走势。测试你的策略在牛市、熊市、震荡市中的生存能力。
*   **策略托管**：在演练模式下，策略自动判断买卖，你只需看着 K 线生长，验证策略表现。

### 3. 🤖 AI 智能顾问
*   **自动评分**：根据收益率、最大回撤、Alpha、夏普比率，给出 ⭐⭐⭐ 星级评分。
*   **诊断报告**：生成“人话”版分析报告（如“风险过高，建议开启止损”、“收益微薄，不如买理财”）。
*   **下一步建议**：明确告知该策略是应该“实盘尝试”还是“直接放弃”。

### 4. 📱 极致体验
*   **移动端适配**：专为手机优化的 UI，折叠式参数面板，原生表格报告。
*   **全中文界面**：所有金融术语（Alpha, Beta, Sharpe）均已汉化，并配有详细的 `?` 工具提示。
*   **专业报告**：集成 `QuantStats`，支持下载全中文的 HTML 深度体检报告。

---

## 🛠️ 安装与运行 (Installation)

### 1. 克隆仓库
```bash
git clone https://github.com/your-username/quant-simulator.git
cd quant-simulator
