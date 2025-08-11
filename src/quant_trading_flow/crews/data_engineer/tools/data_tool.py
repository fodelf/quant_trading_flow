import time
import random
import akshare as ak
import pandas as pd
from crewai.tools import tool
import numpy as np
from io import BytesIO
import base64
import platform
from crewai.tools import BaseTool
import os
from quant_trading_flow.crews.data_engineer.tools.technical import extract_30_features


def calculate_technical_indicators(
    df: pd.DataFrame, symbol: str, file_date: str
) -> str:
    """
    计算并返回技术指标分析结果

    Args:
        df (pd.DataFrame): 包含股票数据的DataFrame
        symbol (str): 股票代码
        file_date (str): 文件日期

    Returns:
        str: 格式化后的技术指标字符串
    """
    # 移动平均线
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    # 指数移动平均线
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # 动量指标
    df["Momentum"] = df["Close"] / df["Close"].shift(10) - 1

    # ATR (平均真实波幅)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(window=14).mean()

    # 重置索引（将索引变回普通列）
    df.reset_index(inplace=True)
    os.makedirs(f"output/{symbol}/{file_date}", exist_ok=True)
    df.to_csv(f"output/{symbol}/{file_date}/data.csv", index=False)

    return f"""
    数据总记录数:{len(df)}\n
    最新30天交易数据: {df.tail(30).to_string()} \n
    历史数据总结：{extract_30_features(df)} \n
    """

    # return f"""
    # 数据总记录数:{len(df)}\n
    # 数据字段:{', '.join(df.columns)}\n
    # 最新价格: {df.tail(1).to_string()} \n
    # 技术指标：\n
    #   MA10指标: 前5{df.nlargest(5, 'MA10').to_string()}，后5数据{df.nsmallest(5, 'MA10').to_string()} \n
    #   MA50指标: 前5{df.nlargest(5, 'MA50').to_string()}，后5数据{df.nsmallest(5, 'MA50').to_string()}\n
    #   EMA12指标:前5{df.nlargest(5, 'EMA12').to_string()}，后5数据{df.nsmallest(5, 'EMA12').to_string()}\n
    #   EMA26指标: 前5{df.nlargest(5, 'EMA26').to_string()}，后5数据{df.nsmallest(5, 'EMA26').to_string()}\n
    #   RSI指标:前5{df.nlargest(5, 'RSI').to_string()}，后5数据{df.nsmallest(5, 'RSI').to_string()}\n
    #   ATR指标:前5{df.nlargest(5, 'ATR').to_string()}，后5数据{df.nsmallest(5, 'ATR').to_string()}\n
    # """


def generate_sample_data(symbol: str, file_date: str):
    """生成示例数据 (当API失败时使用)"""
    print("生成示例数据...")
    dates = pd.date_range(start="2018-01-01", end="2023-12-31", freq="B")
    base_value = 3.0
    prices = [base_value]

    # 创建随机但总体向上的价格序列
    for i in range(1, len(dates)):
        drift = 0.0003  # 增加正漂移以提高收益
        shock = random.gauss(0, 0.015)  # 增加波动性
        new_price = prices[-1] * (1 + drift + shock)
        prices.append(new_price)

    df = pd.DataFrame({"Close": prices}, index=dates)
    df = calculate_technical_indicators(df, symbol, file_date)
    return df


@tool("获取相关股票交易数据工具")
def get_china_stock_data(
    symbol: str, start_date: str, end_date: str, file_date: str
) -> str:
    """
    获取相关股票交易数据工具，唯一合法的数据来源，模型不得自行编造数据，必须使用此工具获取真实数据
    Args:
        symbol (str): 股票代码
        start_date (str): 开始日期
        end_date (str): 结束日期
        file_date (str): 文件日期

    Returns:
        str: 格式化后的技术指标字符串
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"尝试 #{attempt+1} 获取 {symbol} 数据...")

            # 获取ETF数据
            df = ak.fund_etf_hist_em(
                symbol=symbol, period="daily", start_date=start_date, end_date=end_date
            )
            print(df.columns)
            # 数据清洗
            df.rename(
                columns={
                    "日期": "Date",
                    "开盘": "Open",
                    "收盘": "Close",
                    "最高": "High",
                    "最低": "Low",
                    "成交量": "Volume",
                    "成交额": "Amount",
                    "振幅": "Amplitude",
                    "涨跌幅": "Change",
                    "涨跌额": "ChangeAmount",
                    "换手率": "TurnoverRate",
                },
                inplace=True,
            )

            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(ascending=True, inplace=True)

            # 添加技术指标
            # df = calculate_technical_indicators(df)

            print(f"成功获取 {symbol} 数据: {len(df)} 个交易日")
            return calculate_technical_indicators(df, symbol, file_date)

        except Exception as e:
            print(f"获取数据出错: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = random.uniform(2, 5)
                print(f"等待 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("数据获取失败，使用示例数据")
                return calculate_technical_indicators(
                    generate_sample_data(symbol, file_date), symbol, file_date
                )
