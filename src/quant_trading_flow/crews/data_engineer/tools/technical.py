import pandas as pd
import numpy as np
import json


def calculate_price_features(df):
    """计算基础价格特征"""
    # 收益率特征
    df["Return"] = df["Close"].pct_change()

    features = {
        # 收益特征
        "total_return": df["Close"].iloc[-1] / df["Close"].iloc[0] - 1,
        "avg_daily_return": df["Return"].mean(),
        "positive_day_ratio": (df["Return"] > 0).mean(),
        # 波动特征
        "volatility_1y": df["Return"].rolling(252).std().mean() * np.sqrt(252),
        "max_daily_drop": df["Return"].min(),
        "max_drawdown": (df["Close"] / df["Close"].cummax() - 1).min(),
        # 价格分布
        "close_skewness": df["Close"].skew(),
        "close_kurtosis": df["Close"].kurtosis(),
        "high_low_ratio": df["High"].max() / df["Low"].min(),
    }
    return features


def calculate_technical_indicators(df):
    """手动实现技术指标"""
    features = {}

    # 1. 移动平均线
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    features["ma_cross"] = (df["MA_20"] > df["MA_50"]).iloc[-1]
    features["ma_distance"] = (df["Close"].iloc[-1] - df["MA_50"].iloc[-1]) / df[
        "MA_50"
    ].iloc[-1]

    # 2. RSI 相对强弱指标
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    features["rsi_current"] = df["RSI"].iloc[-1]
    features["rsi_extreme_days"] = ((df["RSI"] > 70) | (df["RSI"] < 30)).sum() / len(df)

    # 3. MACD 指标
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    features["macd_diff"] = (df["MACD"] - df["MACD_Signal"]).iloc[-1]
    features["macd_trend"] = (df["MACD"] > df["MACD_Signal"]).iloc[-5:].mean()

    # 4. 布林带
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["STD20"] = df["Close"].rolling(window=20).std()
    df["Bollinger_Upper"] = df["MA20"] + (df["STD20"] * 2)
    df["Bollinger_Lower"] = df["MA20"] - (df["STD20"] * 2)

    bollinger_width = (df["Bollinger_Upper"] - df["Bollinger_Lower"]) / df["MA20"]
    features["bollinger_width_avg"] = bollinger_width.mean()
    features["bollinger_position"] = (
        df["Close"].iloc[-1] - df["Bollinger_Lower"].iloc[-1]
    ) / (df["Bollinger_Upper"].iloc[-1] - df["Bollinger_Lower"].iloc[-1])

    # 5. 成交量指标
    df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
    features["volume_current"] = df["Volume"].iloc[-1] / df["Volume_MA20"].iloc[-1]
    features["volume_skew"] = df["Volume"].skew()

    # 6. 真实波动幅度 (ATR)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(window=14).mean()
    features["atr_current"] = df["ATR"].iloc[-1] / df["Close"].iloc[-1]

    return features


def calculate_time_features(df):
    """计算时间相关特征"""
    features = {}
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # 删除无效日期行
    df = df.dropna(subset=["Date"])
    # 将日期列设为索引
    df.set_index("Date", inplace=True)
    # 季节特征
    monthly_returns = df["Close"].resample("ME").last().pct_change()
    features["best_month"] = (
        monthly_returns.groupby(monthly_returns.index.month).mean().idxmax()
    )
    features["worst_month"] = (
        monthly_returns.groupby(monthly_returns.index.month).mean().idxmin()
    )

    # 周内效应
    df["Weekday"] = df.index.dayofweek
    weekday_returns = df.groupby("Weekday")["Return"].mean()
    features["best_weekday"] = weekday_returns.idxmax()
    features["worst_weekday"] = weekday_returns.idxmin()

    # 月份效应
    df["Month"] = df.index.month
    monthly_volatility = df.groupby("Month")["Return"].std()
    features["high_vol_month"] = monthly_volatility.idxmax()

    # # 节假日前效应
    # # 这里需要自定义节假日列表，以下为示例
    # holidays = ["2015-12-24", "2016-12-24", "2017-12-24", "2018-12-24", "2019-12-24"]
    # pre_holiday_returns = df[df.index.isin(holidays)]["Return"]
    # features["pre_holiday_return"] = (
    #     pre_holiday_returns.mean() if not pre_holiday_returns.empty else 0
    # )

    return features


def calculate_price_patterns(df):
    """识别价格形态特征"""
    features = {}

    # 缺口分析
    df["Gap"] = df["Open"] - df["Close"].shift(1)
    features["gap_up_days"] = (df["Gap"] > 0).sum() / len(df)
    features["gap_down_days"] = (df["Gap"] < 0).sum() / len(df)

    # 趋势强度
    features["uptrend_days"] = (df["Close"] > df["Close"].shift(5)).mean()
    features["downtrend_days"] = (df["Close"] < df["Close"].shift(5)).mean()

    # 支撑阻力
    resistance = df["High"].rolling(50).max()
    support = df["Low"].rolling(50).min()
    features["resistance_distance"] = (
        df["Close"].iloc[-1] - resistance.iloc[-1]
    ) / resistance.iloc[-1]
    features["support_distance"] = (
        df["Close"].iloc[-1] - support.iloc[-1]
    ) / support.iloc[-1]

    # 量价关系
    price_up = df["Return"] > 0
    volume_up = df["Volume"] > df["Volume"].shift(1)
    features["volume_confirmation"] = (price_up & volume_up).mean()

    return features


def extract_30_features(df):
    # 处理所有numpy数值类型
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = (
            df[col].astype(float)
            if "float" in str(df[col].dtype)
            else df[col].astype(int)
        )
    """提取30个核心特征"""
    features = {}

    # 合并所有特征
    features.update(calculate_price_features(df))
    features.update(calculate_technical_indicators(df))
    features.update(calculate_time_features(df))
    features.update(calculate_price_patterns(df))

    # 选择最重要的30个特征
    selected_features = {
        # 价格特征
        "total_return": features["total_return"],
        "volatility_1y": features["volatility_1y"],
        "max_drawdown": features["max_drawdown"],
        "close_skewness": features["close_skewness"],
        # 技术指标
        "rsi_current": features["rsi_current"],
        "rsi_extreme_days": features["rsi_extreme_days"],
        "macd_diff": features["macd_diff"],
        "bollinger_width_avg": features["bollinger_width_avg"],
        "bollinger_position": features["bollinger_position"],
        "atr_current": features["atr_current"],
        # 时间特征
        "best_month": features["best_month"],
        "worst_month": features["worst_month"],
        "best_weekday": features["best_weekday"],
        "high_vol_month": features["high_vol_month"],
        # 价格形态
        "gap_up_days": features["gap_up_days"],
        "gap_down_days": features["gap_down_days"],
        "uptrend_days": features["uptrend_days"],
        "resistance_distance": features["resistance_distance"],
        "support_distance": features["support_distance"],
        "volume_confirmation": features["volume_confirmation"],
        # 组合特征
        "risk_reward_ratio": (
            features["total_return"] / features["max_drawdown"]
            if features["max_drawdown"] != 0
            else 0
        ),
        "trend_strength": features["uptrend_days"] - features["downtrend_days"],
        "gap_imbalance": features["gap_up_days"] - features["gap_down_days"],
        "volume_price_correlation": df["Volume"].corr(df["Close"]),
        "price_momentum": (df["Close"].iloc[-1] / df["Close"].iloc[-20] - 1) * 100,
        "volatility_clustering": df["Return"].abs().autocorr(lag=1),
        "mean_reversion_tendency": (df["Close"] < df["MA_50"]).iloc[-10:].mean(),
    }

    # 确保返回30个特征
    return json.dumps(
        dict(list(selected_features.items())[:30]),
        default=lambda x: x.item() if isinstance(x, np.generic) else x,
    )
