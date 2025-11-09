import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings("ignore")
import os
from datetime import datetime, timedelta
import logging


class AggressiveStockPredictor:
    def __init__(self, symbol, file_date):
        self.symbol = symbol
        self.file_date = file_date
        self.data_path = f"output/{symbol}/{file_date}/data.csv"

        # 涨跌停参数
        self.limit_up_rate = 0.1
        self.limit_down_rate = -0.1

        # 设置日志
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def aggressive_feature_engineering(self, df):
        """激进特征工程 - 专门针对强势股票"""
        df = df.copy()

        # 确保数据类型
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        open_price = df["Open"]

        # === 核心价格特征 ===
        df["Price_Range"] = (high - low) / close
        df["Body_Ratio"] = abs(close - open_price) / (high - low + 1e-8)
        df["Upper_Shadow"] = (high - np.maximum(open_price, close)) / close
        df["Lower_Shadow"] = (np.minimum(open_price, close) - low) / close

        # === 移动平均系统 ===
        windows = [3, 5, 10, 20, 30]
        for window in windows:
            df[f"MA_{window}"] = close.rolling(window=window, min_periods=1).mean()
            df[f"Volume_MA_{window}"] = volume.rolling(
                window=window, min_periods=1
            ).mean()

        # 均线排列强度
        df["MA_Strength"] = (
            (df["MA_5"] > df["MA_10"]).astype(int)
            + (df["MA_10"] > df["MA_20"]).astype(int)
            + (df["MA_20"] > df["MA_30"]).astype(int)
        ) / 3

        # === 动量指标 ===
        # 多重RSI
        for period in [6, 14, 21]:
            delta = close.diff()
            gain = (
                (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            )
            loss = (
                (-delta.where(delta < 0, 0))
                .rolling(window=period, min_periods=1)
                .mean()
            )
            rs = gain / (loss + 1e-8)
            df[f"RSI_{period}"] = 100 - (100 / (1 + rs))

        # MACD系统
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df["MACD"] = ema_12 - ema_26
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # 快速MACD
        ema_6 = close.ewm(span=6).mean()
        ema_19 = close.ewm(span=19).mean()
        df["MACD_Fast"] = ema_6 - ema_19

        # === 波动率指标 ===
        df["BB_Middle"] = close.rolling(window=20, min_periods=1).mean()
        bb_std = close.rolling(window=20, min_periods=1).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
        df["BB_Position"] = (close - df["BB_Lower"]) / (
            df["BB_Upper"] - df["BB_Lower"] + 1e-8
        )

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(window=14, min_periods=1).mean()

        # === 成交量分析 ===
        df["Volume_Ratio_5"] = volume / df["Volume_MA_5"]
        df["Volume_Ratio_10"] = volume / df["Volume_MA_10"]

        # 量价配合
        df["Volume_Price_Sync"] = (
            ((close > close.shift(1)) & (volume > volume.shift(1)))
            | ((close < close.shift(1)) & (volume < volume.shift(1)))
        ).astype(int)

        # OBV
        obv = (volume * (~close.diff().le(0)).astype(int)).cumsum()
        df["OBV"] = obv
        df["OBV_MA_21"] = obv.rolling(window=21, min_periods=1).mean()

        # === 价格动量 ===
        for period in [1, 3, 5, 10]:
            df[f"Return_{period}"] = close.pct_change(period)

        df["Momentum_5"] = close / close.shift(5) - 1
        df["Momentum_10"] = close / close.shift(10) - 1
        df["Acceleration"] = df["Momentum_5"] - df["Momentum_5"].shift(1)

        # === 支撑阻力 ===
        df["Resistance_20"] = high.rolling(window=20, min_periods=1).max()
        df["Support_20"] = low.rolling(window=20, min_periods=1).min()
        df["Distance_to_Resistance"] = (df["Resistance_20"] - close) / close
        df["Distance_to_Support"] = (close - df["Support_20"]) / close

        # === 涨跌停智能检测 ===
        df["Prev_Close"] = close.shift(1).fillna(method="bfill")
        df["Limit_Up_Price"] = df["Prev_Close"] * (1 + self.limit_up_rate)
        df["Limit_Down_Price"] = df["Prev_Close"] * (1 + self.limit_down_rate)

        # 精确涨停识别
        df["Is_Limit_Up"] = (
            (abs(high - df["Limit_Up_Price"]) / df["Limit_Up_Price"] < 0.002)
            & (df["Amplitude"] < 0.03)
            & (close > open_price * 0.995)
        ).astype(int)

        df["Is_Limit_Down"] = (
            (abs(low - df["Limit_Down_Price"]) / df["Limit_Down_Price"] < 0.002)
            & (df["Amplitude"] < 0.03)
            & (close < open_price * 1.005)
        ).astype(int)

        # 连续涨停计数
        df["Limit_Up_Streak"] = 0
        current_streak = 0
        for i in range(len(df)):
            if df["Is_Limit_Up"].iloc[i]:
                current_streak += 1
            else:
                current_streak = 0
            df.loc[df.index[i], "Limit_Up_Streak"] = current_streak

        # 涨停强度分析
        df["Limit_Strength_Volume"] = volume / df["Volume_MA_5"]
        df["Limit_Strength_Price"] = (close - open_price) / (high - low + 1e-8)

        # === 市场情绪 ===
        df["Volatility_10"] = close.rolling(window=10, min_periods=1).std()
        df["Volatility_20"] = close.rolling(window=20, min_periods=1).std()
        df["Volatility_Ratio"] = df["Volatility_10"] / (df["Volatility_20"] + 1e-8)

        # 趋势强度
        df["Trend_Strength"] = (
            (close - df["MA_20"]) / df["MA_20"] * df["Volume_Ratio_5"]
        )

        # === 价格位置 ===
        for ma in [5, 10, 20, 30]:
            df[f"Close_vs_MA{ma}"] = (close - df[f"MA_{ma}"]) / df[f"MA_{ma}"]

        # 价格通道
        df["Price_Channel_Position"] = (close - low.rolling(20).min()) / (
            high.rolling(20).max() - low.rolling(20).min() + 1e-8
        )

        return df

    def create_targets(self, df):
        """创建目标变量"""
        df = df.copy()

        # 基础目标
        df["target_next_low"] = df["Low"].shift(-1)
        df["target_next2_high"] = df["High"].shift(-2)
        df["target_next2_low"] = df["Low"].shift(-2)

        # 上涨概率
        df["target_next2_up"] = (
            (df["Close"].shift(-2) > df["Close"].shift(-1))
            & (df["Close"].shift(-2) > 0)
        ).astype(int)

        return df

    def train_aggressive_models(self, X, y_dict):
        """训练激进模型"""
        models = {}

        # 数据清理
        valid_mask = ~X.isnull().any(axis=1)
        for y in y_dict.values():
            valid_mask = valid_mask & ~y.isnull()

        X_clean = X[valid_mask]

        if len(X_clean) < 50:
            raise ValueError(f"训练数据不足，需要至少50条，当前只有{len(X_clean)}条")

        self.logger.info(f"激进模型训练，样本数量: {len(X_clean)}")

        # 训练各个目标的模型
        for target_name, y in y_dict.items():
            y_clean = y[valid_mask]

            if target_name in ["next_low", "next2_high", "next2_low"]:
                model = GradientBoostingRegressor(
                    n_estimators=400,
                    learning_rate=0.03,
                    max_depth=8,
                    min_samples_split=12,
                    min_samples_leaf=6,
                    subsample=0.7,
                    random_state=42,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                )

            # 训练模型
            model.fit(X_clean, y_clean)
            models[target_name] = model

            # 计算训练精度
            train_pred = model.predict(X_clean)
            if target_name == "next2_up":
                accuracy = np.mean((train_pred > 0.5) == y_clean)
                self.logger.info(f"{target_name} 模型精度: {accuracy:.4f}")
            else:
                mape = np.mean(np.abs(train_pred - y_clean) / (y_clean + 1e-8))
                self.logger.info(f"{target_name} 模型MAPE: {mape:.4f}")

        return models

    def aggressive_limit_prediction(self, models, last_data, current_close, df_history):
        """激进涨停预测逻辑"""
        # 基础模型预测
        base_predictions = {}
        for target_name, model in models.items():
            pred_value = model.predict(last_data)[0]
            base_predictions[target_name] = max(0.01, pred_value)

        # 当前状态深度分析
        is_limit_up = (
            last_data["Is_Limit_Up"].iloc[0] == 1
            if "Is_Limit_Up" in last_data.columns
            else False
        )
        limit_streak = (
            last_data["Limit_Up_Streak"].iloc[0]
            if "Limit_Up_Streak" in last_data.columns
            else 0
        )
        rsi_6 = last_data["RSI_6"].iloc[0] if "RSI_6" in last_data.columns else 50
        rsi_14 = last_data["RSI_14"].iloc[0] if "RSI_14" in last_data.columns else 50
        volume_ratio = (
            last_data["Volume_Ratio_5"].iloc[0]
            if "Volume_Ratio_5" in last_data.columns
            else 1
        )
        macd_hist = (
            last_data["MACD_Hist"].iloc[0] if "MACD_Hist" in last_data.columns else 0
        )
        ma_strength = (
            last_data["MA_Strength"].iloc[0]
            if "MA_Strength" in last_data.columns
            else 0
        )

        # 理论价格限制
        limit_up = current_close * (1 + self.limit_up_rate)
        next_limit_up = limit_up * (1 + self.limit_up_rate)

        self.logger.info(
            f"激进分析: 涨停={is_limit_up}, 连续={limit_streak}天, RSI6={rsi_6:.1f}, RSI14={rsi_14:.1f}, 量比={volume_ratio:.2f}, MACD={macd_hist:.4f}, MA强度={ma_strength:.2f}"
        )

        # === 激进涨停场景处理 ===
        if is_limit_up:
            # 分析历史强势表现
            strong_days = df_history[
                (df_history["Close"] > df_history["MA_20"])
                & (df_history["Volume_Ratio_5"] > 1.2)
                & (df_history["RSI_14"] < 80)
            ]

            if len(strong_days) > 0:
                # 计算强势后的平均涨幅
                strong_returns = []
                for i in range(len(strong_days) - 1):
                    if strong_days.index[i] + 2 in df_history.index:
                        two_day_return = (
                            df_history.loc[strong_days.index[i] + 2, "Close"]
                            / strong_days.iloc[i]["Close"]
                            - 1
                        )
                        strong_returns.append(two_day_return)

                if len(strong_returns) > 0:
                    avg_strong_return = np.mean(strong_returns)
                    max_strong_return = np.max(strong_returns)
                    self.logger.info(
                        f"历史强势后2日平均收益: {avg_strong_return:.4f}, 最大收益: {max_strong_return:.4f}"
                    )

            # 根据技术指标强度调整预测
            tech_strength = 0
            if rsi_6 < 75:
                tech_strength += 1
            if macd_hist > 0:
                tech_strength += 1
            if ma_strength > 0.6:
                tech_strength += 1
            if volume_ratio > 1.5:
                tech_strength += 1

            self.logger.info(f"技术指标强度: {tech_strength}/4")

            if limit_streak == 1:
                # 第一次涨停
                if tech_strength >= 3:
                    # 技术指标强势，预测继续大涨
                    base_predictions["next_low"] = max(
                        base_predictions["next_low"], current_close * 1.06
                    )
                    base_predictions["next2_high"] = min(
                        base_predictions["next2_high"], next_limit_up * 1.02
                    )  # 超过涨停价2%
                    base_predictions["next2_low"] = max(
                        base_predictions["next2_low"], current_close * 1.04
                    )
                    base_predictions["next2_up"] = min(
                        base_predictions["next2_up"] * 1.6, 0.95
                    )
                elif tech_strength >= 2:
                    # 技术指标中等强势
                    base_predictions["next_low"] = max(
                        base_predictions["next_low"], current_close * 1.04
                    )
                    base_predictions["next2_high"] = min(
                        base_predictions["next2_high"], next_limit_up
                    )
                    base_predictions["next2_low"] = max(
                        base_predictions["next2_low"], current_close * 1.02
                    )
                    base_predictions["next2_up"] = min(
                        base_predictions["next2_up"] * 1.4, 0.9
                    )
                else:
                    # 一般涨停
                    base_predictions["next_low"] = max(
                        base_predictions["next_low"], current_close * 1.02
                    )
                    base_predictions["next2_high"] = min(
                        base_predictions["next2_high"], limit_up * 1.05
                    )
                    base_predictions["next2_up"] = base_predictions["next2_up"] * 1.2

            elif limit_streak == 2:
                # 第二次涨停
                if tech_strength >= 3:
                    # 继续强势涨停
                    base_predictions["next_low"] = max(
                        base_predictions["next_low"], current_close * 1.08
                    )
                    base_predictions["next2_high"] = min(
                        base_predictions["next2_high"], next_limit_up * 1.05
                    )
                    base_predictions["next2_low"] = max(
                        base_predictions["next2_low"], current_close * 1.05
                    )
                    base_predictions["next2_up"] = min(
                        base_predictions["next2_up"] * 1.8, 0.98
                    )
                elif tech_strength >= 2:
                    base_predictions["next_low"] = max(
                        base_predictions["next_low"], current_close * 1.05
                    )
                    base_predictions["next2_high"] = min(
                        base_predictions["next2_high"], next_limit_up * 1.02
                    )
                    base_predictions["next2_low"] = max(
                        base_predictions["next2_low"], current_close * 1.03
                    )
                    base_predictions["next2_up"] = min(
                        base_predictions["next2_up"] * 1.5, 0.92
                    )
                else:
                    base_predictions["next_low"] = max(
                        base_predictions["next_low"], current_close * 1.03
                    )
                    base_predictions["next2_high"] = min(
                        base_predictions["next2_high"], limit_up * 1.08
                    )
                    base_predictions["next2_up"] = base_predictions["next2_up"] * 1.3

            elif limit_streak >= 3:
                # 多次涨停
                if volume_ratio > 2.0:
                    # 巨量涨停，可能继续强势
                    base_predictions["next_low"] = max(
                        base_predictions["next_low"], current_close * 1.05
                    )
                    base_predictions["next2_high"] = min(
                        base_predictions["next2_high"], next_limit_up * 1.03
                    )
                    base_predictions["next2_low"] = max(
                        base_predictions["next2_low"], current_close * 1.03
                    )
                    base_predictions["next2_up"] = min(
                        base_predictions["next2_up"] * 1.4, 0.9
                    )
                else:
                    # 缩量涨停，风险高
                    base_predictions["next_low"] = max(
                        base_predictions["next_low"], current_close * 1.01
                    )
                    base_predictions["next2_high"] = min(
                        base_predictions["next2_high"], limit_up * 1.02
                    )
                    base_predictions["next2_up"] = max(
                        base_predictions["next2_up"] * 0.7, 0.4
                    )

        # === 非涨停但强势的情况 ===
        elif (
            not is_limit_up
            and current_close > last_data["MA_20"].iloc[0]
            and volume_ratio > 1.2
        ):
            # 突破均线且放量
            if tech_strength >= 2:
                base_predictions["next2_high"] = min(
                    base_predictions["next2_high"] * 1.1, next_limit_up
                )
                base_predictions["next2_up"] = min(
                    base_predictions["next2_up"] * 1.2, 0.85
                )

        # === 强制价格合理性 ===
        base_predictions["next_low"] = min(
            base_predictions["next_low"], base_predictions["next2_high"] * 0.95
        )
        base_predictions["next2_low"] = min(
            base_predictions["next2_low"], base_predictions["next2_high"] * 0.95
        )

        # === 应用涨跌停限制 ===
        base_predictions["next_low"] = max(
            min(base_predictions["next_low"], limit_up), current_close * 0.92
        )
        base_predictions["next2_high"] = max(
            min(base_predictions["next2_high"], next_limit_up * 1.1),
            current_close * 1.1,
        )
        base_predictions["next2_low"] = max(
            min(base_predictions["next2_low"], next_limit_up), current_close * 0.92
        )

        # === 确保价格序列合理 ===
        base_predictions["next_low"] = min(
            base_predictions["next_low"], base_predictions["next2_low"]
        )
        base_predictions["next2_high"] = max(
            base_predictions["next2_high"], base_predictions["next2_low"]
        )

        # === 概率限制 ===
        base_predictions["next2_up"] = max(
            min(base_predictions["next2_up"], 0.98), 0.05
        )

        return base_predictions

    def calculate_aggressive_confidence(self, df, predictions, last_data):
        """计算激进置信度"""
        confidences = {}

        # 基础置信度
        recent_volatility = df["Close"].pct_change().tail(10).std()
        base_confidence = max(0.7, 0.9 - recent_volatility * 4)

        # 技术指标增强
        rsi_6 = last_data["RSI_6"].iloc[0] if "RSI_6" in last_data.columns else 50
        rsi_14 = last_data["RSI_14"].iloc[0] if "RSI_14" in last_data.columns else 50
        macd_hist = (
            last_data["MACD_Hist"].iloc[0] if "MACD_Hist" in last_data.columns else 0
        )
        ma_strength = (
            last_data["MA_Strength"].iloc[0]
            if "MA_Strength" in last_data.columns
            else 0
        )

        # RSI置信度增强
        if 40 <= rsi_6 <= 60 and 40 <= rsi_14 <= 60:
            rsi_boost = 1.3
        elif 30 <= rsi_6 <= 70 and 30 <= rsi_14 <= 70:
            rsi_boost = 1.1
        else:
            rsi_boost = 0.9

        # MACD置信度增强
        if macd_hist > 0:
            macd_boost = 1.2
        else:
            macd_boost = 0.9

        # 均线排列置信度
        if ma_strength > 0.6:
            ma_boost = 1.2
        else:
            ma_boost = 1.0

        # 涨停状态
        is_limit_up = (
            last_data["Is_Limit_Up"].iloc[0] == 1
            if "Is_Limit_Up" in last_data.columns
            else False
        )
        if is_limit_up:
            limit_boost = 1.3
        else:
            limit_boost = 1.0

        enhanced_confidence = (
            base_confidence * rsi_boost * macd_boost * ma_boost * limit_boost
        )

        # 价格关系检查
        price_consistency = 1.0
        if (
            predictions["next_low"] < predictions["next2_high"]
            and predictions["next2_low"] < predictions["next2_high"]
            and predictions["next_low"] <= predictions["next2_low"]
        ):
            price_consistency = 1.4
        else:
            price_consistency = 0.8

        # 各个预测的置信度
        price_targets = ["next_low", "next2_high", "next2_low"]
        for target in price_targets:
            # 价格变动合理性
            price_ratio = predictions[target] / df["Close"].iloc[-1]
            if 0.98 <= price_ratio <= 1.02:
                rationality = 1.4
            elif 0.95 <= price_ratio <= 1.05:
                rationality = 1.2
            elif 0.92 <= price_ratio <= 1.08:
                rationality = 1.0
            else:
                rationality = 0.8

            confidences[target] = min(
                0.98, enhanced_confidence * rationality * price_consistency
            )

        # 上涨概率置信度
        prob = predictions["next2_up"]
        if 0.4 <= prob <= 0.6:
            prob_confidence = 1.3
        elif 0.3 <= prob <= 0.7:
            prob_confidence = 1.1
        else:
            prob_confidence = 0.9

        confidences["next2_up"] = min(0.95, enhanced_confidence * prob_confidence)

        return confidences

    def run_strategy_development(self):
        """激进策略开发"""
        try:
            self.logger.info(f"开始激进股票预测 - 股票: {self.symbol}")

            # 读取数据
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

            df = pd.read_csv(self.data_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

            self.logger.info(f"数据读取成功: {len(df)} 条记录")

            if len(df) < 60:
                raise ValueError("数据不足，至少需要60个交易日")

            # 激进特征工程
            self.logger.info("进行激进特征工程...")
            df_featured = self.aggressive_feature_engineering(df)

            # 创建目标变量
            self.logger.info("创建目标变量...")
            df_targeted = self.create_targets(df_featured)

            # 选择特征
            base_features = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Amount",
                "Amplitude",
                "Change",
                "ChangeAmount",
                "TurnoverRate",
            ]

            tech_features = [
                "Price_Range",
                "Body_Ratio",
                "Upper_Shadow",
                "Lower_Shadow",
                "MA_3",
                "MA_5",
                "MA_10",
                "MA_20",
                "MA_30",
                "Volume_MA_3",
                "Volume_MA_5",
                "Volume_MA_10",
                "Volume_MA_20",
                "Volume_MA_30",
                "MA_Strength",
                "RSI_6",
                "RSI_14",
                "RSI_21",
                "MACD",
                "MACD_Signal",
                "MACD_Hist",
                "MACD_Fast",
                "BB_Upper",
                "BB_Lower",
                "BB_Middle",
                "BB_Width",
                "BB_Position",
                "ATR",
                "Volume_Ratio_5",
                "Volume_Ratio_10",
                "Volume_Price_Sync",
                "OBV",
                "OBV_MA_21",
                "Return_1",
                "Return_3",
                "Return_5",
                "Return_10",
                "Momentum_5",
                "Momentum_10",
                "Acceleration",
                "Resistance_20",
                "Support_20",
                "Distance_to_Resistance",
                "Distance_to_Support",
                "Is_Limit_Up",
                "Is_Limit_Down",
                "Limit_Up_Streak",
                "Limit_Strength_Volume",
                "Limit_Strength_Price",
                "Volatility_10",
                "Volatility_20",
                "Volatility_Ratio",
                "Trend_Strength",
                "Close_vs_MA5",
                "Close_vs_MA10",
                "Close_vs_MA20",
                "Close_vs_MA30",
                "Price_Channel_Position",
            ]

            # 只选择存在的特征
            available_features = []
            for feature in base_features + tech_features:
                if feature in df_targeted.columns:
                    available_features.append(feature)

            X = df_targeted[available_features]

            # 目标变量
            y_dict = {
                "next_low": df_targeted["target_next_low"],
                "next2_high": df_targeted["target_next2_high"],
                "next2_low": df_targeted["target_next2_low"],
                "next2_up": df_targeted["target_next2_up"],
            }

            # 清理数据
            valid_mask = ~X.isnull().any(axis=1)
            for y in y_dict.values():
                valid_mask = valid_mask & ~y.isnull()

            X_clean = X[valid_mask]
            y_clean_dict = {}
            for key, y in y_dict.items():
                y_clean_dict[key] = y[valid_mask]

            if len(X_clean) < 50:
                raise ValueError("有效训练数据不足50条")

            self.logger.info(
                f"最终训练数据: {len(X_clean)} 条, 特征: {len(available_features)} 个"
            )

            # 训练激进模型
            models = self.train_aggressive_models(X_clean, y_clean_dict)

            # 进行激进预测
            last_data = X_clean.iloc[-1:].copy()
            current_close = df_targeted["Close"].iloc[-1]

            predictions = self.aggressive_limit_prediction(
                models, last_data, current_close, df_targeted
            )

            # 计算激进置信度
            confidences = self.calculate_aggressive_confidence(
                df_targeted, predictions, last_data
            )

            # 准备最终结果
            limit_streak = (
                last_data["Limit_Up_Streak"].iloc[0]
                if "Limit_Up_Streak" in last_data.columns
                else 0
            )
            is_limit_up = (
                last_data["Is_Limit_Up"].iloc[0] == 1
                if "Is_Limit_Up" in last_data.columns
                else False
            )
            rsi_6 = last_data["RSI_6"].iloc[0] if "RSI_6" in last_data.columns else 50
            rsi_14 = (
                last_data["RSI_14"].iloc[0] if "RSI_14" in last_data.columns else 50
            )
            macd_hist = (
                last_data["MACD_Hist"].iloc[0]
                if "MACD_Hist" in last_data.columns
                else 0
            )
            ma_strength = (
                last_data["MA_Strength"].iloc[0]
                if "MA_Strength" in last_data.columns
                else 0
            )

            result = {
                "symbol": self.symbol,
                "prediction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_trading_date": df_targeted["Date"].iloc[-1].strftime("%Y-%m-%d"),
                "last_close": round(float(current_close), 2),
                "predictions": {
                    "next_day_low": {
                        "value": round(float(predictions["next_low"]), 2),
                        "confidence": round(float(confidences["next_low"]), 3),
                    },
                    "next_next_day_high": {
                        "value": round(float(predictions["next2_high"]), 2),
                        "confidence": round(float(confidences["next2_high"]), 3),
                    },
                    "next_next_day_low": {
                        "value": round(float(predictions["next2_low"]), 2),
                        "confidence": round(float(confidences["next2_low"]), 3),
                    },
                    "next_next_day_up_probability": {
                        "value": round(float(predictions["next2_up"]), 3),
                        "confidence": round(float(confidences["next2_up"]), 3),
                    },
                },
                "technical_analysis": {
                    "current_trend": (
                        "强势上涨"
                        if is_limit_up
                        else (
                            "上涨"
                            if df_targeted["Close"].iloc[-1]
                            > df_targeted["MA_20"].iloc[-1]
                            else "下跌"
                        )
                    ),
                    "limit_situation": (
                        f"连续涨停{int(limit_streak)}天" if is_limit_up else "正常"
                    ),
                    "rsi_6_status": (
                        "超买" if rsi_6 > 70 else "超卖" if rsi_6 < 30 else "中性"
                    ),
                    "rsi_14_status": (
                        "超买" if rsi_14 > 70 else "超卖" if rsi_14 < 30 else "中性"
                    ),
                    "macd_signal": "金叉" if macd_hist > 0 else "死叉",
                    "ma_strength": f"{ma_strength:.2f}",
                    "volatility": round(
                        float(df_targeted["Close"].pct_change().std()), 4
                    ),
                    "momentum": (
                        "强势" if df_targeted["Momentum_5"].iloc[-1] > 0.05 else "一般"
                    ),
                },
                "model_info": {
                    "training_samples": len(X_clean),
                    "feature_count": len(available_features),
                    "prediction_quality": (
                        "VERY_HIGH" if min(confidences.values()) > 0.8 else "HIGH"
                    ),
                },
                "success": True,
            }

            self.logger.info("激进预测完成!")
            return result

        except Exception as e:
            self.logger.error(f"激进预测过程出错: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return {"error": str(e), "symbol": self.symbol, "success": False}


def run_strategy_development(symbol, file_date):
    """
    激进股票预测策略

    参数:
    symbol: 股票代码
    file_date: 文件日期
    """
    try:
        predictor = AggressiveStockPredictor(symbol, file_date)
        result = predictor.run_strategy_development()
        return result
    except Exception as e:
        return {
            "error": str(e),
            "symbol": symbol,
            "file_date": file_date,
            "success": False,
        }


# # 激进测试代码
# if __name__ == "__main__":
#     print("=== 激进股票预测系统 ===")

#     # 示例调用
#     result = run_strategy_development("603232", "2024-01-15")

#     if result.get("success", False):
#         print(f"\n📈 股票代码: {result['symbol']}")
#         print(f"📅 最后交易日: {result['last_trading_date']}")
#         print(f"💰 最后收盘价: {result['last_close']}")

#         print(f"\n🔍 深度技术分析:")
#         analysis = result["technical_analysis"]
#         print(f"   当前趋势: {analysis['current_trend']}")
#         print(f"   涨跌停状态: {analysis['limit_situation']}")
#         print(f"   RSI-6状态: {analysis['rsi_6_status']}")
#         print(f"   RSI-14状态: {analysis['rsi_14_status']}")
#         print(f"   MACD信号: {analysis['macd_signal']}")
#         print(f"   均线强度: {analysis['ma_strength']}")
#         print(f"   波动率: {analysis['volatility']}")
#         print(f"   动量: {analysis['momentum']}")

#         print(f"\n🎯 激进预测结果:")
#         preds = result["predictions"]
#         print(
#             f"   ➡️  下一个交易日最低价: {preds['next_day_low']['value']} "
#             f"(置信度: {preds['next_day_low']['confidence']*100:.1f}%)"
#         )
#         print(
#             f"   ➡️  下下个交易日最高价: {preds['next_next_day_high']['value']} "
#             f"(置信度: {preds['next_next_day_high']['confidence']*100:.1f}%)"
#         )
#         print(
#             f"   ➡️  下下个交易日最低价: {preds['next_next_day_low']['value']} "
#             f"(置信度: {preds['next_next_day_low']['confidence']*100:.1f}%)"
#         )
#         print(
#             f"   ➡️  下下个交易日上涨概率: {preds['next_next_day_up_probability']['value']*100:.1f}% "
#             f"(置信度: {preds['next_next_day_up_probability']['confidence']*100:.1f}%)"
#         )

#         print(f"\n🤖 模型信息:")
#         model_info = result["model_info"]
#         print(f"   训练样本: {model_info['training_samples']}")
#         print(f"   特征数量: {model_info['feature_count']}")
#         print(f"   预测质量: {model_info['prediction_quality']}")

#     else:
#         print(f"❌ 错误: {result.get('error', '未知错误')}")
