import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.sparse import data
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings
import platform
import matplotlib.font_manager as fm
from datetime import datetime

warnings.filterwarnings("ignore")


def setup_chinese_font():
    """智能设置中文字体，确保跨平台兼容"""
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

    # 获取系统信息
    system = platform.system()

    # 列出所有可用字体
    available_fonts = set(f.name for f in fm.fontManager.ttflist)

    # 按平台设置字体优先级
    if system == "Darwin":  # macOS
        font_candidates = [
            "PingFang SC",
            "Heiti SC",
            "Hiragino Sans GB",
            "Arial Unicode MS",
            "Songti SC",
            "STHeiti",
        ]
    elif system == "Windows":  # Windows
        font_candidates = [
            "Microsoft YaHei",
            "SimHei",
            "KaiTi",
            "SimSun",
            "FangSong",
            "NSimSun",
        ]
    else:  # Linux/其他
        font_candidates = [
            "WenQuanYi Micro Hei",
            "Noto Sans CJK SC",
            "Droid Sans Fallback",
            "AR PL UMing CN",
        ]

    # 添加通用字体
    font_candidates.extend(["DejaVu Sans", "Arial", "sans-serif"])

    # 查找第一个可用的字体
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break

    # 设置字体
    if selected_font:
        plt.rcParams["font.sans-serif"] = [selected_font]
        print(f"使用字体: {selected_font}")
    else:
        # 如果找不到任何字体，使用默认字体并打印警告
        plt.rcParams["font.sans-serif"] = ["sans-serif"]
        print("警告: 未找到中文字体，可能无法正确显示中文")

    return selected_font


import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import warnings

warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import warnings

warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


class StockPredictor:
    def __init__(self, symbol):
        """
        初始化股票预测器

        参数:
        symbol: 股票代码
        """
        self.symbol = symbol
        self.model_high = None
        self.model_low = None
        self.model_close = None
        self.model_change = None
        self.model_direction = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")
        self.feature_columns = None
        self.df_processed = None
        self.lookback_period = 5  # 添加滞后特征的回顾期

    def handle_missing_values(self, df):
        """
        处理缺失值 - 优化版本
        """
        df_filled = df.copy()

        # 定义需要特殊处理的数值列
        numeric_cols = [
            "Open",
            "Close",
            "High",
            "Low",
            "Volume",
            "Amount",
            "Change",
            "ChangeAmount",
            "Amplitude",
            "TurnoverRate",
            "MA10",
            "MA50",
            "EMA12",
            "EMA26",
            "MACD",
            "MACD_Signal",
            "MACD_Hist",
            "RSI",
            "Momentum",
            "ATR",
        ]

        # 只处理实际存在的列
        numeric_cols = [col for col in numeric_cols if col in df_filled.columns]

        # 对数值列使用插值法
        for col in numeric_cols:
            df_filled[col] = df_filled[col].interpolate(
                method="linear", limit_direction="both"
            )

        # 对于其他列，使用前向和后向填充
        for col in df_filled.columns:
            if df_filled[col].isnull().sum() > 0:
                df_filled[col] = df_filled[col].fillna(method="ffill")
                df_filled[col] = df_filled[col].fillna(method="bfill")

        return df_filled

    def create_lag_features(self, df, columns, periods=5):
        """
        创建滞后特征 - 防止使用未来数据
        """
        df_lagged = df.copy()

        for col in columns:
            for period in range(1, periods + 1):
                df_lagged[f"{col}_lag{period}"] = df_lagged[col].shift(period)

        return df_lagged

    def create_technical_features(self, df):
        """
        创建技术指标特征 - 确保无未来数据泄露
        """
        df_tech = df.copy()

        # 价格动量特征 (使用滞后数据)
        df_tech["price_change_1d"] = df_tech["Close"].pct_change().shift(1)
        df_tech["price_change_3d"] = df_tech["Close"].pct_change(periods=3).shift(1)
        df_tech["price_change_5d"] = df_tech["Close"].pct_change(periods=5).shift(1)

        # 成交量变化特征
        df_tech["volume_change_1d"] = df_tech["Volume"].pct_change().shift(1)
        df_tech["volume_change_3d"] = df_tech["Volume"].pct_change(periods=3).shift(1)

        # 价格与均线偏离度 - 确保使用滞后数据
        if "MA10" in df_tech.columns:
            df_tech["price_ma10_deviation"] = (
                df_tech["Close"].shift(1) - df_tech["MA10"].shift(1)
            ) / df_tech["MA10"].shift(1)

        if "MA50" in df_tech.columns:
            df_tech["price_ma50_deviation"] = (
                df_tech["Close"].shift(1) - df_tech["MA50"].shift(1)
            ) / df_tech["MA50"].shift(1)

        # 波动率特征 - 使用滞后数据
        df_tech["daily_range"] = (df_tech["High"] - df_tech["Low"]) / df_tech[
            "Close"
        ].shift(1)

        df_tech["volatility_5d"] = (
            df_tech["Close"].pct_change().rolling(window=5).std().shift(1)
        )

        return df_tech

    def prepare_features_and_targets(self, df):
        """
        准备特征和目标变量 - 修复数据泄露问题并改进特征
        """
        df_processed = df.copy()

        # 处理日期列
        if "Date" in df_processed.columns:
            try:
                df_processed["Date"] = pd.to_datetime(
                    df_processed["Date"], format="%Y-%m-%d"
                )
            except ValueError:
                try:
                    df_processed["Date"] = pd.to_datetime(
                        df_processed["Date"], format="%Y%m%d"
                    )
                except ValueError:
                    df_processed["Date"] = pd.to_datetime(df_processed["Date"])

        # 按日期排序确保时间顺序
        df_processed = df_processed.sort_values("Date").reset_index(drop=True)

        # 处理缺失值
        df_processed = self.handle_missing_values(df_processed)

        # 创建滞后特征 - 使用前5天的数据
        price_cols = ["Open", "High", "Low", "Close", "Volume"]
        df_processed = self.create_lag_features(
            df_processed, price_cols, periods=self.lookback_period
        )

        # 创建技术指标特征
        df_processed = self.create_technical_features(df_processed)

        # 添加星期几特征（股市周一效应等）
        df_processed["day_of_week"] = df_processed["Date"].dt.dayofweek

        # 添加月份特征（季节性效应）
        df_processed["month"] = df_processed["Date"].dt.month

        # 定义目标变量 - 使用未来一天的数据
        df_processed["target_high"] = df_processed["High"].shift(-1)
        df_processed["target_low"] = df_processed["Low"].shift(-1)
        df_processed["target_close"] = df_processed["Close"].shift(-1)
        df_processed["target_change"] = df_processed["Change"].shift(-1)

        # 分类目标: 未来一天是否上涨 (1: 上涨, 0: 下跌)
        df_processed["target_direction"] = (
            df_processed["Close"].shift(-1) > df_processed["Close"]
        ).astype(int)

        # 保存特征列名 (排除目标列和日期列)
        self.feature_columns = [
            col
            for col in df_processed.columns
            if col
            not in [
                "target_high",
                "target_low",
                "target_close",
                "target_change",
                "target_direction",
                "Date",
            ]
            and not col.startswith("target_")
        ]

        return df_processed

    def prepare_training_data(self, df):
        """
        准备训练数据 - 删除包含NaN的行，但保留最后一行用于预测
        """
        # 创建副本，避免修改原始数据
        df_processed = df.copy()

        # 找出所有目标列都为NaN的行（最后一行）
        nan_mask = (
            df_processed[
                [
                    "target_high",
                    "target_low",
                    "target_close",
                    "target_change",
                    "target_direction",
                ]
            ]
            .isna()
            .all(axis=1)
        )

        # 分离出最后一行（用于预测）
        last_row = df_processed[nan_mask].copy()
        training_data = df_processed[~nan_mask].copy()

        # 删除训练数据中的NaN行
        training_data = training_data.dropna()

        # 确保最后一行被保留用于预测
        if not last_row.empty:
            # 检查最后一行是否包含必要的特征数据
            feature_mask = last_row[self.feature_columns].isna().any(axis=1)
            if not feature_mask.any():
                # 如果特征数据完整，保留最后一行
                return training_data, last_row

        return training_data, pd.DataFrame()  # 返回空DataFrame如果没有有效最后一行

    def time_series_split(self, df, test_size=0.2):
        """
        时间序列分割 - 确保测试集在训练集之后
        """
        n_samples = len(df)
        test_start_idx = int(n_samples * (1 - test_size))

        train_data = df.iloc[:test_start_idx]
        test_data = df.iloc[test_start_idx:]

        print(f"时间序列分割: 训练集={len(train_data)}, 测试集={len(test_data)}")

        return train_data, test_data

    def walk_forward_backtest(
        self, df, train_ratio=0.7, val_ratio=0.15, update_frequency=5
    ):
        """
        执行滚动窗口回测 - 修复数据泄露问题并提高效率
        """
        # 准备训练数据（删除NaN行）
        training_data, _ = self.prepare_training_data(df)

        n_samples = len(training_data)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        test_size = n_samples - train_size - val_size

        # 初始化存储预测结果的列表
        results = {
            "high": {"preds": [], "true": []},
            "low": {"preds": [], "true": []},
            "close": {"preds": [], "true": []},
            "change": {"preds": [], "true": []},
            "direction": {"preds": [], "true": []},
            "dates": [],
        }

        # 初始化模型
        models = {
            "high": None,
            "low": None,
            "close": None,
            "change": None,
            "direction": None,
        }

        # 通用参数
        common_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbose": -1,
            "max_depth": 4,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "n_jobs": -1,  # 使用所有可用的CPU核心
        }

        # 使用时间序列交叉验证
        for i in range(train_size + val_size, n_samples):
            # 训练集: 从开始到当前前一天
            train_data = training_data.iloc[:i]

            # 每update_frequency次迭代重新训练模型，否则使用现有模型
            if (i == train_size + val_size) or (
                (i - (train_size + val_size)) % update_frequency == 0
            ):
                # 划分训练和验证集 (用于早期停止)
                train_sub_size = int(len(train_data) * 0.85)
                train_sub = train_data.iloc[:train_sub_size]
                val_sub = train_data.iloc[train_sub_size:]

                # 准备训练数据
                X_train = train_sub[self.feature_columns]
                y_train_high = train_sub["target_high"]
                y_train_low = train_sub["target_low"]
                y_train_close = train_sub["target_close"]
                y_train_change = train_sub["target_change"]
                y_train_direction = train_sub["target_direction"]

                # 准备验证数据
                X_val = val_sub[self.feature_columns]
                y_val_high = val_sub["target_high"]
                y_val_low = val_sub["target_low"]
                y_val_close = val_sub["target_close"]
                y_val_change = val_sub["target_change"]
                y_val_direction = val_sub["target_direction"]

                # 标准化特征 (只在训练集上拟合)
                if i == train_size + val_size:  # 第一次训练时拟合scaler
                    X_train_scaled = self.scaler.fit_transform(X_train)
                else:
                    X_train_scaled = self.scaler.transform(X_train)

                X_val_scaled = self.scaler.transform(X_val)

                # 训练模型 (使用验证集进行早停)
                models["high"] = lgb.LGBMRegressor(**common_params)
                models["high"].fit(
                    X_train_scaled,
                    y_train_high,
                    eval_set=[(X_val_scaled, y_val_high)],
                )

                # 训练回归模型 (预测最低价)
                models["low"] = lgb.LGBMRegressor(**common_params)
                models["low"].fit(
                    X_train_scaled,
                    y_train_low,
                    eval_set=[(X_val_scaled, y_val_low)],
                )

                # 训练回归模型 (预测收盘价)
                models["close"] = lgb.LGBMRegressor(**common_params)
                models["close"].fit(
                    X_train_scaled,
                    y_train_close,
                    eval_set=[(X_val_scaled, y_val_close)],
                )

                # 训练回归模型 (预测涨跌幅)
                models["change"] = lgb.LGBMRegressor(**common_params)
                models["change"].fit(
                    X_train_scaled,
                    y_train_change,
                    eval_set=[(X_val_scaled, y_val_change)],
                )

                # 训练分类模型 (预测涨跌方向)
                models["direction"] = lgb.LGBMClassifier(**common_params)
                models["direction"].fit(
                    X_train_scaled,
                    y_train_direction,
                    eval_set=[(X_val_scaled, y_val_direction)],
                )

            # 准备测试数据 (预测第i+1天)
            test_data = training_data.iloc[i : i + 1]
            X_test = test_data[self.feature_columns]
            X_test_scaled = self.scaler.transform(X_test)

            # 使用现有模型进行预测
            pred_high = models["high"].predict(X_test_scaled)[0]
            pred_low = models["low"].predict(X_test_scaled)[0]
            pred_close = models["close"].predict(X_test_scaled)[0]
            pred_change = models["change"].predict(X_test_scaled)[0]
            pred_direction = models["direction"].predict(X_test_scaled)[0]

            # 获取真实值
            true_high = test_data["target_high"].values[0]
            true_low = test_data["target_low"].values[0]
            true_close = test_data["target_close"].values[0]
            true_change = test_data["target_change"].values[0]
            true_direction = test_data["target_direction"].values[0]

            # 存储预测结果和真实值
            results["high"]["preds"].append(pred_high)
            results["high"]["true"].append(true_high)
            results["low"]["preds"].append(pred_low)
            results["low"]["true"].append(true_low)
            results["close"]["preds"].append(pred_close)
            results["close"]["true"].append(true_close)
            results["change"]["preds"].append(pred_change)
            results["change"]["true"].append(true_change)
            results["direction"]["preds"].append(pred_direction)
            results["direction"]["true"].append(true_direction)

            # 存储日期
            if "Date" in training_data.columns:
                results["dates"].append(test_data["Date"].values[0])

            # 每10次迭代打印进度
            if (i - (train_size + val_size)) % 10 == 0:
                progress = i - (train_size + val_size) + 1
                total = test_size
                print(f"已处理 {progress} / {total} 个测试样本")

        return results

    def evaluate_model(self, results):
        """
        评估模型性能
        """
        metrics = {}

        # 回归任务评估 (最高价)
        high_preds = np.array(results["high"]["preds"])
        high_true = np.array(results["high"]["true"])
        metrics["high_mae"] = mean_absolute_error(high_true, high_preds)
        metrics["high_rmse"] = np.sqrt(mean_squared_error(high_true, high_preds))
        metrics["high_mape"] = (
            np.mean(np.abs((high_true - high_preds) / high_true)) * 100
        )
        metrics["high_r2"] = r2_score(high_true, high_preds)

        # 回归任务评估 (最低价)
        low_preds = np.array(results["low"]["preds"])
        low_true = np.array(results["low"]["true"])
        metrics["low_mae"] = mean_absolute_error(low_true, low_preds)
        metrics["low_rmse"] = np.sqrt(mean_squared_error(low_true, low_preds))
        metrics["low_mape"] = np.mean(np.abs((low_true - low_preds) / low_true)) * 100
        metrics["low_r2"] = r2_score(low_true, low_preds)

        # 回归任务评估 (收盘价)
        close_preds = np.array(results["close"]["preds"])
        close_true = np.array(results["close"]["true"])
        metrics["close_mae"] = mean_absolute_error(close_true, close_preds)
        metrics["close_rmse"] = np.sqrt(mean_squared_error(close_true, close_preds))
        metrics["close_mape"] = (
            np.mean(np.abs((close_true - close_preds) / close_true)) * 100
        )
        metrics["close_r2"] = r2_score(close_true, close_preds)

        # 回归任务评估 (涨跌幅)
        change_preds = np.array(results["change"]["preds"])
        change_true = np.array(results["change"]["true"])
        metrics["change_mae"] = mean_absolute_error(change_true, change_preds)
        metrics["change_rmse"] = np.sqrt(mean_squared_error(change_true, change_preds))
        metrics["change_r2"] = r2_score(change_true, change_preds)

        # 分类任务评估 (涨跌方向)
        direction_preds = np.array(results["direction"]["preds"])
        direction_true = np.array(results["direction"]["true"])
        metrics["direction_accuracy"] = accuracy_score(direction_true, direction_preds)
        metrics["direction_f1"] = f1_score(direction_true, direction_preds)
        metrics["direction_precision"] = precision_score(
            direction_true, direction_preds
        )
        metrics["direction_recall"] = recall_score(direction_true, direction_preds)

        # 计算方向预测准确率
        metrics["price_direction_accuracy"] = np.mean(
            (np.sign(change_preds) == np.sign(change_true)).astype(float)
        )

        return metrics

    def train_final_models(self, df):
        """
        使用全部数据训练最终模型 - 优化版本
        """
        # 准备训练数据（删除NaN行）
        training_data, _ = self.prepare_training_data(df)

        # 准备数据
        X = training_data[self.feature_columns]
        y_high = training_data["target_high"]
        y_low = training_data["target_low"]
        y_close = training_data["target_close"]
        y_change = training_data["target_change"]
        y_direction = training_data["target_direction"]

        # 使用更高效的数据分割方式
        X_train, X_val, y_high_train, y_high_val = train_test_split(
            X, y_high, test_size=0.2, shuffle=False
        )

        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # 通用参数
        common_params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbose": -1,
            "max_depth": 6,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "n_jobs": -1,
        }

        # 训练回归模型 (预测最高价) - 使用早停
        self.model_high = lgb.LGBMRegressor(**common_params)
        self.model_high.fit(
            X_train_scaled,
            y_high_train,
            eval_set=[(X_val_scaled, y_high_val)],
        )

        # 训练回归模型 (预测最低价)
        X_train, X_val, y_low_train, y_low_val = train_test_split(
            X, y_low, test_size=0.2, shuffle=False
        )
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model_low = lgb.LGBMRegressor(**common_params)
        self.model_low.fit(
            X_train_scaled,
            y_low_train,
            eval_set=[(X_val_scaled, y_low_val)],
        )

        # 训练回归模型 (预测收盘价)
        X_train, X_val, y_close_train, y_close_val = train_test_split(
            X, y_close, test_size=0.2, shuffle=False
        )
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model_close = lgb.LGBMRegressor(**common_params)
        self.model_close.fit(
            X_train_scaled,
            y_close_train,
            eval_set=[(X_val_scaled, y_close_val)],
        )

        # 训练回归模型 (预测涨跌幅)
        X_train, X_val, y_change_train, y_change_val = train_test_split(
            X, y_change, test_size=0.2, shuffle=False
        )
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model_change = lgb.LGBMRegressor(**common_params)
        self.model_change.fit(
            X_train_scaled,
            y_change_train,
            eval_set=[(X_val_scaled, y_change_val)],
        )

        # 训练分类模型 (预测涨跌方向)
        X_train, X_val, y_direction_train, y_direction_val = train_test_split(
            X, y_direction, test_size=0.2, shuffle=False
        )
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model_direction = lgb.LGBMClassifier(**common_params)
        self.model_direction.fit(
            X_train_scaled,
            y_direction_train,
            eval_set=[(X_val_scaled, y_direction_val)],
        )

        print("模型训练完成")

    def predict_day(self, data):
        """
        预测指定日期的数据
        """
        if (
            self.model_high is None
            or self.model_low is None
            or self.model_close is None
            or self.model_change is None
            or self.model_direction is None
        ):
            raise ValueError("模型尚未训练。请先调用 train_final_models 方法。")

        # 准备特征数据
        X = data[self.feature_columns].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # 进行预测
        pred_high = self.model_high.predict(X_scaled)[0]
        pred_low = self.model_low.predict(X_scaled)[0]
        pred_close = self.model_close.predict(X_scaled)[0]
        pred_change = self.model_change.predict(X_scaled)[0]

        # 预测涨跌概率
        direction_proba = self.model_direction.predict_proba(X_scaled)[0]
        up_probability = direction_proba[1]  # 上涨概率

        return {
            "predicted_high": pred_high,
            "predicted_low": pred_low,
            "predicted_close": pred_close,
            "predicted_change": pred_change,
            "up_probability": up_probability,
            "down_probability": 1 - up_probability,
        }

    def predict_for_dates(self, target_dates):
        """
        预测多个指定日期的数据

        参数:
        target_dates: 指定日期列表，每个日期格式为 '20250829'

        返回:
        预测结果列表，每个元素是一个字典
        """
        if self.df_processed is None:
            raise ValueError("请先加载和处理数据")

        # 确保输入是列表
        if not isinstance(target_dates, list):
            target_dates = [target_dates]

        # 对日期进行排序，确保按时间顺序处理
        target_dates_sorted = sorted(
            target_dates, key=lambda x: pd.to_datetime(x, format="%Y%m%d")
        )

        predictions = []

        for i, target_date in enumerate(target_dates_sorted):
            print(f"处理第 {i+1}/{len(target_dates_sorted)} 个日期: {target_date}")

            # 将目标日期转换为datetime (处理 YYYYMMDD 格式)
            target_date_dt = pd.to_datetime(target_date, format="%Y%m%d")

            # 找到目标日期在数据中的位置
            date_mask = self.df_processed["Date"] == target_date_dt
            if not date_mask.any():
                # 如果找不到确切日期，找到最接近的日期
                date_diffs = abs(self.df_processed["Date"] - target_date_dt)
                closest_idx = date_diffs.idxmin()
                closest_date = self.df_processed.loc[closest_idx, "Date"]
                print(
                    f"  未找到确切日期 {target_date}，使用最接近的日期 {closest_date.strftime('%Y%m%d')}"
                )
                target_data = self.df_processed.loc[closest_idx:closest_idx]
                actual_date = closest_date.strftime("%Y%m%d")
            else:
                target_data = self.df_processed[date_mask]
                actual_date = target_date

            # 使用目标日期之前的所有数据训练模型
            target_idx = target_data.index[0]
            train_data = self.df_processed.iloc[:target_idx]

            # 准备训练数据（删除NaN行）
            training_data, _ = self.prepare_training_data(train_data)

            # 训练模型
            self.train_final_models(training_data)

            # 进行预测
            prediction = self.predict_day(target_data)

            # 添加预测日期信息
            prediction["prediction_date"] = actual_date

            predictions.append(prediction)

        return predictions

    def plot_results(self, results, df, symbol, file_date):
        """
        绘制回测结果图表
        """
        if not results["dates"]:
            return  # 如果没有日期数据，则不绘制图表

        dates = results["dates"]
        high_true = results["high"]["true"]
        high_pred = results["high"]["preds"]
        low_true = results["low"]["true"]
        low_pred = results["low"]["preds"]
        close_true = results["close"]["true"]
        close_pred = results["close"]["preds"]

        # 创建图表
        plt.figure(figsize=(15, 12))

        # 绘制价格预测
        plt.subplot(3, 1, 1)
        plt.plot(dates, high_true, "g-", label="实际最高价", alpha=0.7, linewidth=1)
        plt.plot(dates, high_pred, "r--", label="预测最高价", alpha=0.7, linewidth=1)
        plt.plot(dates, low_true, "b-", label="实际最低价", alpha=0.7, linewidth=1)
        plt.plot(dates, low_pred, "y--", label="预测最低价", alpha=0.7, linewidth=1)
        plt.plot(dates, close_true, "k-", label="实际收盘价", alpha=0.7, linewidth=1)
        plt.plot(dates, close_pred, "m--", label="预测收盘价", alpha=0.7, linewidth=1)
        plt.fill_between(
            dates, low_pred, high_pred, color="orange", alpha=0.2, label="预测价格区间"
        )
        plt.title("价格预测 vs 实际价格")
        plt.xlabel("日期")
        plt.ylabel("价格")
        plt.legend()
        plt.xticks(rotation=45)

        # 绘制涨跌幅预测
        change_true = results["change"]["true"]
        change_pred = results["change"]["preds"]

        plt.subplot(3, 1, 2)
        plt.plot(dates, change_true, "b-", label="实际涨跌幅", alpha=0.7, linewidth=1)
        plt.plot(dates, change_pred, "r--", label="预测涨跌幅", alpha=0.7, linewidth=1)
        plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        plt.title("涨跌幅预测 vs 实际涨跌幅")
        plt.xlabel("日期")
        plt.ylabel("涨跌幅")
        plt.legend()
        plt.xticks(rotation=45)

        # 绘制分类准确率
        direction_true = results["direction"]["true"]
        direction_pred = results["direction"]["preds"]
        accuracy_rolling = [
            np.mean(
                np.array(direction_true[: i + 1]) == np.array(direction_pred[: i + 1])
            )
            for i in range(len(direction_true))
        ]

        plt.subplot(3, 1, 3)
        plt.plot(dates, accuracy_rolling, "b-", label="滚动准确率")
        plt.axhline(y=0.5, color="r", linestyle="-", label="随机猜测 (0.5)", alpha=0.5)
        plt.title("涨跌方向预测准确率随时间变化")
        plt.xlabel("日期")
        plt.ylabel("准确率")
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        data_path = f"output/{symbol}/{file_date}/backtest_results.png"
        plt.savefig(data_path, dpi=300, bbox_inches="tight")
        # plt.show()


# 主函数
def run_strategy_development_action(symbol, file_date, target_dates=None):
    """
    主函数：执行完整的策略评估流程 - 优化版本
    """
    # 1. 加载数据
    print("加载数据...")
    data_path = f"output/{symbol}/{file_date}/data.csv"
    df = pd.read_csv(data_path)

    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"数据列: {list(df.columns)}")

    # 2. 初始化预测器
    predictor = StockPredictor(symbol)

    # 3. 准备特征和目标
    print("准备特征和目标...")
    df_processed = predictor.prepare_features_and_targets(df)
    predictor.df_processed = df_processed  # 保存处理后的数据供后续使用

    # 准备训练数据（删除NaN行）
    training_data, last_row = predictor.prepare_training_data(df_processed)
    print(f"特征工程完成，剩余 {len(training_data)} 条有效训练记录")
    print(f"保留 {len(last_row)} 条记录用于未来预测")

    # 显示数据日期范围
    if "Date" in training_data.columns:
        start_date = training_data["Date"].min().strftime("%Y%m%d")
        end_date = training_data["Date"].max().strftime("%Y%m%d")
        print(f"训练数据日期范围: {start_date} 到 {end_date}")

    # 6. 使用全部数据训练最终模型
    print("使用全部数据训练最终模型...")
    predictor.train_final_models(df_processed)
    # 4. 执行滚动窗口回测
    print("执行滚动窗口回测...")
    backtest_results = predictor.walk_forward_backtest(df_processed)

    # 5. 评估模型性能
    print("评估模型性能...")
    metrics = predictor.evaluate_model(backtest_results)

    # 7. 预测最新日期
    print("预测最新日期...")
    if not last_row.empty:
        latest_pred = predictor.predict_day(last_row)
        # 添加日期信息
        if "Date" in last_row.columns:
            latest_date = last_row["Date"].iloc[0].strftime("%Y%m%d")
            latest_pred["prediction_date"] = latest_date
    else:
        # 如果没有最后一行，使用最后可用的数据
        latest_data = df_processed.iloc[-1:]
        latest_pred = predictor.predict_day(latest_data)
        if "Date" in latest_data.columns:
            latest_date = latest_data["Date"].iloc[0].strftime("%Y%m%d")
            latest_pred["prediction_date"] = latest_date

    # 8. 绘制结果图表（如果有回测结果）
    if "backtest_results" in locals():
        print("绘制结果图表...")
        predictor.plot_results(backtest_results, training_data, symbol, file_date)

    # 9. 输出结果
    result = {
        "symbol": symbol,
        "model_used": "LightGBM",
        "features": predictor.feature_columns,
        "backtest_metrics": metrics,
        "latest_prediction": latest_pred,
        "data_points": {"original": len(df), "processed": len(training_data)},
    }

    print("\n=== 策略评估结果 ===")
    print(f"股票代码: {result['symbol']}")
    print(f"模型: {result['model_used']}")
    print(
        f"数据点: 原始={result['data_points']['original']}, 处理后={result['data_points']['processed']}"
    )

    # print("\n回测指标:")
    backtest = f"回测指标:\n"
    for metric, value in result["backtest_metrics"].items():
        if "mape" in metric:
            backtest += f"  {metric}: {value:.2f}%\n"
        elif "r2" in metric:
            backtest += f"  {metric}: {value:.4f}\n"
        else:
            backtest += f"  {metric}: {value:.6f}\n"

    # print("\n最新日期预测:")
    # if "prediction_date" in result["latest_prediction"]:
    #     print(f"  预测日期: {result['latest_prediction']['prediction_date']}")
    # if result["latest_prediction"]["predicted_high"] is not None:
    #     print(f"  预测最高价: {result['latest_prediction']['predicted_high']:.4f}")
    #     print(f"  预测最低价: {result['latest_prediction']['predicted_low']:.4f}")
    #     print(f"  预测收盘价: {result['latest_prediction']['predicted_close']:.4f}")
    #     print(f"  预测涨跌幅: {result['latest_prediction']['predicted_change']:.4f}%")
    #     print(f"  上涨概率: {result['latest_prediction']['up_probability']:.4f}")
    #     print(f"  下跌概率: {result['latest_prediction']['down_probability']:.4f}")
    # else:
    #     print("  无最新预测数据")

    return (
        backtest
        + f"""
    最新日期预测:\n
    预测日期: {result['latest_prediction']['prediction_date']}
    预测最高价: {result['latest_prediction']['predicted_high']:.4f}
    预测最低价: {result['latest_prediction']['predicted_low']:.4f}
    预测收盘价: {result['latest_prediction']['predicted_close']:.4f}
    预测涨跌幅: {result['latest_prediction']['predicted_change']:.4f}%
    上涨概率: {result['latest_prediction']['up_probability']:.4f}
    下跌概率: {result['latest_prediction']['down_probability']:.4f}\n
    """
    )


# 使用示例
def run_strategy_development(symbol, file_date):
    setup_chinese_font()
    # data_path = f"output/{symbol}/{file_date}/data.csv"
    result = run_strategy_development_action(symbol, file_date)
    return result
    # # 保存结果到文件
    # import json

    # with open(f"{symbol}_strategy_result.json", "w") as f:
    #     json.dump(result, f, indent=2, ensure_ascii=False)

    # print(f"\n结果已保存到 {symbol}_strategy_result.json")
