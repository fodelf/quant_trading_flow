import numpy as np
import pandas as pd
import lightgbm as lgb
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
        self.model_direction = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")
        self.feature_columns = None
        self.df_processed = None
        self.df = None

    def handle_missing_values(self, df):
        """
        处理缺失值
        """
        # 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"发现缺失值: {missing_values[missing_values > 0].to_dict()}")

        # 使用前向填充和后向填充结合的方法处理缺失值
        df_filled = df.copy()

        # 对于价格和交易量等连续变量，使用线性插值
        numeric_cols = [
            "Open",
            "Close",
            "High",
            "Low",
            "Volume",
            "Amount",
            "Change",
            "ChangeAmount",
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
            "Amplitude",
            "TurnoverRate",
        ]

        for col in numeric_cols:
            if col in df_filled.columns:
                # 先使用前向填充
                df_filled[col] = df_filled[col].fillna(method="ffill")
                # 然后使用后向填充处理开头可能存在的缺失值
                df_filled[col] = df_filled[col].fillna(method="bfill")
                # 如果还有缺失值，使用均值填充
                if df_filled[col].isnull().sum() > 0:
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())

        # 对于其他列，使用简单填充
        for col in df_filled.columns:
            if df_filled[col].isnull().sum() > 0:
                df_filled[col] = df_filled[col].fillna(method="ffill")
                df_filled[col] = df_filled[col].fillna(method="bfill")
                if df_filled[col].dtype == "object":
                    df_filled[col] = df_filled[col].fillna("unknown")
                else:
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())

        return df_filled

    def prepare_features_and_targets(self, df):
        """
        准备特征和目标变量
        """
        df_processed = df.copy()

        # 确保日期列正确解析 (处理两种格式: YYYY-MM-DD 和 YYYYMMDD)
        if "Date" in df_processed.columns:
            # 尝试解析为 YYYY-MM-DD 格式
            try:
                df_processed["Date"] = pd.to_datetime(
                    df_processed["Date"], format="%Y-%m-%d"
                )
            except ValueError:
                # 如果失败，尝试解析为 YYYYMMDD 格式
                try:
                    df_processed["Date"] = pd.to_datetime(
                        df_processed["Date"], format="%Y%m%d"
                    )
                except ValueError:
                    # 如果两种格式都失败，使用默认解析
                    df_processed["Date"] = pd.to_datetime(df_processed["Date"])

        # 处理缺失值
        df_processed = self.handle_missing_values(df_processed)

        # 定义目标变量 - 当天数据
        df_processed["target_high"] = df_processed["High"].shift(-1)
        df_processed["target_low"] = df_processed["Low"].shift(-1)
        # df_processed["target_close"] = df_processed["Close"]
        # df_processed["target_change"] = df_processed["Change"]

        # # 分类目标: 当天是否上涨 (1: 上涨, 0: 下跌)
        # df_processed["target_direction"] = (
        #     df_processed["Close"] > df_processed["Open"]
        # ).astype(int)

        # 删除包含NaN的行
        df_processed = df_processed.dropna()

        # 保存特征列名 (排除目标列和日期列)
        # self.feature_columns = [
        #     col
        #     for col in df_processed.columns
        #     if col
        #     not in [
        #         "target_high",
        #         "target_low",
        #         "target_close",
        #         "target_change",
        #         "target_direction",
        #         "Date",
        #     ]
        #     and not col.startswith("target_")
        # ]

        self.feature_columns = [
            "Open",
            "Close",
            "High",
            "Low",
            "Volume",
            "Amount",
            "Amplitude",
            "Change",
            "ChangeAmount",
            "TurnoverRate",
            "MA5",
            "MA10",
            "MA20",
            "Close_LAG_1",
            "Close_LAG_2",
            "Close_LAG_3",
            "Close_LAG_5",
            "Close_LAG_10",
            "Volume_LAG_1",
            "Volume_LAG_2",
            "Volume_LAG_3",
            "Volume_LAG_5",
            "Volume_LAG_10",
            "RSI",
            "Returns",
            "Volatility",
            "DayOfWeek",
            "Month",
            "Quarter",
            "EMA12",
            "EMA26",
            "MACD",
            "ATR",
        ]
        return df_processed

    def split_data(self, df):
        """
        将数据划分为训练集(70%)、验证集(15%)和测试集(15%)
        """
        n_samples = len(df)

        # 计算划分点
        train_size = int(n_samples * 0.7)
        valid_size = int(n_samples * 0.15)

        # 确保有足够的数据进行划分
        if n_samples < (train_size + valid_size + 1):
            raise ValueError(
                f"数据量不足，至少需要{int(n_samples * 0.85) + 1}条数据进行划分"
            )

        # 划分数据集
        train_data = df.iloc[:train_size]
        valid_data = df.iloc[train_size : train_size + valid_size]
        test_data = df.iloc[train_size + valid_size :]

        print(
            f"数据划分: 训练集={len(train_data)}, 验证集={len(valid_data)}, 测试集={len(test_data)}"
        )

        return train_data, valid_data, test_data

    def walk_forward_backtest(self, df):
        """
        执行滚动窗口回测，使用70%训练集，15%验证集，15%测试集
        """
        # 划分数据
        train_data, valid_data, test_data = self.split_data(df)

        # 合并训练集和验证集用于模型训练
        train_valid_data = pd.concat([train_data, valid_data])

        # 初始化存储预测结果的列表
        all_high_preds = []
        all_high_true = []
        all_low_preds = []
        all_low_true = []
        all_close_preds = []
        all_close_true = []
        all_change_preds = []
        all_change_true = []
        all_direction_preds = []
        all_direction_true = []
        all_dates = []

        # 计算测试集的起始索引
        start_index = len(train_valid_data)
        end_index = len(df)  # 预测当天数据，所以可以到最后一个数据点

        y_train_high = df["target_high"]
        X_train = df[self.feature_columns]
        print(f"开始滚动回测，从索引 {start_index} 到 {end_index}")
        self.model_high = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            verbose=-1,
            max_depth=3,  # 减小深度，防止过拟合
            min_child_samples=20,  # 增加最小叶子样本数
            subsample=0.8,  # 行采样
            colsample_bytree=0.8,  # 列采样
            reg_alpha=0.1,  # L1正则化
            reg_lambda=0.1,  # L2正则化
            random_state=42,
            objective="regression",
            metric="rmse",
        )
        X_scaled = self.scaler.fit_transform(X_train)
        self.model_high.fit(X_scaled, y_train_high)
        org_df = self.df
        for i in range(start_index, end_index):
            # 训练集: 从开始到当前前一天
            # train_data = df.iloc[:i]

            # 准备训练数据
            # X_train = train_data[self.feature_columns]
            # y_train_high = train_data["target_high"]
            # y_train_low = train_data["target_low"]
            # y_train_close = train_data["target_close"]
            # y_train_change = train_data["target_change"]
            # y_train_direction = train_data["target_direction"]

            # 标准化特征
            # if i == start_index:  # 第一次训练时拟合scaler
            #     X_train_scaled = self.scaler.fit_transform(X_train)
            # else:
            #     X_train_scaled = self.scaler.transform(X_train)

            # 训练回归模型 (预测最高价)
            # model_high = lgb.LGBMRegressor(
            #     n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1
            # )

            # # 训练回归模型 (预测最低价)
            # model_low = lgb.LGBMRegressor(
            #     n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1
            # )
            # model_low.fit(X_train_scaled, y_train_low)

            # # 训练回归模型 (预测收盘价)
            # model_close = lgb.LGBMRegressor(
            #     n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1
            # )
            # model_close.fit(X_train_scaled, y_train_close)

            # # 训练回归模型 (预测涨跌幅)
            # model_change = lgb.LGBMRegressor(
            #     n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1
            # )
            # model_change.fit(X_train_scaled, y_train_change)

            # # 训练分类模型 (预测涨跌方向)
            # model_direction = lgb.LGBMClassifier(
            #     n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1
            # )
            # model_direction.fit(X_train_scaled, y_train_direction)

            # 准备测试数据 (预测第i天)
            test_data = df.iloc[i : i + 1]
            X_test = test_data[self.feature_columns]
            X_test_scaled = self.scaler.transform(X_test)

            # 进行预测
            pred_high = self.model_high.predict(X_test_scaled)[0]
            # pred_low = model_low.predict(X_test_scaled)[0]
            # pred_close = model_close.predict(X_test_scaled)[0]
            # pred_change = model_change.predict(X_test_scaled)[0]
            # pred_direction = model_direction.predict(X_test_scaled)[0]

            # 获取真实值
            true_high = org_df.iloc[i + 1]["High"]
            # true_low = df.iloc[i]["target_low"]
            # true_close = df.iloc[i]["target_close"]
            # true_change = df.iloc[i]["target_change"]
            # true_direction = df.iloc[i]["target_direction"]

            # 存储预测结果和真实值
            all_high_preds.append(pred_high)
            all_high_true.append(true_high)
            # all_low_preds.append(pred_low)
            # all_low_true.append(true_low)
            # all_close_preds.append(pred_close)
            # all_close_true.append(true_close)
            # all_change_preds.append(pred_change)
            # all_change_true.append(true_change)
            # all_direction_preds.append(pred_direction)
            # all_direction_true.append(true_direction)

            # 存储日期
            if "Date" in df.columns:
                all_dates.append(df.iloc[i]["Date"])

            # 每50次迭代打印进度
            if (i - start_index) % 50 == 0:
                print(
                    f"已处理 {i - start_index} / {end_index - start_index} 个测试样本"
                )

        return {
            "high": {"preds": all_high_preds, "true": all_high_true},
            # "low": {"preds": all_low_preds, "true": all_low_true},
            # "close": {"preds": all_close_preds, "true": all_close_true},
            # "change": {"preds": all_change_preds, "true": all_change_true},
            # "direction": {"preds": all_direction_preds, "true": all_direction_true},
            "dates": all_dates,
        }

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
        使用全部数据训练最终模型
        """
        # 准备数据
        X = df[self.feature_columns]
        y_high = df["target_high"]
        # y_low = df["target_low"]
        # y_close = df["target_close"]
        # y_change = df["target_change"]
        # y_direction = df["target_direction"]

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 训练回归模型 (预测最高价)
        self.model_high = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            verbose=-1,
            max_depth=3,  # 减小深度，防止过拟合
            min_child_samples=20,  # 增加最小叶子样本数
            subsample=0.8,  # 行采样
            colsample_bytree=0.8,  # 列采样
            reg_alpha=0.1,  # L1正则化
            reg_lambda=0.1,  # L2正则化
            random_state=42,
        )
        self.model_high.fit(X_scaled, y_high)

        # 训练回归模型 (预测最低价)
        # self.model_low = lgb.LGBMRegressor(
        #     n_estimators=150,
        #     learning_rate=0.05,
        #     random_state=42,
        #     verbose=-1,
        #     max_depth=3,  # 减小深度，防止过拟合
        #     min_child_samples=20,  # 增加最小叶子样本数
        #     subsample=0.8,  # 行采样
        #     colsample_bytree=0.8,  # 列采样
        #     reg_alpha=0.1,  # L1正则化
        #     reg_lambda=0.1,  # L2正则化
        # )
        # self.model_low.fit(X_scaled, y_low)

        # # 训练回归模型 (预测收盘价)
        # self.model_close = lgb.LGBMRegressor(
        #     n_estimators=150,
        #     learning_rate=0.05,
        #     random_state=42,
        #     verbose=-1,
        #     max_depth=3,  # 减小深度，防止过拟合
        #     min_child_samples=20,  # 增加最小叶子样本数
        #     subsample=0.8,  # 行采样
        #     colsample_bytree=0.8,  # 列采样
        #     reg_alpha=0.1,  # L1正则化
        #     reg_lambda=0.1,  # L2正则化
        # )
        # self.model_close.fit(X_scaled, y_close)

        # # 训练回归模型 (预测涨跌幅)
        # self.model_change = lgb.LGBMRegressor(
        #     n_estimators=150,
        #     learning_rate=0.05,
        #     random_state=42,
        #     verbose=-1,
        #     max_depth=3,  # 减小深度，防止过拟合
        #     min_child_samples=20,  # 增加最小叶子样本数
        #     subsample=0.8,  # 行采样
        #     colsample_bytree=0.8,  # 列采样
        #     reg_alpha=0.1,  # L1正则化
        #     reg_lambda=0.1,  # L2正则化
        # )
        # self.model_change.fit(X_scaled, y_change)

        # # 训练分类模型 (预测涨跌方向)
        # self.model_direction = lgb.LGBMClassifier(
        #     n_estimators=150,
        #     learning_rate=0.05,
        #     random_state=42,
        #     verbose=-1,
        #     max_depth=3,  # 减小深度，防止过拟合
        #     min_child_samples=20,  # 增加最小叶子样本数
        #     subsample=0.8,  # 行采样
        #     colsample_bytree=0.8,  # 列采样
        #     reg_alpha=0.1,  # L1正则化
        #     reg_lambda=0.1,  # L2正则化
        # )
        # self.model_direction.fit(X_scaled, y_direction)

    def predict_day(self, data):
        """
        预测指定日期的数据
        """
        if (
            self.model_high
            is None
            # or self.model_low is None
            # or self.model_close is None
            # or self.model_change is None
            # or self.model_direction is None
        ):
            raise ValueError("模型尚未训练。请先调用 train_final_models 方法。")

        # 准备特征数据
        X = data[self.feature_columns].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # 进行预测
        pred_high = self.model_high.predict(X_scaled)[0]
        # pred_low = self.model_low.predict(X_scaled)[0]
        # pred_close = self.model_close.predict(X_scaled)[0]
        # pred_change = self.model_change.predict(X_scaled)[0]

        # 预测涨跌概率
        # direction_proba = self.model_direction.predict_proba(X_scaled)[0]
        # up_probability = direction_proba[1]  # 上涨概率

        return {
            "predicted_high": pred_high,
            # "predicted_low": pred_low,
            # "predicted_close": pred_close,
            # "predicted_change": pred_change,
            # "up_probability": up_probability,
            # "down_probability": 1 - up_probability,
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

            # 训练模型
            self.train_final_models(train_data)

            # 进行预测
            prediction = self.predict_day(target_data)

            # 添加预测日期信息
            prediction["prediction_date"] = actual_date

            predictions.append(prediction)

        return predictions

    def plot_results(self, results, df):
        """
        绘制回测结果图表
        """
        if not results["dates"]:
            return  # 如果没有日期数据，则不绘制图表

        dates = results["dates"]
        high_true = results["high"]["true"]
        high_pred = results["high"]["preds"]
        # low_true = results["low"]["true"]
        # low_pred = results["low"]["preds"]
        # close_true = results["close"]["true"]
        # close_pred = results["close"]["preds"]

        # 创建图表
        plt.figure(figsize=(15, 12))

        # 绘制价格预测
        plt.subplot(3, 1, 1)
        plt.plot(dates, high_true, "g-", label="实际最高价", alpha=0.7)
        plt.plot(dates, high_pred, "r--", label="预测最高价", alpha=0.7)
        # plt.plot(dates, low_true, "b-", label="实际最低价", alpha=0.7)
        # plt.plot(dates, low_pred, "y--", label="预测最低价", alpha=0.7)
        # plt.plot(dates, close_true, "k-", label="实际收盘价", alpha=0.7)
        # plt.plot(dates, close_pred, "m--", label="预测收盘价", alpha=0.7)
        # plt.fill_between(
        #     dates, low_pred, high_pred, color="orange", alpha=0.2, label="预测价格区间"
        # )
        plt.title("价格预测 vs 实际价格")
        plt.xlabel("日期")
        plt.ylabel("价格")
        plt.legend()
        plt.xticks(rotation=45)

        # 绘制涨跌幅预测
        # change_true = results["change"]["true"]
        # change_pred = results["change"]["preds"]

        # plt.subplot(3, 1, 2)
        # plt.plot(dates, change_true, "b-", label="实际涨跌幅", alpha=0.7)
        # plt.plot(dates, change_pred, "r--", label="预测涨跌幅", alpha=0.7)
        # plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        # plt.title("涨跌幅预测 vs 实际涨跌幅")
        # plt.xlabel("日期")
        # plt.ylabel("涨跌幅")
        # plt.legend()
        # plt.xticks(rotation=45)

        # 绘制分类准确率
        # direction_true = results["direction"]["true"]
        # direction_pred = results["direction"]["preds"]
        # accuracy_rolling = [
        #     np.mean(
        #         np.array(direction_true[: i + 1]) == np.array(direction_pred[: i + 1])
        #     )
        #     for i in range(len(direction_true))
        # ]

        # plt.subplot(3, 1, 3)
        # plt.plot(dates, accuracy_rolling, "b-", label="滚动准确率")
        # plt.axhline(y=0.5, color="r", linestyle="-", label="随机猜测 (0.5)", alpha=0.5)
        # plt.title("涨跌方向预测准确率随时间变化")
        # plt.xlabel("日期")
        # plt.ylabel("准确率")
        # plt.legend()
        # plt.xticks(rotation=45)

        # plt.tight_layout()
        plt.savefig(f"{self.symbol}_backtest_results.png", dpi=300, bbox_inches="tight")
        plt.show()


# 主函数
def run_strategy_development_action(symbol, data_path):
    """
    主函数：执行完整的策略评估流程

    参数:
    symbol: 股票代码
    data_path: 数据文件路径
    target_dates: 指定预测日期列表，格式为 ['20250829', '20250830']（可选）
    """
    # 1. 加载数据
    print("加载数据...")
    df = pd.read_csv(data_path)

    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"数据列: {list(df.columns)}")

    # 2. 初始化预测器
    predictor = StockPredictor(symbol)

    # 3. 准备特征和目标
    # print("准备特征和目标...")
    # if "Date" in df.columns:
    #     df["Date"] = pd.to_datetime(df["Date"])
    #     df.set_index("Date", inplace=True)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(window=14).mean().shift(1)
    df["MA5"] = df["Close"].rolling(5).mean().shift(1)
    df["MA10"] = df["Close"].rolling(10).mean().shift(1)
    df["MA20"] = df["Close"].rolling(20).mean().shift(1)
    # 指数移动平均线
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean().shift(1)
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean().shift(1)
    df["MACD"] = df["EMA12"] - df["EMA26"]
    lags = [1, 2, 3, 5, 10]
    for lag in lags:
        df[f"Close_LAG_{lag}"] = df["Close"].shift(lag)
        df[f"Volume_LAG_{lag}"] = df["Volume"].shift(lag)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs)).shift(1)
    # 3. 波动率
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std().shift(1)
    # 4. 日期特征
    df["Date1"] = pd.to_datetime(df["Date"])
    df["DayOfWeek"] = df["Date1"].dt.dayofweek
    df["Month"] = df["Date1"].dt.month
    df["Quarter"] = df["Date1"].dt.quarter
    new_df = df.iloc[:-1]
    predictor.df = df
    df_processed = predictor.prepare_features_and_targets(new_df)
    predictor.df_processed = df_processed  # 保存处理后的数据供后续使用
    print(f"特征工程完成，剩余 {len(df_processed)} 条有效记录")

    # 显示数据日期范围
    if "Date" in df_processed.columns:
        start_date = df_processed["Date"].min().strftime("%Y%m%d")
        end_date = df_processed["Date"].max().strftime("%Y%m%d")
        print(f"数据日期范围: {start_date} 到 {end_date}")

    # 4. 执行滚动窗口回测
    # print("执行滚动窗口回测...")
    backtest_results = predictor.walk_forward_backtest(df_processed)

    # # 5. 评估模型性能
    # print("评估模型性能...")
    # metrics = predictor.evaluate_model(backtest_results)

    # 6. 使用全部数据训练最终模型
    # print("使用全部数据训练最终模型...")
    # predictor.train_final_models(df_processed)

    # 7. 预测最新日期
    print("预测最新日期...")
    # latest_data = df_processed.iloc[-2:]  # 获取最新数据
    # print(latest_data)
    latest_data = df.iloc[-1:]  # 获取最新数据
    latest_data["target_high"] = 0
    print(latest_data)
    latest_pred = predictor.predict_day(latest_data)
    # 添加日期信息
    if "Date" in df_processed.columns:
        latest_date = df_processed["Date"].max().strftime("%Y%m%d")
        latest_pred["prediction_date"] = latest_date

    # 8. 绘制结果图表
    print("绘制结果图表...")
    predictor.plot_results(backtest_results, df_processed)

    # 9. 输出结果
    result = {
        "symbol": symbol,
        "model_used": "LightGBM",
        "features": predictor.feature_columns,
        # "backtest_metrics": metrics,
        "latest_prediction": latest_pred,
        "data_points": {"original": len(df), "processed": len(df_processed)},
    }

    # print("\n=== 策略评估结果 ===")
    # print(f"股票代码: {result['symbol']}")
    # print(f"模型: {result['model_used']}")
    # print(
    #     f"数据点: 原始={result['data_points']['original']}, 处理后={result['data_points']['processed']}"
    # )

    # print("\n回测指标:")
    # for metric, value in result["backtest_metrics"].items():
    #     if "mape" in metric:
    #         print(f"  {metric}: {value:.2f}%")
    #     elif "r2" in metric:
    #         print(f"  {metric}: {value:.4f}")
    #     else:
    #         print(f"  {metric}: {value:.6f}")

    print("\n最新日期预测:")
    if "prediction_date" in result["latest_prediction"]:
        print(f"  预测日期: {result['latest_prediction']['prediction_date']}")
    print(f"  预测最高价: {result['latest_prediction']['predicted_high']:.4f}")
    print(f"  预测最低价: {result['latest_prediction']['predicted_low']:.4f}")
    # print(f"  预测收盘价: {result['latest_prediction']['predicted_close']:.4f}")
    # print(f"  预测涨跌幅: {result['latest_prediction']['predicted_change']:.4f}%")
    # print(f"  上涨概率: {result['latest_prediction']['up_probability']:.4f}")
    # print(f"  下跌概率: {result['latest_prediction']['down_probability']:.4f}")

    return result


# 使用示例
def run_strategy_development(symbol, data_path):
    setup_chinese_font()
    # # 执行策略评估
    result = run_strategy_development_action(symbol, data_path)

    # # 保存结果到文件
    # import json

    # with open(f"{symbol}_strategy_result.json", "w") as f:
    #     json.dump(result, f, indent=2, ensure_ascii=False)

    # print(f"\n结果已保存到 {symbol}_strategy_result.json")
