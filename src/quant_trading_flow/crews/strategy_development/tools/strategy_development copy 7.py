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
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import platform
import matplotlib.font_manager as fm
from datetime import datetime

warnings.filterwarnings("ignore")

n_trials = 10


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
            "Amplitude",
            "Change",
            "ChangeAmount",
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
        df_processed["target_close"] = df_processed["Close"].shift(-1)
        df_processed["target_change"] = df_processed["Change"].shift(-1)
        # # 分类目标: 当天是否上涨 (1: 上涨, 0: 下跌)
        df_processed["target_direction"] = (
            df_processed["Change"].shift(-1) > 0
        ).astype(int)

        df_processed["target_open"] = df_processed["Open"].shift(-1)
        df_processed["target_volume"] = df_processed["Volume"].shift(-1)
        df_processed["target_amount"] = df_processed["Amount"].shift(-1)
        df_processed["target_amplitude"] = df_processed["Amplitude"].shift(-1)
        df_processed["target_turnoverRate"] = df_processed["TurnoverRate"].shift(-1)
        df_processed["target_changeAmount"] = df_processed["ChangeAmount"].shift(-1)
        # 删除包含NaN的行
        df_processed = df_processed.dropna()
        self.feature_columns = [
            col
            for col in df_processed.columns
            if col not in ["Date", "股票代码"] and not col.startswith("target_")
        ]
        return df_processed

    def split_data(self, df):
        """
        将数据划分为训练集(70%)、验证集(15%)和测试集(15%)
        """
        n_samples = len(df)

        # 计算划分点
        train_size = int(n_samples * 0.6)
        valid_size = int(n_samples * 0.2)

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

    def objective_change(self, trial, X_scaled, y_train_high, df, org_df, test_size):
        """Optuna目标函数，用于优化LightGBM参数"""
        # 参数建议范围
        params = {
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["gbdt", "dart"]
            ),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
            "random_state": 42,
            "verbose": -1,
            "objective": "regression",
            "metric": "rmse",
        }
        # 创建模型
        model = lgb.LGBMRegressor(**params)
        # 3. 训练模型
        model.fit(X_scaled, y_train_high)
        end_index = len(df)
        start_index = end_index - test_size
        all_high_preds = []
        all_high_true = []
        scores = []
        for i in range(start_index, end_index):
            test_data = org_df.iloc[i : i + 1]
            X_test = test_data[self.feature_columns]
            X_test_scaled = self.scaler.transform(X_test)
            pred_high = model.predict(X_test_scaled)[0]
            true_high = org_df.iloc[i + 1]["Change"]
            all_high_preds.append(pred_high)
            all_high_true.append(true_high)
        # 预测并计算RMSE
        rmse = np.sqrt(mean_squared_error(all_high_true, all_high_preds))
        scores.append(rmse)

        # 返回平均RMSE
        return np.mean(scores)

    def objective(self, trial, X_scaled, y_train_high, df, org_df, test_size, key):
        """Optuna目标函数，用于优化LightGBM参数"""
        # 参数建议范围
        params = {
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["gbdt", "dart"]
            ),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
            "random_state": 42,
            "verbose": -1,
            "objective": "regression",
            "metric": "rmse",
        }
        # 创建模型
        model = lgb.LGBMRegressor(**params)
        # 3. 训练模型
        model.fit(X_scaled, y_train_high)
        end_index = len(df)
        start_index = end_index - test_size
        all_high_preds = []
        all_high_true = []
        scores = []
        for i in range(start_index, end_index):
            test_data = org_df.iloc[i : i + 1]
            X_test = test_data[self.feature_columns]
            X_test_scaled = self.scaler.transform(X_test)
            pred_high = model.predict(X_test_scaled)[0]
            true_high = org_df.iloc[i + 1][key]
            all_high_preds.append(pred_high)
            all_high_true.append(true_high)
        # 预测并计算RMSE
        rmse = np.sqrt(mean_squared_error(all_high_true, all_high_preds))
        scores.append(rmse)

        # 返回平均RMSE
        return np.mean(scores)

    def objective_direction(self, trial, X_scaled, y_train_high, df, org_df, test_size):
        """Optuna目标函数，用于优化LightGBM参数"""
        # 参数建议范围
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
            "boosting_type": "gbdt",
            "random_state": 42,
            "objective": "binary",
            "metric": "binary_logloss",  # 使用对数损失作为优化目标
            "verbosity": -1,
        }
        # 创建模型
        model = lgb.LGBMClassifier(**params)
        # 3. 训练模型
        model.fit(X_scaled, y_train_high)
        end_index = len(df)
        start_index = end_index - test_size
        all_high_preds = []
        all_high_true = []
        scores = []
        for i in range(start_index, end_index):
            test_data = org_df.iloc[i : i + 1]
            X_test = test_data[self.feature_columns]
            X_test_scaled = self.scaler.transform(X_test)
            pred_high = model.predict(X_test_scaled)[0]
            true_high = (org_df.iloc[i + 1]["Change"] > 0).astype(int)
            all_high_preds.append(pred_high)
            all_high_true.append(true_high)
        # 预测并计算RMSE
        # rmse = np.sqrt(accuracy_score(all_high_true, all_high_preds))
        # scores.append(accuracy_score(all_high_true, all_high_preds))

        # 返回平均RMSE
        return log_loss(all_high_true, all_high_preds)

    def tran_model(self, X_scaled, y_train, df, org_df, test_data, key):
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.objective(
                trial, X_scaled, y_train, df, org_df, len(test_data), key
            ),
            n_trials=n_trials,
        )  # 尝试100组参数组
        trial = study.best_trial
        best_params = trial.params
        best_params.update(
            {
                "random_state": 42,
                "verbose": -1,
                "objective": "regression",
                "metric": "rmse",
            }
        )
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_scaled, y_train)
        return model

    # 定义目标函数
    # 3. 定义Optuna目标函数
    def walk_forward_backtest(self, df, next):
        """
        执行滚动窗口回测，使用70%训练集，15%验证集，15%测试集
        """
        # 划分数据
        train_data, valid_data, test_data = self.split_data(df)
        train_valid_data1 = pd.concat([train_data, valid_data])
        # train_valid_data = df.copy()
        train_valid_data = pd.concat([train_data, valid_data])
        org_df = self.df
        y_train_high = train_valid_data["target_high"]
        y_train_low = train_valid_data["target_low"]
        y_train_close = train_valid_data["target_close"]
        y_train_change = train_valid_data["target_change"]
        y_train_direction = train_valid_data["target_direction"]
        y_train_open = train_valid_data["target_open"]
        y_train_volume = train_valid_data["target_volume"]
        y_train_amount = train_valid_data["target_amount"]
        y_train_amplitude = train_valid_data["target_amplitude"]
        y_train_changeAmount = train_valid_data["target_changeAmount"]
        y_train_turnoverRate = train_valid_data["target_turnoverRate"]

        X_train = train_valid_data[self.feature_columns]
        X_scaled = self.scaler.fit_transform(X_train)
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
        start_index = len(train_valid_data1)
        end_index = len(df)  # 预测当天数据，所以可以到最后一个数据点
        # 创建Optuna研究并优化
        study = optuna.create_study(direction="minimize")  # 最小化RMSE
        study.optimize(
            lambda trial: self.objective(
                trial, X_scaled, y_train_high, df, org_df, len(test_data), "High"
            ),
            n_trials=n_trials,
        )  # 尝试100组参数组
        trial = study.best_trial
        best_params = trial.params
        best_params.update(
            {
                "random_state": 42,
                "verbose": -1,
                "objective": "regression",
                "metric": "rmse",
            }
        )
        self.model_high = lgb.LGBMRegressor(**best_params)
        self.model_high.fit(X_scaled, y_train_high)

        study = optuna.create_study(direction="minimize")  # 最小化RMSE
        if next:
            self.model_open = self.tran_model(
                X_scaled, y_train_open, df, org_df, test_data, "Open"
            )
            self.model_volume = self.tran_model(
                X_scaled, y_train_volume, df, org_df, test_data, "Volume"
            )
            self.model_amount = self.tran_model(
                X_scaled, y_train_amount, df, org_df, test_data, "Amount"
            )
            self.model_amplitude = self.tran_model(
                X_scaled, y_train_amplitude, df, org_df, test_data, "Amplitude"
            )
            self.model_changeAmount = self.tran_model(
                X_scaled, y_train_changeAmount, df, org_df, test_data, "ChangeAmount"
            )
            self.model_turnoverRate = self.tran_model(
                X_scaled, y_train_turnoverRate, df, org_df, test_data, "TurnoverRate"
            )

        study = optuna.create_study(direction="minimize")  # 最小化RMSE
        study.optimize(
            lambda trial: self.objective(
                trial, X_scaled, y_train_low, df, org_df, len(test_data), "Low"
            ),
            n_trials=n_trials,
        )  # 尝试100组参数组
        trial = study.best_trial
        best_params = trial.params
        best_params.update(
            {
                "random_state": 42,
                "verbose": -1,
                "objective": "regression",
                "metric": "rmse",
            }
        )
        self.model_low = lgb.LGBMRegressor(**best_params)
        self.model_low.fit(X_scaled, y_train_low)

        study = optuna.create_study(direction="minimize")  # 最小化RMSE
        study.optimize(
            lambda trial: self.objective(
                trial, X_scaled, y_train_close, df, org_df, len(test_data), "Close"
            ),
            n_trials=n_trials,
        )  # 尝试100组参数组
        trial = study.best_trial
        best_params = trial.params
        best_params.update(
            {
                "random_state": 42,
                "verbose": -1,
                "objective": "regression",
                "metric": "rmse",
            }
        )
        self.model_close = lgb.LGBMRegressor(**best_params)
        self.model_close.fit(X_scaled, y_train_close)

        # study = optuna.create_study(direction="minimize")  # 最小化RMSE
        # study.optimize(
        #     lambda trial: self.objective_change(
        #         trial, X_scaled, y_train_change, df, org_df, len(test_data)
        #     ),
        #     n_trials=n_trials,
        # )
        # trial = study.best_trial
        # best_params = trial.params
        # best_params.update(
        #     {
        #         "random_state": 42,
        #         "verbose": -1,
        #         "objective": "regression",
        #         "metric": "rmse",
        #     }
        # )
        # self.model_change = lgb.LGBMRegressor(**best_params)
        # self.model_change.fit(X_scaled, y_train_change)

        study = optuna.create_study(direction="minimize")  # 最小化RMSE
        study.optimize(
            lambda trial: self.objective_direction(
                trial, X_scaled, y_train_direction, df, org_df, len(test_data)
            ),
            n_trials=n_trials,
        )  # 尝试100组参数组
        trial = study.best_trial
        best_params = trial.params
        best_params.update(
            {
                "random_state": 42,
                "objective": "binary",
                "metric": "binary_logloss",  # 使用对数损失作为优化目标
                "verbosity": -1,
            }
        )
        self.model_direction = lgb.LGBMClassifier(**best_params)
        self.model_direction.fit(X_scaled, y_train_direction)
        for i in range(start_index, end_index):
            # 准备测试数据 (预测第i天)
            test_data = org_df.iloc[i : i + 1]
            X_test = test_data[self.feature_columns]
            X_test_scaled = self.scaler.transform(X_test)

            # 进行预测
            pred_high = self.model_high.predict(X_test_scaled)[0]
            pred_low = self.model_low.predict(X_test_scaled)[0]
            pred_close = self.model_close.predict(X_test_scaled)[0]
            # pred_change = self.model_change.predict(X_test_scaled)[0]
            pred_direction = self.model_direction.predict(X_test_scaled)[0]

            # 获取真实值
            true_high = org_df.iloc[i + 1]["High"]
            true_low = org_df.iloc[i + 1]["Low"]
            true_close = org_df.iloc[i + 1]["Close"]
            true_change = org_df.iloc[i + 1]["Change"]
            true_direction = (org_df.iloc[i + 1]["Change"] > 0).astype(int)

            # 存储预测结果和真实值
            all_high_preds.append(pred_high)
            all_high_true.append(true_high)
            all_low_preds.append(pred_low)
            all_low_true.append(true_low)
            all_close_preds.append(pred_close)
            all_close_true.append(true_close)
            # all_change_preds.append(pred_change)
            all_change_true.append(true_change)
            all_direction_preds.append(pred_direction)
            all_direction_true.append(true_direction)

            # 存储日期
            if "Date" in df.columns:
                all_dates.append(df.iloc[i]["Date"])

            # 每50次迭代打印进度
            if (i - start_index) % 50 == 0:
                print(
                    f"已处理 {i - start_index} / {end_index - start_index} 个测试样本"
                )
        # print(all_high_preds)
        # print(all_high_true)
        return {
            "high": {"preds": all_high_preds, "true": all_high_true},
            "low": {"preds": all_low_preds, "true": all_low_true},
            "close": {"preds": all_close_preds, "true": all_close_true},
            "change": {"preds": all_change_preds, "true": all_change_true},
            "direction": {"preds": all_direction_preds, "true": all_direction_true},
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
        # change_preds = np.array(results["change"]["preds"])
        # change_true = np.array(results["change"]["true"])
        # metrics["change_mae"] = mean_absolute_error(change_true, change_preds)
        # metrics["change_rmse"] = np.sqrt(mean_squared_error(change_true, change_preds))
        # metrics["change_r2"] = r2_score(change_true, change_preds)

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
        # metrics["price_direction_accuracy"] = np.mean(
        #     (np.sign(change_preds) == np.sign(change_true)).astype(float)
        # )

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
            n_estimators=1000,
            learning_rate=0.01,
            verbose=-1,
            max_depth=8,  # 减小深度，防止过拟合
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
        pred_low = self.model_low.predict(X_scaled)[0]
        pred_close = self.model_close.predict(X_scaled)[0]
        # pred_change = self.model_change.predict(X_scaled)[0]

        # pred_open = self.model_open.predict(X_scaled)[0]
        # pred_volume = self.model_volume.predict(X_scaled)[0]
        # pred_amount = self.model_amount.predict(X_scaled)[0]

        # pred_amplitude = self.model_amplitude.predict(X_scaled)[0]
        # pred_changeAmount = self.model_changeAmount.predict(X_scaled)[0]
        # pred_turnoverRate = self.model_turnoverRate.predict(X_scaled)[0]

        # 预测涨跌概率
        direction_proba = self.model_direction.predict_proba(X_scaled)[0]
        up_probability = direction_proba[1]  # 上涨概率

        return {
            "predicted_high": pred_high,
            "predicted_low": pred_low,
            "predicted_close": pred_close,
            # "predicted_change": pred_change,
            # "predicted_open": pred_open,
            # "predicted_volume": pred_volume,
            # "predicted_amount": pred_amount,
            # "predicted_amplitude": pred_amplitude,
            # "predicted_changeAmount": pred_changeAmount,
            # "predicted_turnoverRate": pred_turnoverRate,
            "up_probability": up_probability,
            "down_probability": 1 - up_probability,
        }

    def plot_results(self, results, symbol, file_date):
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
        plt.plot(dates, high_true, "g-", label="实际最高价", alpha=0.7)
        plt.plot(dates, high_pred, "r--", label="预测最高价", alpha=0.7)
        plt.plot(dates, low_true, "b-", label="实际最低价", alpha=0.7)
        plt.plot(dates, low_pred, "y--", label="预测最低价", alpha=0.7)
        plt.plot(dates, close_true, "k-", label="实际收盘价", alpha=0.7)
        plt.plot(dates, close_pred, "m--", label="预测收盘价", alpha=0.7)
        plt.fill_between(
            dates, low_pred, high_pred, color="orange", alpha=0.2, label="预测价格区间"
        )
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
        plt.savefig(
            f"output/{symbol}/{file_date}/_backtest_results.png",
            dpi=300,
            bbox_inches="tight",
        )
        # plt.show()
        plt.close()


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def vwap(daily_data):
    # 假设daily_data有'High','Low','Close','Volume'
    typical_price = (daily_data["High"] + daily_data["Low"] + daily_data["Close"]) / 3
    vwap = (typical_price * daily_data["Volume"]).cumsum() / daily_data[
        "Volume"
    ].cumsum()
    return vwap


# 计算鳄鱼线
def alligator(price, jaw_period=13, teeth_period=8, lips_period=5):
    jaw = price.rolling(jaw_period).mean().shift(8)
    teeth = price.rolling(teeth_period).mean().shift(5)
    lips = price.rolling(lips_period).mean().shift(3)
    return jaw, teeth, lips


# 计算肖夫趋势周期
def schaff_trend_cycle(price, period=10):
    ema1 = price.ewm(span=period).mean()
    ema2 = ema1.ewm(span=period).mean()
    macd = ema1 - ema2
    stc = macd.rolling(10).mean()  # 简化版本，实际STC更复杂
    return stc


def create_microstructure_features(df):
    """市场微观结构相关特征"""

    # 订单流失衡估计 (使用收盘价与VWAP的关系)
    df["vwap"] = df["Amount"] / df["Volume"]  # 计算VWAP
    df["close_vwap_ratio"] = df["Close"] / df["vwap"]
    df["order_imbalance"] = (df["Close"] - df["vwap"]) / (df["High"] - df["Low"] + 1e-8)

    # 价格离散性
    df["price_discreteness"] = ((df["Close"] * 100) % 1) / 100  # 收盘价的小数部分

    # 报价驱动特征 (估算)
    df["effective_spread"] = (
        2 * abs(df["Close"] - (df["High"] + df["Low"]) / 2) / df["Close"]
    )
    df["realized_volatility"] = df["ChangeAmount"].rolling(5).std()

    return df


from scipy.stats import entropy


def create_information_features(df):
    """信息理论和熵相关特征"""

    # 价格序列熵
    price_changes = df["Close"].pct_change().dropna()
    hist, _ = np.histogram(price_changes, bins=20, density=True)
    df["price_entropy"] = entropy(hist + 1e-8)  # 避免log(0)

    # 成交量信息熵
    volume_changes = df["Volume"].pct_change().dropna()
    vol_hist, _ = np.histogram(volume_changes, bins=20, density=True)
    df["volume_entropy"] = entropy(vol_hist + 1e-8)

    # 互信息估计 (价格与成交量)
    def estimate_mutual_info(price_changes, volume_changes, bins=10):
        joint_hist, _, _ = np.histogram2d(
            price_changes, volume_changes, bins=bins, density=True
        )
        margin1 = np.sum(joint_hist, axis=1)
        margin2 = np.sum(joint_hist, axis=0)
        mutual_info = 0
        for i in range(bins):
            for j in range(bins):
                if joint_hist[i, j] > 0:
                    mutual_info += joint_hist[i, j] * np.log(
                        joint_hist[i, j] / (margin1[i] * margin2[j] + 1e-8)
                    )
        return mutual_info

    df["price_volume_mutual_info"] = estimate_mutual_info(
        price_changes.values, volume_changes.values
    )

    return df


def create_fractal_features(df):
    """分形和市场复杂性特征"""

    # Hurst指数估计 (简化版本)
    def hurst_exponent(ts, max_lag=20):
        """估算Hurst指数"""
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]

    df["hurst_20"] = (
        df["Close"]
        .rolling(60)
        .apply(lambda x: hurst_exponent(x.values) if len(x) == 60 else np.nan)
    )

    # 分形维度 (近似)
    def fractal_dimension(ts):
        n = len(ts)
        if n < 2:
            return np.nan
        range_ts = np.max(ts) - np.min(ts)
        if range_ts == 0:
            return 1.0
        adjusted_ts = (ts - np.min(ts)) / range_ts
        l = np.sum(np.abs(np.diff(adjusted_ts)))
        return 1 + np.log(l) / np.log(n - 1)

    df["fractal_dim"] = df["Close"].rolling(30).apply(fractal_dimension)

    return df


def create_behavioral_features(df):
    """行为金融学相关特征"""

    # 处置效应指标 (基于历史高低点)
    df["52w_high"] = df["Close"].rolling(252).max()
    df["52w_low"] = df["Close"].rolling(252).min()
    df["disposition_effect"] = (df["Close"] - df["52w_low"]) / (
        df["52w_high"] - df["52w_low"] + 1e-8
    )

    # 锚定效应
    df["anchor_5d"] = df["Close"] / df["Close"].shift(5)
    df["anchor_20d"] = df["Close"] / df["Close"].shift(20)

    # 过度反应/反应不足
    df["overreaction_3d"] = (df["Close"] - df["Close"].shift(3)) / df["Close"].shift(3)
    df["overreaction_10d"] = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(
        10
    )

    # 羊群效应指标 (使用换手率异常)
    turnover_ma = df["TurnoverRate"].rolling(20).mean()
    turnover_std = df["TurnoverRate"].rolling(20).std()
    df["herding_effect"] = (df["TurnoverRate"] - turnover_ma) / (turnover_std + 1e-8)

    return df


def create_multiscale_features(df):
    """多时间尺度分析特征"""

    # 小波变换特征 (简化版本)
    def wavelet_energy(ts):
        """计算小波能量 (简化版)"""
        if len(ts) < 4:
            return np.nan
        # 简单的高通和低通滤波
        high_freq = ts - ts.rolling(2).mean()
        low_freq = ts.rolling(4).mean()
        energy_high = np.sum(high_freq**2)
        energy_low = np.sum(low_freq**2)
        return energy_high / (energy_low + 1e-8)

    df["wavelet_energy_ratio"] = df["Close"].rolling(16).apply(wavelet_energy)

    # 多尺度波动率
    for scale in [1, 3, 5, 10]:
        df[f"volatility_scale_{scale}"] = df["Close"].pct_change().rolling(scale).std()

    # 尺度相关性
    df["multiscale_correlation"] = (
        df["volatility_scale_1"].rolling(10).corr(df["volatility_scale_5"])
    )

    return df


import numpy as np
from scipy.spatial.distance import pdist, squareform


def create_topological_features(df):
    """拓扑数据分析特征 (简化版)"""

    # 持久同调特征 (近似)
    def persistence_entropy(ts, window=10):
        if len(ts) < window:
            return np.nan
        sub_ts = ts[-window:]
        # 计算距离矩阵
        dist_matrix = squareform(pdist(sub_ts.values.reshape(-1, 1)))
        # 近似持久同调特征
        birth_death_ratios = []
        for i in range(len(sub_ts) - 1):
            for j in range(i + 1, len(sub_ts)):
                if dist_matrix[i, j] > 0:
                    birth = min(sub_ts.iloc[i], sub_ts.iloc[j])
                    death = max(sub_ts.iloc[i], sub_ts.iloc[j])
                    if death > birth:
                        birth_death_ratios.append((death - birth) / death)
        if not birth_death_ratios:
            return 0
        hist, _ = np.histogram(birth_death_ratios, bins=5, density=True)
        return entropy(hist + 1e-8)

    df["topological_entropy"] = (
        df["Close"]
        .rolling(15)
        .apply(lambda x: persistence_entropy(x) if len(x) == 15 else np.nan)
    )

    return df


def create_regime_features(df):
    """市场状态和regime转移特征"""

    # 隐马尔可夫模型特征 (简化)
    def regime_indicator(returns, window=20):
        """简化的市场状态指标"""
        if len(returns) < window:
            return np.nan
        vol = returns.rolling(5).std()
        mean_return = returns.rolling(5).mean()

        # 基于波动率和收益率的简单状态分类
        high_vol = vol > vol.rolling(window).quantile(0.7)
        low_return = mean_return < mean_return.rolling(window).quantile(0.3)

        # 状态编码
        state = 0  # 正常状态
        if high_vol.iloc[-1] and low_return.iloc[-1]:
            state = 1  # 高风险低收益
        elif high_vol.iloc[-1] and not low_return.iloc[-1]:
            state = 2  # 高风险高收益
        elif not high_vol.iloc[-1] and low_return.iloc[-1]:
            state = 3  # 低风险低收益

        return state

    returns = df["Close"].pct_change()
    df["market_regime"] = returns.rolling(30).apply(
        lambda x: regime_indicator(pd.Series(x)) if len(x) == 30 else np.nan
    )

    # 状态转移概率
    df["regime_transition_prob"] = (
        df["market_regime"]
        .rolling(10)
        .apply(lambda x: len(set(x)) / len(x) if len(x) == 10 else np.nan)
    )

    return df


def formatData(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 确保Date列是日期时间类型
    if "Date" in df.columns:
        # 尝试解析日期格式
        try:
            # 先尝试YYYY-MM-DD格式
            df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        except ValueError:
            try:
                # 再尝试YYYYMMDD格式
                df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
            except ValueError:
                # 最后使用默认解析
                df["Date"] = pd.to_datetime(df["Date"])

        # 日期特征（只在Date列是日期时间类型时添加）
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["day"] = df["Date"].dt.day
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["is_month_start"] = df["Date"].dt.is_month_start
        df["is_month_end"] = df["Date"].dt.is_month_end
        df["quarter"] = df["Date"].dt.quarter
    # 价格变动特征

    df["price_range"] = df["High"] - df["Low"]  # 日内波动幅度
    df["price_change"] = df["Close"] - df["Open"]  # 日内价格变化
    df["close_open_ratio"] = df["Close"] / df["Open"]  # 收盘开盘比

    # 移动平均特征
    windows = [5, 10, 20, 60]  # 周、半月、月、季度
    for window in windows:
        df[f"MA_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"Volume_MA_{window}"] = df["Volume"].rolling(window=window).mean()
    # RSI (相对强弱指数)

    df["RSI_14"] = calculate_rsi(df["Close"])

    # MACD
    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp12 - exp26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 布林带
    df["BB_middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_middle"] + 2 * bb_std
    df["BB_lower"] = df["BB_middle"] - 2 * bb_std
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / (
        df["BB_upper"] - df["BB_lower"]
    )

    # 相对强度特征
    df["relative_strength"] = df["Close"] / df["收盘"]  # 个股vs大盘
    df["volume_ratio"] = df["Volume"] / df["成交量"]  # 成交量相对比例

    # 大盘衍生特征
    df["market_trend"] = df["收盘"].pct_change()  # 大盘收益率
    df["market_volatility"] = df["收盘"].rolling(20).std()  # 大盘波动率

    # 相关性特征
    df["corr_5d"] = df["Close"].rolling(5).corr(df["收盘"])  # 5日相关性
    # 成交量特征
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_price_trend"] = df["Volume"] * df["Close"]  # 量价趋势

    # 异常成交量
    volume_ma = df["Volume"].rolling(20).mean()
    volume_std = df["Volume"].rolling(20).std()
    df["volume_zscore"] = (df["Volume"] - volume_ma) / volume_std

    df["direction"] = (df["Change"] > 0).astype(int)
    # 确保数据按时间排序
    df = df.sort_values("Date")
    # 创建各种特征组
    df = create_microstructure_features(df)
    df = create_information_features(df)
    df = create_fractal_features(df)
    df = create_behavioral_features(df)
    df = create_multiscale_features(df)
    df = create_topological_features(df)
    df = create_regime_features(df)

    return df


# 主函数
def run_strategy_development_action(symbol, file_date):
    """
    主函数：执行完整的策略评估流程

    参数:
    symbol: 股票代码
    data_path: 数据文件路径
    target_dates: 指定预测日期列表，格式为 ['20250829', '20250830']（可选）
    """
    # 1. 加载数据
    print("加载数据...")
    df = df = pd.read_csv(f"output/{symbol}/{file_date}/data.csv")

    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"数据列: {list(df.columns)}")

    # 2. 初始化预测器
    predictor = StockPredictor(symbol)

    predictor.df = formatData(df)
    df_processed = predictor.prepare_features_and_targets(df)
    predictor.df_processed = df_processed  # 保存处理后的数据供后续使用
    print(f"特征工程完成，剩余 {len(df_processed)} 条有效记录")

    # 显示数据日期范围
    if "Date" in df_processed.columns:
        start_date = df_processed["Date"].min().strftime("%Y%m%d")
        end_date = df_processed["Date"].max().strftime("%Y%m%d")
        print(f"数据日期范围: {start_date} 到 {end_date}")

    # 4. 执行滚动窗口回测
    print("执行滚动窗口回测...")
    backtest_results = predictor.walk_forward_backtest(df_processed, False)
    # # 5. 评估模型性能
    print("评估模型性能...")
    metrics = predictor.evaluate_model(backtest_results)
    # 7. 预测最新日期
    print("预测最新日期...")
    latest_data = df.iloc[-1:]  # 获取最新数据
    print("latest_data----------")
    print(latest_data)
    latest_pred = predictor.predict_day(latest_data)
    close_value_iloc = latest_data["Close"].iloc[0]
    high_value_iloc = close_value_iloc * 1.1
    low_value_iloc = close_value_iloc * 0.9
    # 添加日期信息
    if "Date" in df.columns:
        latest_date = pd.to_datetime(df["Date"]).max().strftime("%Y%m%d")
        latest_pred["prediction_date"] = latest_date

    # 8. 绘制结果图表
    print("绘制结果图表...")
    predictor.plot_results(backtest_results, symbol, file_date)

    # 9. 输出结果
    result = {
        "symbol": symbol,
        "model_used": "LightGBM",
        "features": predictor.feature_columns,
        "backtest_metrics": metrics,
        "latest_prediction": latest_pred,
        "data_points": {"original": len(df), "processed": len(df_processed)},
    }
    module_dec = "模型评估指标:\n"
    for metric, value in result["backtest_metrics"].items():
        if "mape" in metric:
            module_dec += f"  {metric}: {value:.2f}%"
        elif "r2" in metric:
            module_dec += f"  {metric}: {value:.4f}"
        else:
            module_dec += f"  {metric}: {value:.6f}"
    # module_dec = f"""
    #   下个交易日预测结果:
    #   理论最高价: {high_value_iloc}
    #   预测最高价: {result['latest_prediction']['predicted_high']:.4f}
    #   预测最低价: {result['latest_prediction']['predicted_low']:.4f}
    #   理论最低价: {low_value_iloc}
    #   预测收盘价: {result['latest_prediction']['predicted_close']:.4f}
    #   预测开盘价: {result['latest_prediction']['predicted_open']:.4f}
    #   预测涨跌幅: {result['latest_prediction']['predicted_change']:.4f}%
    #   上涨概率: {result['latest_prediction']['up_probability']:.4f}
    #   下跌概率: {result['latest_prediction']['down_probability']:.4f}
    # """
    # high_value_iloc = close_value_iloc * 1.1 * 1.1
    # low_value_iloc = close_value_iloc * 0.9 * 0.9
    # newLastDate = pd.to_datetime(df["Date"]).max() + pd.DateOffset(days=1)
    # newLastDate = newLastDate.strftime("%Y-%m-%d")
    # newLastData = latest_data.copy()
    # newLastData["Date"] = newLastDate
    # newLastData["High"] = latest_pred["predicted_high"]
    # newLastData["Low"] = latest_pred["predicted_low"]
    # newLastData["Close"] = latest_pred["predicted_close"]
    # newLastData["Open"] = latest_pred["predicted_open"]
    # newLastData["Change"] = latest_pred["predicted_change"]
    # newLastData["Volume"] = latest_pred["predicted_volume"]
    # newLastData["Amount"] = latest_pred["predicted_amount"]
    # newLastData["Amplitude"] = latest_pred["predicted_amplitude"]
    # newLastData["ChangeAmount"] = latest_pred["predicted_changeAmount"]
    # newLastData["TurnoverRate"] = latest_pred["predicted_turnoverRate"]
    # df = pd.concat([df, newLastData], ignore_index=True)
    # predictor.df = df
    # df_processed = predictor.prepare_features_and_targets(df)
    # predictor.df_processed = df_processed  # 保存处理后的数据供后续使用
    # print(f"特征工程完成，剩余 {len(df_processed)} 条有效记录")

    # # 显示数据日期范围
    # if "Date" in df_processed.columns:
    #     start_date = df_processed["Date"].min().strftime("%Y%m%d")
    #     end_date = df_processed["Date"].max().strftime("%Y%m%d")
    #     print(f"数据日期范围: {start_date} 到 {end_date}")

    # # 4. 执行滚动窗口回测
    # # print("执行滚动窗口回测...")
    # backtest_results = predictor.walk_forward_backtest(df_processed, False)
    # latest_data = df.iloc[-1:]  # 获取最新数据
    # print("----------")
    # print(latest_data)
    # latest_pred = predictor.predict_day(latest_data)
    # result = {
    #     "symbol": symbol,
    #     "model_used": "LightGBM",
    #     "features": predictor.feature_columns,
    #     "backtest_metrics": metrics,
    #     "latest_prediction": latest_pred,
    #     "data_points": {"original": len(df), "processed": len(df_processed)},
    # }
    return (
        module_dec
        + f"""  
      下个交易日预测结果:
      理论最高价: {high_value_iloc}
      预测最高价: {result['latest_prediction']['predicted_high']:.4f}
      预测最低价: {result['latest_prediction']['predicted_low']:.4f}
      理论最低价: {low_value_iloc}
      预测收盘价: {result['latest_prediction']['predicted_close']:.4f}
      上涨概率: {result['latest_prediction']['up_probability']:.4f}
      下跌概率: {result['latest_prediction']['down_probability']:.4f}
    """
    )


# 使用示例
def run_strategy_development(symbol, file_date):
    setup_chinese_font()
    # # 执行策略评估
    result = run_strategy_development_action(symbol, file_date)
    return result
