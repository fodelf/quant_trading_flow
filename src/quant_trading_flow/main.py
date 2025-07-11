#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, and_, router

from datetime import datetime

from rich import console

import re
import json
import os
from typing import Union, List, Dict, Optional

import pandas as pd
import os
import time
from quant_trading_flow.crews.data_engineer.data_engineer import DataEngineerCrew
from quant_trading_flow.crews.fundamental_analysis.fundamental_analysis import (
    FundamentalAnalysisCrew,
)
from quant_trading_flow.crews.government_affairs.government_affairs import (
    GovernmentAffairsCrew,
)
from quant_trading_flow.crews.public_sentiment.public_sentiment import (
    PublicSentimentCrew,
)

from quant_trading_flow.crews.strategy_development.strategy_development_crew import (
    StrategyDevelopmentCrew,
)
from quant_trading_flow.crews.risk_management.risk_management import RiskManagementCrew
from quant_trading_flow.crews.cfo.cfo import CfoCrew
from quant_trading_flow.crews.stock_screener.stock_screener import StockScreenerCrew
import re
import json
from typing import Union, List, Dict, Optional
import csv
from quant_trading_flow.crews.stock_screener.tools.stock_screener_tool import (
    get_filtered_stocks,
)


def append_number_to_csv(file_path, number):
    """
    向CSV文件追加数字：如果文件不存在则创建并写入，如果存在则追加

    参数:
        file_path (str): CSV文件路径
        number (int/float): 要追加的数字
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        # 文件不存在，创建新文件并写入数字
        df = pd.DataFrame([number], columns=["Value"])
        df.to_csv(file_path, index=False)
        print(f"✅ 创建新文件 '{file_path}' 并写入初始值: {number}")
    else:
        try:
            # 文件存在，读取现有数据
            df = pd.read_csv(file_path)

            # 创建新行并追加
            new_row = pd.DataFrame([number], columns=["Value"])
            df = pd.concat([df, new_row], ignore_index=True)

            # 保存回文件
            df.to_csv(file_path, index=False)
            print(f"📝 在 '{file_path}' 中追加值: {number}")

        except Exception as e:
            print(f"❌ 追加数据失败: {e}")


class TradingState(BaseModel):
    symbol: str = "SHA:600513"
    start_date: str = "20180101"
    end_date: str = datetime.now().strftime("%Y%m%d")
    file_date: str = datetime.now().strftime("%Y%m%d%H%M%S")
    risk_approval_flag: bool = True
    trade_approval_flag: bool = True


class TradingFlow(Flow[TradingState]):

    @start()
    def init_market_data(self):
        # 获取目标市场数据
        return {
            # "symbol": "SHE:002036",
            # "symbol": "SHA:688551",
            "symbol": self.state.symbol,
            "start_date": self.state.start_date,
            "end_date": self.state.end_date,
            "file_date": self.state.file_date,
        }

        # @listen(init_market_data)
        # def handle_data(self, market_data):
        #     result = DataEngineerCrew().crew().kickoff(inputs=market_data)
        #     return result.raw

        # @listen(init_market_data)
        # def fundamental_analysis(self, market_data):
        #     result = FundamentalAnalysisCrew().crew().kickoff(inputs=market_data)
        #     return result.raw

        # @listen(init_market_data)
        # def government_affairs(self, market_data):
        #     result = GovernmentAffairsCrew().crew().kickoff(inputs=market_data)
        #     return result.raw

        # @listen(init_market_data)
        # def public_sentiment(self, market_data):
        #     result = PublicSentimentCrew().crew().kickoff(inputs=market_data)
        #     return result.raw

        # @listen(handle_data)
        # def strategy_development(self):
        #     print(self.state.file_date)
        #     result = (
        #         StrategyDevelopmentCrew()
        #         .crew()
        #         .kickoff(
        #             inputs={
        #                 "symbol": self.state.symbol,
        #                 "start_date": self.state.start_date,
        #                 "end_date": self.state.end_date,
        #                 "file_date": self.state.file_date,
        #             }
        #         )
        #     )
        #     return result.raw

        # @listen(
        #     and_(
        #         strategy_development,
        #         public_sentiment,
        #         fundamental_analysis,
        #         government_affairs,
        #     )
        # )
        # def risk_management(self):
        #     result = (
        #         RiskManagementCrew()
        #         .crew()
        #         .kickoff(
        #             inputs={
        #                 "symbol": self.state.symbol,
        #                 "start_date": self.state.start_date,
        #                 "end_date": self.state.end_date,
        #                 "file_date": self.state.file_date,
        #             }
        #         )
        #     )
        #     self.state.risk_approval_flag = "同意买入" in result.raw
        #     return result.raw

        # @router(risk_management)
        # def risk_approval(self):
        #     if self.state.risk_approval_flag:
        #         return "risk_approval_pass"
        #     else:
        #         return "approval_reject"

        # @router("risk_approval_pass")
        # def pass_risk_approval(self):
        print("分析风险，通过")
        return "pass_risk_approval"

    # @listen(pass_risk_approval)
    @listen(init_market_data)
    def trade_management(self):
        result = (
            CfoCrew()
            .crew()
            .kickoff(
                inputs={
                    "symbol": self.state.symbol,
                    "start_date": self.state.start_date,
                    "end_date": self.state.end_date,
                    "file_date": self.state.file_date,
                }
            )
        )
        self.state.trade_approval_flag = "同意买入" in result.raw
        return result.raw

    @router(trade_management)
    def trade_approval(self):
        if self.state.trade_approval_flag:
            return "trade_approval_pass"
        else:
            return "approval_reject"

    # @listen("trade_approval_pass")
    # def trade(self):
    #     print("允许交易")
    #     csv_file = "trade.csv"
    #     append_number_to_csv(csv_file, self.state.symbol)
    #     return "trade"

    # @listen("approval_reject")
    # def reject_approval(self):
    #     print("交易不通过")
    #     return "reject_approval"


def create_object(num):
    return {
        "symbol": num,
        "start_date": "20180101",
        "end_date": datetime.now().strftime("%Y%m%d"),
        "file_date": "20250711220526",
        # "file_date": datetime.now().strftime("%Y%m%d%H%M%S"),
    }


def clean_json_string(json_str: str) -> str:
    """
    清理包含注释的 JSON 字符串

    参数:
        json_str: 包含注释的 JSON 字符串

    返回:
        清理后的标准 JSON 字符串
    """
    # 1. 移除单行注释 (// 开头的注释)
    json_str = re.sub(r"//.*", "", json_str)

    # 2. 移除多行注释 (/* ... */)
    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)

    # 3. 处理中文括号（将全角括号替换为半角）
    json_str = json_str.replace("（", "(").replace("）", ")")

    # 4. 替换单引号为双引号
    json_str = json_str.replace("'", '"')

    # 5. 移除尾随逗号（可能导致 JSON 解析错误）
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    # 6. 移除空白行
    json_str = "\n".join(line for line in json_str.splitlines() if line.strip())

    return json_str.strip()


def extract_json_from_markdown(markdown_text: str) -> Optional[Union[Dict, List]]:
    """
    从Markdown文本中提取并解析JSON内容（支持带注释的JSON）

    参数:
        markdown_text: 包含JSON代码块的Markdown字符串

    返回:
        解析后的JSON对象 (dict或list)，若未找到有效JSON则返回None
    """
    # 匹配JSON代码块的正则表达式
    pattern = r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```"

    # 使用DOTALL标志匹配多行内容
    match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE)

    if not match:
        # 尝试匹配无语言标签的代码块
        pattern_unlabeled = r"```\s*(\{.*?\}|\[.*?\])\s*```"
        match = re.search(pattern_unlabeled, markdown_text, re.DOTALL)
        if not match:
            return None

    json_str = match.group(1).strip()

    try:
        # 清理非标准JSON内容
        cleaned_json = clean_json_string(json_str)

        # 解析JSON
        return json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"问题内容: {json_str[:200]}...")  # 打印前200个字符
        return None


def read_csv_values(csv_path: str, column_name: str = "Value") -> List[str]:
    """
    从CSV文件中读取指定列的值

    参数:
        csv_path: CSV文件路径
        column_name: 要读取的列名 (默认为"Value")

    返回:
        去重后的值列表
    """
    values = set()

    if not os.path.exists(csv_path):
        print(f"错误: 文件 '{csv_path}' 不存在")
        return list(values)

    try:
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if column_name in row:
                    value = row[column_name].strip()
                    if value:  # 确保值不为空
                        values.add(value)
        return list(values)
    except Exception as e:
        print(f"读取CSV时出错: {e}")
        return []


def merge_stock_lists(
    csv_values: List[str], json_stocks: List[str], format_codes: bool = True
) -> List[str]:
    """
    合并并去重两个股票列表

    参数:
        csv_values: 从CSV读取的值列表
        json_stocks: JSON中的股票列表
        format_codes: 是否统一格式化股票代码

    返回:
        合并去重后的排序股票列表
    """
    # 创建集合用于去重
    unique_stocks = set()

    # 添加CSV值
    for stock in csv_values:
        if format_codes:
            # 统一格式化为6位数字符串，不足补零
            stock = stock.zfill(6)
        unique_stocks.add(stock)

    # 添加JSON股票
    for stock in json_stocks:
        if format_codes:
            stock = stock.zfill(6)
        unique_stocks.add(stock)

    # 转换为列表并排序
    sorted_stocks = sorted(unique_stocks)
    return sorted_stocks


def runTask(list):
    for item in list:
        TradingState.symbol = item["symbol"]
        TradingState.start_date = item["start_date"]
        TradingState.end_date = item["end_date"]
        TradingState.file_date = item["file_date"]
        trading_flow = TradingFlow()
        trading_flow.kickoff()


def kickoff():
    # csv_values = read_csv_values("trade.csv")
    # csv_values_symbols = list(map(create_object, csv_values))
    # runTask(csv_values_symbols)
    # result = (
    #     StockScreenerCrew()
    #     .crew()
    #     .kickoff(inputs={"end_date": datetime.now().strftime("%Y%m%d")})
    # )
    # extract_json = extract_json_from_markdown(result.raw)
    # stock_list = extract_json.get("stock_list", [])
    # new_list = [stock for stock in stock_list if stock not in csv_values]
    new_list = ["600031"]
    symbols = list(map(create_object, new_list))
    runTask(symbols)


def plot():
    trading_flow = TradingFlow()
    trading_flow.plot("trading_flow_plot")


if __name__ == "__main__":
    kickoff()
